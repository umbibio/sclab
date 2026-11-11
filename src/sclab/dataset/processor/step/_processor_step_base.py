import functools
import inspect
import traceback
from html import escape
from typing import Any

from anndata import AnnData
from IPython.display import HTML, Markdown, display
from ipywidgets import HTML as HTMLWidget
from ipywidgets import Button, Output, VBox
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_description import DescriptionWidget

from ...._compat import missing
from ....event import EventClient, StepCancelled
from .._processor import Processor
from .._results_panel import _Results

_ENTRIES = ("obsm", "varm", "obsp", "varp", "layers", "uns")
_MISSING = object()
# dataset events after which a step's inputs may have become (in)valid
_DATASET_EVENTS = (
    "dset_metadata_change",
    "dset_data_dict_change",
    "dset_anndata_layers_change",
    "dset_anndata_neighbors_change",
    "dset_total_rows_change",
    "dset_total_vars_change",
    "dset_var_dataframe_change",
    "dset_state_change",  # after every step run
    "dset_counts_layer_change",
)


def error_html(error: BaseException | str) -> str:
    """A failure as shown in a step's panel: its first line, the rest in Logs."""
    if isinstance(error, BaseException):
        error = f"{type(error).__name__}: {error}"
    text = error.strip().split("\n")[0][:200]
    return f"<span style='color:#d32f2f'>{escape(text, quote=False)} (details in Logs)</span>"


def _shallow_copy(adata: AnnData) -> tuple[AnnData, dict]:
    """An AnnData sharing ``adata``'s arrays, and its entries for :func:`_apply_changes`.

    Setting or deleting entries (or obs/var columns) of the copy leaves
    ``adata`` unchanged; modifying shared arrays in place does not.
    """
    entries = {key: dict(getattr(adata, key)) for key in _ENTRIES}
    obs, var = adata.obs.copy(), adata.var.copy()
    # copies: AnnData may keep the dicts it is given as its own
    copies = {key: dict(value) for key, value in entries.items()}
    work = AnnData(X=adata.X, obs=obs, var=var, raw=adata.raw, **copies)
    return work, entries


def _apply_changes(adata: AnnData, work: AnnData, entries: dict):
    """Set in ``adata`` what ``work`` (from :func:`_shallow_copy`) set or deleted."""
    if work.X is not adata.X:
        adata.X = work.X
    for key, before in entries.items():
        old, new = getattr(adata, key), getattr(work, key)
        for k in before.keys() - new.keys():
            old.pop(k, None)
        for k, v in new.items():
            if before.get(k, _MISSING) is not v:
                old[k] = v
    for key in ("obs", "var"):
        old, new = getattr(adata, key), getattr(work, key)
        old.drop(columns=old.columns.difference(new.columns), inplace=True)
        for c in new.columns:
            if (
                c not in old
                or new[c].dtype != old[c].dtype
                or not new[c].equals(old[c])
            ):
                old[c] = new[c]


class ProcessorStepBase(EventClient):
    """Base class for interactive processor steps in the SCLab dashboard.

    Subclass this to create a custom analysis step that integrates with the
    :class:`~sclab.dataset.processor.Processor` panel. Each step exposes a
    widget-based control panel and publishes events when it starts and
    finishes running.

    Subclasses must define the following class attributes:

    - ``name`` (str): Unique step identifier (e.g. ``"my_step"``).
    - ``description`` (str): Human-readable label shown in the UI.

    Subclasses must override:

    - :meth:`function`: The analysis logic to execute when the step is run.

    Subclasses may override :meth:`validate` to disable the Run button while
    required inputs are missing.

    Parameters
    ----------
    parent : Processor
        The parent :class:`~sclab.dataset.processor.Processor` that owns
        this step.
    fixed_params : dict
        Parameters that are fixed at construction time and passed directly
        to :meth:`function` at runtime (not exposed as widgets).
    variable_controls : dict
        Mapping of parameter names to ipywidgets
        (:class:`~ipywidgets.widgets.widget_description.DescriptionWidget`
        or :class:`~ipywidgets.widgets.valuewidget.ValueWidget`). Each
        widget's ``.value`` is passed to :meth:`function` at runtime under
        its key.
    results : _Results or None, optional
        Optional results panel to display after the step runs. If provided,
        it is registered with the parent's results panel. Default is None.

    Attributes
    ----------
    name : str
        Step identifier. Must be set as a class variable in subclasses.
    description : str
        Human-readable step name. Must be set as a class variable.
    order : int
        Integer used to sort steps in the processor accordion. Lower values
        appear first. Default is 1000.
    background : bool
        If True, the Run button runs the step as a background job (see
        :mod:`sclab.event._jobs`): the dashboard stays responsive and shows
        a progress panel with Cancel and Interrupt. :meth:`function` should
        then be a generator doing its heavy work through :meth:`compute`.
        :meth:`run` stays synchronous. Default is False.
    controls : VBox
        The assembled widget panel for this step.
    run_button : Button
        The button that triggers :meth:`run`.
    status : ipywidgets.HTML
        Below the Run button: why the step cannot run (see :meth:`validate`)
        and the last error, if any (cleared by a successful run or a change
        of the step's controls).
    tool_label : ipywidgets.HTML
        Below the Run button: a muted line naming the tool the step runs
        (see :meth:`tool`); hidden when :meth:`tool` returns "".
    output : Output
        Widget area for displaying step output messages.
    """

    events: list[str] = None
    parent: Processor
    name: str = None
    description: str = None
    fixed_params: dict[str, Any]
    variable_controls: dict[str, DescriptionWidget | ValueWidget]
    output: Output
    run_button: Button
    controls_list: list[DescriptionWidget | ValueWidget | Button]
    controls: VBox
    results: _Results | None
    order: int = 1000
    background: bool = False

    run_button_description = "Run"

    def __init__(
        self,
        parent: Processor,
        fixed_params: dict[str, Any],
        variable_controls: dict[str, DescriptionWidget | ValueWidget],
        results: _Results | None = None,
    ):
        assert self.name
        assert self.description

        self.parent = parent
        self.fixed_params = fixed_params
        self.variable_controls = variable_controls

        self.events = [
            f"step_{self.name}_started",
            f"step_{self.name}_ended",
        ]

        self.output = Output()
        self.status = HTMLWidget()
        self.tool_label = HTMLWidget(layout={"display": "none"})
        self._running = False
        self._reason: str | None = None
        self._error: tuple[str, dict] | None = None  # (html, control params)
        self.run_button = Button(
            description=self.run_button_description, button_style="primary"
        )
        self.run_button.on_click(self.button_callback)

        self.controls_list = [
            *self.variable_controls.values(),
            self.run_button,
            self.status,
            self.output,
        ]
        self.make_controls()

        if results is not None:
            self.results = results
            parent.results_panel.add_result(self.results)
        super().__init__(parent.broker)

        # the first refresh comes from Processor.add_steps, once the step is built
        self.refresh = self.broker.coalesce(self.guarded(self._refresh))
        for control in self.variable_controls.values():
            control.observe(self.refresh, "value")
        for event in _DATASET_EVENTS:
            self.broker.subscribe(event, self.refresh)

    def make_controls(self):
        # here, so that subclasses building their own controls_list get it too
        at = self.controls_list.index(self.run_button) + 1
        self.controls_list.insert(at, self.tool_label)
        for control in self.controls_list:
            control.layout.width = "98%"
            self.parent.all_controls_list.append(control)

        self.controls = VBox(children=self.controls_list)

    @property
    def variable_params(self):
        return {key: control.value for key, control in self.variable_controls.items()}

    def _control_params(self) -> dict:
        return {**self.fixed_params, **self.variable_params}

    def validate(self, params: dict) -> str | None:
        """Why the step cannot run with ``params``, or None if it can.

        ``params`` are the merged params :meth:`function` would get. Override
        to check the inputs the step needs, e.g. a chosen column or a computed
        neighbor graph, returning a short reason for the user ("Choose a
        'Group by' column"). While it returns a reason the Run button is
        disabled and shows it, and :meth:`run` raises ``ValueError``. It is
        re-checked when the step's controls or the dataset change, so it must
        be cheap. Leave rarer cases to the library's own errors.
        """
        return None

    def unavailable(self, params: dict) -> str | None:
        """Why a run with ``params`` cannot happen here for lack of a package.

        None if it can. Unlike :meth:`validate` it must not depend on the
        data: replaying a history (:meth:`Processor.apply_step_history`)
        checks every recorded entry with it before applying any. Override
        (calling ``super()``) for packages a method, flavor or backend needs;
        :meth:`validate` may return it too. The default is
        :meth:`scanpy_missing`.
        """
        return self.scanpy_missing(params)

    @staticmethod
    def scanpy_missing(params: dict, *features: str) -> str | None:
        """For a run recorded with ``backend="scanpy"``: the reason if scanpy,
        or another of the ``features`` it ran with, is missing. A replay does not
        switch to sclab's native port: its results may differ."""
        if params.get("backend") != "scanpy":
            return None
        for feature in ("scanpy", *features):
            if reason := missing(feature):
                return (
                    f"{reason} (recorded with backend 'scanpy', not replayed with "
                    "sclab's native port as results may differ: run the step "
                    "again to use it)"
                )
        return None

    def tool(self, params: dict) -> str:
        """The tool the step runs with ``params``, shown below the Run button.

        Language and package -> function, e.g. ``"Python · scanpy.pp.pca"``
        or ``"R · scDblFinder::scDblFinder"``. Override to describe the
        method, flavor or backend chosen in ``params`` (the merged params
        :meth:`function` would get); it is re-rendered with :meth:`validate`,
        so it must be cheap. The default, "", shows nothing.
        """
        return ""

    def _check(self, params: dict):
        if (reason := self.validate(params)) is not None:
            raise ValueError(reason)

    def _refresh(self, *_, **__):
        if self._running:  # _run refreshes when done
            return
        params = self._control_params()
        self._reason = self.validate(params)
        tool = self.tool(params)
        self.tool_label.value = (
            f"<span style='color:#888;font-size:0.8em'>{escape(tool)}</span>"
        )
        self.tool_label.layout.display = None if tool else "none"
        if self._error is not None and self._error[1] != params:
            self._error = None  # the inputs changed
        self.broker.jobs.set_disabled(self.run_button, self._reason is not None)
        self._render_status()

    def _render_status(self):
        parts = []
        if self._reason is not None:
            parts.append(
                f"<span style='color:#b26a00'>{escape(self._reason, quote=False)}</span>"
            )
        if self._error is not None:
            parts.append(self._error[0])
        self.status.value = "<br>".join(parts)

    def show_error(self, error: BaseException | str):
        """Show ``error`` below the Run button until the step's controls change."""
        self._error = (error_html(error), self._control_params())
        self._render_status()

    def guarded(self, fn: callable) -> callable:
        """``fn``, showing its exceptions in the panel (and Logs) instead of raising.

        For widget callbacks (``observe``, ``on_click``), whose exceptions
        would otherwise reach only the kernel's log.
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                self.broker._log_exception(e)
                self.show_error(e)

        return wrapper

    def button_callback(self, _: Button | None = None):
        try:
            if not self.background:
                self.run()
                return
            self._check(self._control_params())
            lock = self.parent.lockable_controls()
            self.broker.jobs.start(self._run({}), self.description, lock=lock)
        except Exception as e:  # not runnable now, e.g. another job runs
            self.broker._log_exception(e)
            self.show_error(e)

    def function(self, *pargs, **kwargs):
        raise NotImplementedError

    def compute(self, fn: callable, *args, **kwargs):
        """The heavy work of :meth:`function`: ``fn(adata, *args, **kwargs)``.

        Use as ``result = yield from self.compute(fn, ...)`` (making
        :meth:`function` a generator). ``fn`` gets a shallow copy of the
        dataset's adata (see :func:`_shallow_copy`) and runs in the job's
        worker thread for a background run, inline for :meth:`run`. What it
        sets or deletes in the copy is then applied to the adata on the main
        thread, so a cancelled or failed run leaves the adata unchanged.
        ``fn`` must not touch widgets nor modify arrays in place, nor change
        the shape.
        """
        adata = self.parent.dataset.adata
        work, entries = _shallow_copy(adata)
        result = yield functools.partial(fn, work, *args, **kwargs)
        _apply_changes(adata, work, entries)
        return result

    def resolved_params(self, params: dict) -> dict:
        """Return the params to record in the step history.

        ``params`` are the merged fixed, widget and extra params passed to
        :meth:`function`. The default fills in :meth:`function`'s defaults for
        params not given. Override (calling ``super()``) to add implicit
        inputs the step reads at run time, e.g. ``self.parent.batch_key``, or
        to resolve placeholders such as ``layer=None`` -> the counts layer.
        """
        defaults = {
            name: p.default
            for name, p in inspect.signature(self.function).parameters.items()
            if p.default is not p.empty and name not in params
        }
        return {**params, **defaults}

    def to_code(self, params: dict) -> str | None:
        """Python code reproducing a run of this step with ``params``.

        ``params`` are the recorded params of a history entry (see
        :meth:`resolved_params`); values equal to the dashboard's counts layer
        or batch key arrive as the variable names ``counts_layer`` /
        ``batch_key`` (see :mod:`sclab.dataset.processor._codegen`, whose
        ``call`` helper renders a call with explicit keyword arguments). Return
        None if the step does not change the data (e.g. a plot). Steps that do
        not override this get a "no code available" comment in the script.
        """
        return None

    def run(self, **extra_params):
        """Run the step now, on this thread, also if it is a background step.

        One flush of coalesced renders after the whole step. Refused (raises
        ``RuntimeError``) while a background job runs, and (``ValueError``)
        while :meth:`validate` gives a reason; nothing is recorded then.
        """
        self.broker.jobs.run(self._run(extra_params))

    def _run(self, extra_params: dict):
        """The run as a job: :meth:`run` and the Run button drive it."""
        # extra params will override fixed and variable params
        params = {**self.fixed_params, **self.variable_params, **extra_params}
        self._check(params)
        self.output.clear_output(wait=False)
        self._error = None
        self._render_status()
        shape_before = self.parent.dataset.adata.shape
        recorded = params
        info = {}
        n_exceptions = len(self.broker.exceptions_log)
        try:
            self._running = True
            self.run_button.disabled = True
            self.run_button.button_style = "warning"
            self.run_button.description = "..."

            self.broker.publish(f"step_{self.name}_started")
            recorded = self.resolved_params(params)
            result = self.function(**params)
            if inspect.isgenerator(result):
                yield from result

            self.parent.append_to_step_history(
                self.description, recorded, shape_before=shape_before
            )
            info = dict(status="success")
            self.run_button.button_style = "success"

        except StepCancelled:
            self.run_button.button_style = ""
            self.parent.append_to_step_history(
                self.description,
                recorded,
                status="cancelled",
                shape_before=shape_before,
            )
            info = dict(status="cancelled")
            self.update_output("Cancelled.")

        except Exception as e:
            self.run_button.button_style = "danger"
            self.parent.append_to_step_history(
                self.description,
                recorded,
                status="failed",
                error=f"{type(e).__name__}: {e}".split("\n")[0][:200],
                shape_before=shape_before,
            )

            info = dict(status="failed", error=e, traceback=traceback.format_exc())
            self.broker.exceptions_log.append(traceback.format_exc())
            with self.broker.exceptions_output.output:
                display(HTML(f"<pre>{traceback.format_exc()}</pre>"))
            self.show_error(e)

        finally:
            self._running = False
            self.run_button.description = self.run_button_description
            self.broker.publish(f"step_{self.name}_ended", **info)
            self.refresh()  # re-enables the button, if the inputs are still valid
            if self.broker._batch_depth == 1:  # outermost: run the deferred renders now
                self.broker.flush()
            # a failed redraw or event callback does not leave the button green
            failed = len(self.broker.exceptions_log) > n_exceptions
            if failed and self.run_button.button_style == "success":
                self.run_button.button_style = "danger"
                self.show_error("The step ran, but updating the dashboard failed")

    def update_output(self, message: str | Any | None, clear: bool = True):
        if clear:
            self.output.clear_output(wait=True)

        if isinstance(message, str):
            message = Markdown(message)

        elif message is None:
            message = Markdown("")

        with self.output:
            display(message)
