import functools
import inspect
import traceback
from typing import Any

from anndata import AnnData
from IPython.display import HTML, Markdown, display
from ipywidgets import Button, Output, VBox
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_description import DescriptionWidget

from ....event import EventClient, StepCancelled
from .._processor import Processor
from .._results_panel import _Results

_ENTRIES = ("obsm", "varm", "obsp", "varp", "layers", "uns")
_MISSING = object()


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
        self.run_button = Button(
            description=self.run_button_description, button_style="primary"
        )
        self.run_button.on_click(self.button_callback)

        self.controls_list = [
            *self.variable_controls.values(),
            self.run_button,
            self.output,
        ]
        self.make_controls()

        if results is not None:
            self.results = results
            parent.results_panel.add_result(self.results)
        super().__init__(parent.broker)

    def make_controls(self):
        for control in self.controls_list:
            control.layout.width = "98%"
            self.parent.all_controls_list.append(control)

        self.controls = VBox(children=self.controls_list)

    @property
    def variable_params(self):
        return {key: control.value for key, control in self.variable_controls.items()}

    def button_callback(self, _: Button | None = None):
        if not self.background:
            self.run()
            return
        lock = self.parent.lockable_controls()
        self.broker.jobs.start(self._run({}), self.description, lock=lock)

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
        ``RuntimeError``) while a background job runs.
        """
        self.broker.jobs.run(self._run(extra_params))

    def _run(self, extra_params: dict):
        """The run as a job: :meth:`run` and the Run button drive it."""
        self.output.clear_output(wait=False)
        shape_before = self.parent.dataset.adata.shape
        recorded = extra_params
        info = {}
        try:
            # extra params will override fixed and variable params
            params = {**self.fixed_params, **self.variable_params, **extra_params}
            recorded = params

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
            self.update_output(f"{type(e)}: {e}")

        finally:
            self.run_button.description = self.run_button_description
            self.run_button.disabled = False
            self.broker.publish(f"step_{self.name}_ended", **info)
            if self.broker._batch_depth == 1:  # outermost: run the deferred renders now
                n_exceptions = len(self.broker.exceptions_log)
                self.broker.flush()
                # so that a failed redraw does not leave the button green
                failed = len(self.broker.exceptions_log) > n_exceptions
                if failed and self.run_button.button_style == "success":
                    self.run_button.button_style = "danger"
                    self.update_output("The step ran, but redrawing failed; see Logs.")

    def update_output(self, message: str | Any | None, clear: bool = True):
        if clear:
            self.output.clear_output(wait=True)

        if isinstance(message, str):
            message = Markdown(message)

        elif message is None:
            message = Markdown("")

        with self.output:
            display(message)
