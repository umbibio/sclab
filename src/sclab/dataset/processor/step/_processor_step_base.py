import traceback
from typing import Any

from IPython.display import HTML, Markdown, display
from ipywidgets import Button, Output, VBox
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_description import DescriptionWidget

from ....event import EventClient
from .._processor import Processor
from .._results_panel import _Results


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
        self.run()

    def function(self, *pargs, **kwargs):
        raise NotImplementedError

    def run(self, **extra_params):
        self.output.clear_output(wait=False)
        try:
            # extra params will override fixed and variable params
            params = {**self.fixed_params, **self.variable_params, **extra_params}

            self.run_button.disabled = True
            self.run_button.button_style = "warning"
            self.run_button.description = "..."

            self.broker.publish(f"step_{self.name}_started")
            self.function(**params)

            self.parent.append_to_step_history(self.description, params)
            info = dict(status="success")
            self.run_button.button_style = "success"

        except Exception as e:
            self.run_button.button_style = "danger"

            info = dict(status="failed", error=e, traceback=traceback.format_exc())
            self.broker.exceptions_log.append(traceback.format_exc())
            with self.broker.exceptions_output.output:
                display(HTML(f"<pre>{traceback.format_exc()}</pre>"))
            self.update_output(f"{type(e)}: {e}")

        finally:
            self.run_button.description = self.run_button_description
            self.run_button.disabled = False
            self.broker.publish(f"step_{self.name}_ended", **info)

    def update_output(self, message: str | Any | None, clear: bool = True):
        if clear:
            self.output.clear_output(wait=True)

        if isinstance(message, str):
            message = Markdown(message)

        elif message is None:
            message = Markdown("")

        with self.output:
            display(message)
