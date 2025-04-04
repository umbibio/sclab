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
