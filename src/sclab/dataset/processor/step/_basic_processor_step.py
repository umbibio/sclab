import traceback
from typing import Any

from IPython.display import display
from ipywidgets.widgets import (
    HTML,
    Button,
    Output,
    VBox,
)
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_description import DescriptionWidget

from ....event import EventClient
from .. import Processor


class BasicProcessorStep(EventClient):
    events: list[str] = None
    parent: Processor
    description: str
    function_name: str
    function: callable
    fixed_params: dict[str, Any]
    variable_controls: dict[str, DescriptionWidget | ValueWidget]
    output: Output
    run_button: Button
    controls: VBox

    def __init__(
        self,
        parent: Processor,
        description: str,
        function: callable,
        fixed_params: dict[str, Any] = {},
        variable_controls: dict[str, DescriptionWidget | ValueWidget] = {},
        use_run_button: bool = True,
    ):
        self.parent = parent
        self.description = description
        self.function = function
        self.function_name = function.__name__

        self.events = [
            f"step_{self.function_name}_started",
            f"step_{self.function_name}_ended",
        ]

        self.fixed_params = fixed_params
        self.variable_controls = variable_controls
        self.output = Output()

        controls = []
        for control in self.variable_controls.values():
            control.layout.width = "98%"
            self.parent.all_controls_list.append(control)
            controls.append(control)

        self.use_run_button = use_run_button
        if use_run_button:
            self.run_button = Button(description="Run", button_style="primary")
            self.run_button.on_click(self.callback)
            controls.append(self.run_button)

        controls.append(self.output)
        self.controls = VBox(controls)
        super().__init__(parent.broker)

    @property
    def variable_params(self):
        return {key: control.value for key, control in self.variable_controls.items()}

    def callback(self, _: Button | None = None):
        self.run()

    def run(self, **extra_params):
        try:
            # extra params will override fixed and variable params
            params = {**self.fixed_params, **self.variable_params, **extra_params}

            if self.use_run_button:
                self.run_button.disabled = True
                self.run_button.button_style = "warning"
                self.run_button.description = "Running..."
            self.broker.publish(f"step_{self.function_name}_started")

            self.function(**params)
            self.parent.append_to_step_history(self.description, params)

            if self.use_run_button:
                self.run_button.button_style = "success"

            self.broker.publish(f"step_{self.function_name}_ended", status="success")

        except Exception as e:
            if self.use_run_button:
                self.run_button.button_style = "danger"

            info = dict(status="failed", error=e, traceback=traceback.format_exc())
            self.broker.publish(f"step_{self.function_name}_ended", **info)
            self.broker.exceptions_log.append(traceback.format_exc())

            with self.broker.exceptions_output.output:
                display(HTML(f"<pre>{traceback.format_exc()}</pre>"))

        finally:
            if self.use_run_button:
                self.run_button.description = "Run"
                self.run_button.disabled = False
