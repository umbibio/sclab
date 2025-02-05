from ipywidgets.widgets import Button, Layout, Output, VBox


class LogOutput(VBox):
    def __init__(self):
        self.clear_button = Button(description="Clear", button_style="warning")
        self.output = Output(layout=Layout(width="auto", height="500px"))

        super().__init__(children=[self.clear_button, self.output])

        self.clear_button.on_click(self._clear)

    def _clear(self, _):
        self.output.clear_output()
