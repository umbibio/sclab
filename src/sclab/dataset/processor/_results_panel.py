from ipywidgets import GridBox, Layout, Stack, ToggleButtons, link

from sclab.event import EventBroker, EventClient


class _Results:
    namespace: str


class ResultsPanel(GridBox, EventClient):
    available_results: ToggleButtons
    results_stack: Stack

    events: list[str] = [
        # "rslt_add_result",
        # "rslt_remove_result",
    ]

    def __init__(
        self,
        broker: EventBroker,
    ):
        EventClient.__init__(self, broker)

        self.available_results = ToggleButtons(options={})
        self.results_stack = Stack([])

        link(
            (self.available_results, "value"),
            (self.results_stack, "selected_index"),
        )

        GridBox.__init__(
            self,
            [self.available_results, self.results_stack],
            layout=Layout(
                width="100%",
                grid_template_columns="150px auto",
                grid_template_areas=""" "available-results selected-results_stack" """,
                border="0px solid black",
            ),
        )

    def add_result(self, results: _Results):
        current_stack = list(self.results_stack.children)
        namespace = results.namespace

        options: dict[str, int] = self.available_results.options
        options = options.copy()
        idx = options.get(namespace, len(options))
        options[namespace] = idx

        if len(current_stack) < idx + 1:
            current_stack.append(results)
        else:
            current_stack[idx] = results

        self.results_stack.children = tuple(current_stack)
        self.available_results.options = options

    def remove_result(self, name: str):
        options: dict[str, int] = self.available_results.options
        options = options.copy()
        idx = options.pop(name)

        current_stack = list(self.results_stack.children)
        current_stack.pop(idx)

        current_selection = self.results_stack.selected_index
        if (
            current_selection is not None
            and current_selection > 0
            and current_selection == idx
        ):
            idx = current_selection - 1
            self.results_stack.selected_index = idx

        self.results_stack.children = tuple(current_stack)
        self.available_results.options = options
        self.available_results.value = idx
