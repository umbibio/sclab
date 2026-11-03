from ipywidgets import Box

from sclab.dataset.processor._results_panel import ResultsPanel
from sclab.event import EventBroker


class Result(Box):
    def __init__(self, namespace: str):
        super().__init__()
        self.namespace = namespace


def make_panel(*namespaces: str) -> ResultsPanel:
    panel = ResultsPanel(EventBroker())
    for ns in namespaces:
        panel.add_result(Result(ns))
    return panel


class TestCategoryDropdownKeepsValue:
    def test_selection_survives_adding_a_result(self):
        panel = make_panel("a", "b")
        panel.available_results.value = 1
        panel.add_result(Result("c"))
        assert panel.available_results.value == 1
        assert panel.results_stack.selected_index == 1

    def test_selection_survives_replacing_a_result(self):
        panel = make_panel("a", "b")
        panel.available_results.value = 1
        panel.add_result(Result("b"))
        assert panel.available_results.value == 1

    def test_frontend_gets_a_single_update(self):
        panel = make_panel("a", "b")
        panel.available_results.value = 1
        sent = []
        panel.available_results.send_state = lambda key=None: sent.append(key)
        panel.add_result(Result("c"))
        assert len(sent) == 1
