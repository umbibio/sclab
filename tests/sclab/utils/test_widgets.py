from ipywidgets import HTML, Stack, Tab, VBox

from sclab.utils import LazyOutput


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return [{"text/html": f"<b>{self.n}</b>"}]


def shown_html(out):
    return [o["data"]["text/html"] for o in out.outputs]


def test_hidden_until_show_when():
    render = Counter()
    out = LazyOutput(render)
    out.invalidate()
    assert render.n == 0 and out.outputs == () and out.stale
    out.show_when()  # no path: always shown
    assert render.n == 1 and shown_html(out) == ["<b>1</b>"]
    out.invalidate()
    assert render.n == 2 and shown_html(out) == ["<b>2</b>"]


def test_nested_path_and_stale_flag():
    render = Counter()
    out = LazyOutput(render)
    page = VBox([out])
    stack = Stack([HTML(), page], selected_index=0)
    tab = Tab([HTML(), stack])
    out.show_when((tab, stack), (stack, page))
    assert tab.selected_index == 0 and render.n == 0

    tab.selected_index = 1  # stack shows the other page
    assert render.n == 0 and not out.shown
    stack.selected_index = 1
    assert out.shown and render.n == 1 and out.outputs

    tab.selected_index = 0  # hidden: unloaded, kept rendered
    assert out.outputs == () and not out.stale
    tab.selected_index = 1
    assert render.n == 1 and shown_html(out) == ["<b>1</b>"]

    tab.selected_index = 0
    out.invalidate()
    out.invalidate()
    assert render.n == 1 and out.stale  # hidden: only marked
    tab.selected_index = 1
    assert render.n == 2 and shown_html(out) == ["<b>2</b>"]


def test_children_change_hides():
    render = Counter()
    out = LazyOutput(render)
    page = VBox([out])
    stack = Stack([page], selected_index=0)
    out.show_when((stack, page))
    assert out.shown
    stack.children = (HTML(),)  # replaced, e.g. a newer result
    assert not out.shown and out.outputs == ()


def test_keep_loaded_when_unload_is_false():
    render = Counter()
    out = LazyOutput(render, unload=False)
    tab = Tab([HTML(), out])
    out.show_when((tab, out))
    tab.selected_index = 1
    tab.selected_index = 0
    assert shown_html(out) == ["<b>1</b>"]
    out.invalidate()  # stale content stays hidden until shown again
    assert render.n == 1
    tab.selected_index = 1
    assert render.n == 2 and shown_html(out) == ["<b>2</b>"]


def test_show_when_rewires():
    render = Counter()
    out = LazyOutput(render)
    old, new = Tab([HTML(), out]), Tab([out, HTML()])
    out.show_when((old, out))
    out.show_when((new, out))
    assert out.shown and render.n == 1
    old.selected_index = 1  # no longer observed
    new.selected_index = 1
    assert not out.shown and render.n == 1


def test_render_error_is_shown_and_retried():
    calls = []

    def render():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("boom")
        return [{"text/plain": "ok"}]

    out = LazyOutput(render)
    tab = Tab([HTML(), out])
    out.show_when((tab, out))
    tab.selected_index = 1
    assert "ValueError: boom" in out.outputs[0]["data"]["text/plain"]
    assert out.stale
    tab.selected_index = 0
    tab.selected_index = 1
    assert out.outputs[0]["data"]["text/plain"] == "ok" and not out.stale
