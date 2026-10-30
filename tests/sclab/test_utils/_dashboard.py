"""Helpers for driving the dashboard's ipywidgets headlessly in tests.

``ProcessorStepBase.run`` and ``EventBroker.publish`` catch every exception,
so a failing step or event callback never reaches pytest. These helpers turn
those swallowed errors into assertion failures.
"""

import json

from ipywidgets import Button, Widget
from ipywidgets.widgets.widget import _remove_buffers

from sclab import SCLabDashboard


def new_exceptions(dash: SCLabDashboard, since: int) -> list[str]:
    return dash.broker.exceptions_log[since:]


def run_step(dash: SCLabDashboard, step_name: str, **params):
    """Run a step and assert it succeeded without any swallowed exception."""
    step = dash.processor.steps[step_name]
    n_hist = len(dash.processor.step_history)
    n_exc = len(dash.broker.exceptions_log)

    step.run(**params)

    errors = new_exceptions(dash, n_exc)
    assert not errors, f"{step_name!r} raised:\n" + "\n".join(errors)
    assert step.run_button.button_style == "success"
    assert len(dash.processor.step_history) == n_hist + 1
    assert dash.processor.step_history[-1]["step_description"] == step_name
    return step


def click(dash: SCLabDashboard, button: Button):
    """Click a button as a user would and assert no swallowed exception."""
    n_exc = len(dash.broker.exceptions_log)
    button.click()
    errors = new_exceptions(dash, n_exc)
    assert not errors, f"clicking {button.description!r} raised:\n" + "\n".join(errors)


def live_widget_ids() -> set[str]:
    return set(Widget._active_widgets)


def assert_widgets_serializable(exclude_ids: set[str] = frozenset()):
    """Assert every live widget's state can be sent to the frontend.

    Calls ``get_state()`` (trait serialization) and JSON-encodes the result,
    after splitting off binary buffers, which is what the comm layer does.
    """
    failures = []
    for model_id, widget in list(Widget._active_widgets.items()):
        if model_id in exclude_ids:
            continue
        try:
            state, _, _ = _remove_buffers(widget.get_state())
            json.dumps(state)
        except Exception as e:  # noqa: BLE001 - report every failing widget
            failures.append(f"{type(widget).__name__}: {type(e).__name__}: {e}")
    assert not failures, "unserializable widgets:\n" + "\n".join(failures)
