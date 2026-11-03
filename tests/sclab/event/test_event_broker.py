from typing import Any, List
from uuid import UUID

import pytest
from ipywidgets import Output, Tab

from sclab.event._broker import EventBroker
from sclab.event._client import EventClient
from sclab.event._utils import LogOutput


class TheEventClient(EventClient):
    """Test implementation of EventClient with callback logging."""

    events: List[str] = ["test_event", "another_event"]
    preemptions: dict[str, list[str]] = {"test_event": ["another_event"]}

    def __init__(self, broker: EventBroker) -> None:
        super().__init__(broker)  # broker will be set later

        self.call_log: List[tuple[str, tuple, dict]] = []  # (event_name, args, kwargs)

    def test_event_callback(self, *args: Any, **kwargs: Any) -> None:
        """A test callback that logs its calls"""
        self.call_log.append(("test_event", args, kwargs))
        self.broker.publish("another_event", *args, **kwargs)

    def another_event_callback(self, *args: Any, **kwargs: Any) -> None:
        """Another test callback that logs its calls"""
        self.call_log.append(("another_event", args, kwargs))


@pytest.fixture
def broker() -> EventBroker:
    """Create a broker for testing."""
    return EventBroker()


@pytest.fixture
def test_client(broker: EventBroker) -> TheEventClient:
    """Create a test client with logging callbacks."""
    return TheEventClient(broker)


def test_broker_initialization(broker: EventBroker) -> None:
    """Test that broker is properly initialized."""
    assert isinstance(broker.id, UUID)
    assert broker.clients == []
    assert broker.available_events == set()
    assert broker.preemptions == {}
    assert broker._subscribers == {}
    assert broker._disabled_events == []
    assert broker.depth == 0

    # Check UI components
    assert isinstance(broker.std_output, Output)
    assert isinstance(broker.execution_output, LogOutput)
    assert isinstance(broker.exceptions_output, LogOutput)
    assert isinstance(broker.event_log, Output)
    assert isinstance(broker.logs_tab, Tab)


def test_client_registration(broker: EventBroker, test_client: TheEventClient) -> None:
    """Test client registration process."""
    broker.register_client(test_client)

    # Check client is registered
    assert test_client in broker.clients
    assert test_client.broker is broker

    # Check events are registered
    assert broker.available_events == set(test_client.events)

    # Check preemptions are registered
    assert broker.preemptions == test_client.preemptions


def test_client_unregistration(
    broker: EventBroker, test_client: TheEventClient
) -> None:
    """Test client unregistration process."""
    broker.register_client(test_client)
    broker.unregister_client(test_client)

    assert test_client not in broker.clients
    assert not broker.available_events
    assert not broker.preemptions


def test_update_subscriptions(broker: EventBroker, test_client: TheEventClient) -> None:
    """Test that update_subscriptions correctly wires up callbacks."""
    broker.register_client(test_client)
    broker.update_subscriptions()

    # Check that both test callbacks were subscribed
    assert "test_event" in broker._subscribers
    assert "another_event" in broker._subscribers
    assert test_client.test_event_callback in broker._subscribers["test_event"]
    assert test_client.another_event_callback in broker._subscribers["another_event"]


def test_publish_success(broker: EventBroker, test_client: TheEventClient) -> None:
    """Test successful event publishing."""
    broker.register_client(test_client)
    broker.update_subscriptions()

    # Publish test event
    test_args = (1, 2, 3)
    test_kwargs = {"key": "value"}
    broker.publish("test_event", *test_args, **test_kwargs)

    # Verify callback was called with correct arguments
    assert len(test_client.call_log) == 1
    event_name, args, kwargs = test_client.call_log[0]
    assert event_name == "test_event"
    assert args == test_args
    assert kwargs == test_kwargs


def test_publish_invalid_event(broker: EventBroker) -> None:
    """Test that publishing an invalid event raises ValueError."""
    with pytest.raises(ValueError):
        broker.publish("invalid_event")


def test_preemption(broker: EventBroker, test_client: TheEventClient) -> None:
    """Test that event preemption works correctly."""
    broker.register_client(test_client)
    broker.update_subscriptions()

    # Publish test_event which should preempt another_event
    broker.publish("test_event")

    # Get the HTML content from the execution output
    # The content should show that another_event was disabled
    assert any(
        "Disabled Event: another_event" in str(msg) for msg in broker.execution_log
    )


def test_coalesce_runs_immediately_outside_batch(broker: EventBroker) -> None:
    calls = []
    render = broker.coalesce(lambda x: calls.append(x) or "done")

    assert render(1) == "done"
    assert calls == [1]


def test_batch_coalesces_with_latest_args_in_first_seen_order(
    broker: EventBroker,
) -> None:
    calls = []
    a = broker.coalesce(lambda *args, **kw: calls.append(("a", args, kw)))
    b = broker.coalesce(lambda *args, **kw: calls.append(("b", args, kw)))

    with broker.batch():
        a(1)
        b(2)
        with broker.batch():  # nested: only the outermost exit flushes
            a(3, k=4)
        assert calls == []

    assert calls == [("a", (3,), {"k": 4}), ("b", (2,), {})]


def test_coalesce_keys_per_bound_instance(broker: EventBroker) -> None:
    calls = []

    class Thing:
        def __init__(self, name):
            self.name = name
            self.render = broker.coalesce(self.render)

        def render(self):
            calls.append(self.name)

    x, y = Thing("x"), Thing("y")
    with broker.batch():
        x.render()
        y.render()
        x.render()

    assert calls == ["x", "y"]
    assert x.render.__self__ is x


def test_publish_opens_a_batch(broker: EventBroker) -> None:
    calls = []
    render = broker.coalesce(lambda: calls.append("render"))

    class Client(EventClient):
        events = ["outer", "inner"]

        def outer_callback(self):
            render()
            self.broker.publish("inner")
            assert calls == []

        def inner_callback(self):
            render()

    client = Client(broker)  # noqa: F841 - keep subscribed
    broker.publish("outer")
    assert calls == ["render"]
    assert not broker.exceptions_log


def test_flush_exception_is_logged_and_others_still_run(broker: EventBroker) -> None:
    calls = []

    def boom():
        raise RuntimeError("boom")

    with broker.batch():
        broker.coalesce(boom)()
        broker.coalesce(lambda: calls.append("ok"))()

    assert calls == ["ok"]
    assert len(broker.exceptions_log) == 1
    assert "boom" in broker.exceptions_log[0]


def test_calls_made_during_flush_run_in_the_same_flush(broker: EventBroker) -> None:
    calls = []
    second = broker.coalesce(lambda: calls.append("second"))

    def first():
        calls.append("first")
        with broker.batch():  # e.g. a publish from inside a render
            second()
        assert calls == ["first"]

    with broker.batch():
        broker.coalesce(first)()

    assert calls == ["first", "second"]
    assert not broker.exceptions_log
    assert broker._batch_depth == 0


def test_self_requeueing_flush_stops_and_logs(broker: EventBroker) -> None:
    broker.max_flush_calls = 5
    calls = []

    def loop():
        calls.append(1)
        render()

    render = broker.coalesce(loop)
    with broker.batch():
        render()

    assert len(calls) == 5
    assert not broker._pending
    assert "did not settle" in broker.exceptions_log[-1]


def test_flush_runs_one_pending_call_early(broker: EventBroker) -> None:
    calls = []
    a = broker.coalesce(lambda: calls.append("a"))
    b = broker.coalesce(lambda: calls.append("b"))

    with broker.batch():
        b()
        a()
        broker.flush(a)
        assert calls == ["a"]
        broker.flush(a)  # nothing pending: no-op
        assert calls == ["a"]

    assert calls == ["a", "b"]


def test_coalescing_disabled_runs_immediately(broker: EventBroker) -> None:
    broker.coalescing = False
    calls = []
    render = broker.coalesce(lambda: calls.append(1))

    with broker.batch():
        render()
        render()
        assert calls == [1, 1]
