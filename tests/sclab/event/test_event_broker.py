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
