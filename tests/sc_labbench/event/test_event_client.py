from typing import Any, Callable, List, Tuple
from uuid import UUID, uuid4

import pytest

from sc_labbench.event._client import EventClient


class DummyButton:
    """Mock button class for testing click events."""

    def __init__(self, id: str = "test-button") -> None:
        self.id = id


class MockBroker:
    """Mock implementation of EventBroker for testing."""

    def __init__(self) -> None:
        self.id: UUID = uuid4().hex
        self.registered_clients: List[EventClient] = []
        self.published_events: List[Tuple[str, tuple, dict]] = []
        self.unsubscribed_events: List[Tuple[str, Callable]] = []
        self.subscriptions: List[Tuple[str, Callable]] = []

    def register_client(self, client: EventClient) -> None:
        """Record client registration"""
        self.registered_clients.append(client)
        client.broker = self

    def update_subscriptions(self) -> None:
        """No-op for testing"""
        pass

    def publish(self, event_str: str, *args: Any, **kwargs: Any) -> None:
        """Record published events with their arguments"""
        self.published_events.append((event_str, args, kwargs))

    def subscribe(self, event: str, callback: Callable) -> None:
        """Record event subscriptions"""
        self.subscriptions.append((event, callback))

    def unsubscribe(self, event: str, callback: Callable) -> None:
        """Record event unsubscriptions"""
        self.unsubscribed_events.append((event, callback))


@pytest.fixture
def broker() -> MockBroker:
    """Fixture to provide a fresh MockBroker instance for each test"""
    return MockBroker()


@pytest.fixture
def client(broker: MockBroker) -> EventClient:
    """Fixture to provide a fresh EventClient instance for each test"""
    client = EventClient(broker)
    # Set up some test events
    client.events = ["test_button_click", "test_control_change"]
    return client


def test_client_initialization(broker: MockBroker, client: EventClient) -> None:
    """Test that client is properly initialized with broker"""
    assert isinstance(client.uuid, str)
    assert len(client.uuid) > 0
    assert client.subscriptions == {}
    assert client in broker.registered_clients
    assert client.broker == broker


def test_button_click_event_publisher(broker: MockBroker, client: EventClient) -> None:
    """Test button click event publisher creation and execution"""
    # Get publisher function
    publisher = client.button_click_event_publisher("test", "button")
    assert callable(publisher)

    # Simulate button click with our dummy button
    button = DummyButton()
    publisher(button)

    # Verify event was published
    assert len(broker.published_events) == 1
    event_name, args, kwargs = broker.published_events[0]
    assert event_name == "test_button_click"
    assert not kwargs  # No kwargs expected for button click


def test_button_click_invalid_event(broker: MockBroker, client: EventClient) -> None:
    """Test that using an invalid event name raises AssertionError"""
    with pytest.raises(AssertionError):
        client.button_click_event_publisher("invalid", "button")


def test_control_value_change_publisher(
    broker: MockBroker, client: EventClient
) -> None:
    """Test control value change event publisher creation and execution"""
    # Get publisher function
    publisher = client.control_value_change_event_publisher("test", "control")
    assert callable(publisher)

    # Simulate value change event
    test_value = 42
    publisher({"type": "change", "new": test_value})

    # Verify event was published with correct value
    assert len(broker.published_events) == 1
    event_name, args, kwargs = broker.published_events[0]
    assert event_name == "test_control_change"
    assert kwargs.get("new_value") == test_value


def test_control_value_change_invalid_event_type(
    broker: MockBroker, client: EventClient
) -> None:
    """Test that using an invalid event type raises ValueError"""
    publisher = client.control_value_change_event_publisher("test", "control")
    with pytest.raises(ValueError, match="Event type must be 'change'"):
        publisher({"type": "invalid", "new": 42})


def test_cleanup_on_deletion(broker: MockBroker, client: EventClient) -> None:
    """Test that subscriptions are properly cleaned up when client is deleted"""

    # Add a test subscription
    def test_callback(x: Any) -> None:
        return x

    client.subscriptions["test_event"] = test_callback

    # Trigger cleanup
    client.__del__()

    # Verify unsubscribe was called
    assert ("test_event", test_callback) in broker.unsubscribed_events


def test_multiple_publishers_same_client(
    broker: MockBroker, client: EventClient
) -> None:
    """Test that multiple publishers can coexist on the same client"""
    button_publisher = client.button_click_event_publisher("test", "button")
    control_publisher = client.control_value_change_event_publisher("test", "control")

    # Use both publishers
    button = DummyButton()
    button_publisher(button)
    control_publisher({"type": "change", "new": 42})

    # Verify both events were published
    assert len(broker.published_events) == 2
    assert broker.published_events[0][0] == "test_button_click"
    assert broker.published_events[1][0] == "test_control_change"
