from uuid import uuid4

from ipywidgets.widgets import Button, Output


# forward declaration
class EventBroker:
    clients: list["EventClient"]
    available_events: set[str]
    preemptions: dict[str, list[str]]
    _subscribers: dict[str, list[callable]]
    _disabled_events: list[str]
    std_output: Output

    def register_client(self, client: "EventClient"):
        ...

    def unregister_client(self, client: "EventClient"):
        ...

    def update_subscriptions(self):
        ...

    def subscribe(self, event: str, callback: callable):
        ...

    def unsubscribe(self, event: str, callback: callable):
        ...

    def publish(self, event: str, *args, **kwargs):
        ...

    def disable_event(self, event: str):
        ...

    def enable_event(self, event: str):
        ...


class EventClient:
    subscriptions: dict[str, callable]
    preemptions: dict[str, list[str]] = {}  # override this in subclasses if needed
    broker: "EventBroker"
    events: list[str]

    def __init__(self, broker: "EventBroker"):
        self.uuid = uuid4().hex
        self.subscriptions = {}
        broker.register_client(self)
        broker.update_subscriptions()

    def button_click_event_publisher(self, control_prefix: str, control_name: str):
        event_str = f"{control_prefix}_{control_name}_click"
        assert (
            event_str in self.events
        ), f"Event {event_str} not in {self.__class__.__name__}.events list."

        def publisher(_: Button):
            self.broker.publish(event_str)

        return publisher

    def control_value_change_event_publisher(
        self, control_prefix: str, control_name: str
    ):
        event_str = f"{control_prefix}_{control_name}_change"
        assert (
            event_str in self.events
        ), f"Event {event_str} not in {self.__class__.__name__}.events list."

        def publisher(event: dict):
            if event["type"] != "change":
                raise ValueError("Event type must be 'change'")
            new_value = event["new"]
            self.broker.publish(event_str, new_value=new_value)

        return publisher

    def __del__(self):
        for event, callback in self.subscriptions.items():
            self.broker.unsubscribe(event, callback)
