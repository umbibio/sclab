import functools
import traceback
from contextlib import contextmanager
from uuid import uuid4

from IPython.display import HTML, display
from ipywidgets.widgets import Layout, Output, Tab

from ._client import EventClient
from ._utils import LogOutput


class EventBroker:
    """Simple event broker for publishing and subscribing to events.

    Example:
    ```
    broker = EventBroker()

    def callback(*args, **kwargs):
        print(args, kwargs)

    broker.subscribe("event", callback)
    broker.publish("event", 1, 2, 3, a=4, b=5)
    ```
    """

    clients: list[EventClient]
    available_events: set[str]
    preemptions: dict[str, list[str]]
    _subscribers: dict[str, list[callable]]
    _disabled_events: list[str]

    def __init__(self):
        self.clients = []
        self.available_events = set()
        self.preemptions = {}
        self._subscribers = {}
        self._disabled_events = []
        self.depth = 0
        # False makes coalesced functions run immediately, e.g. to compare final states
        self.coalescing = True
        self.max_flush_calls = 1000
        self._batch_depth = 0
        self._pending: dict[callable, tuple[callable, tuple, dict]] = {}
        self.std_output = Output(layout=Layout(width="auto", height="500px"))
        self.execution_log = []
        self.execution_output = LogOutput()
        self.exceptions_log = []
        self.exceptions_output = LogOutput()
        self.id = uuid4()
        self.event_log: Output = Output(
            layout=Layout(height="200px"),
            # style={"overflow-y": "scroll"},  TODO: it seems it the way to set this has changed
        )
        self.logs_tab = Tab(
            [
                self.std_output,
                self.execution_output,
                self.exceptions_output,
            ],
            titles=[
                "Standard Output",
                "Events",
                "Exceptions",
            ],
        )

    def register_client(self, client: EventClient):
        if not isinstance(client, EventClient):
            raise TypeError("client must be an instance of EventClient")

        if client in self.clients:
            return

        self.clients.append(client)
        self.available_events.update(client.events)
        self.preemptions.update(client.preemptions)
        client.broker = self

    def unregister_client(self, client: EventClient):
        # TODO: ensure full cleanup of subscriptions

        if client in self.clients:
            self.clients.remove(client)

        self.available_events.difference_update(client.events)

        self._disabled_events = [
            e for e in self._disabled_events if e not in client.events
        ]

        for event in client.events:
            self._subscribers.pop(event, None)
            self.preemptions.pop(event, None)

    def update_subscriptions(self):
        for client in self.clients:
            for event in self.available_events:
                if callback := getattr(client, f"{event}_callback", None):
                    self.subscribe(event, callback)
                    client.subscriptions[event] = callback

    def subscribe(self, event: str, callback: callable):
        if event not in self._subscribers:
            self._subscribers[event] = []

        if callback not in self._subscribers[event]:
            # Prevent duplicate subscriptions
            self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: callable):
        if event in self._subscribers and callback in self._subscribers[event]:
            self._subscribers[event].remove(callback)

    def publish(self, event: str, *args, **kwargs):
        if event not in self.available_events:
            raise ValueError(f"Event '{event}' is not available.")

        with self.batch():
            self._publish(event, *args, **kwargs)

    def _publish(self, event: str, *args, **kwargs):
        try:
            self.depth += 1
            tab = " " * self.depth * 4
            txt_args = []
            for arg in args:
                if isinstance(arg, str | float | int | bool | tuple | Exception | None):
                    txt_args.append(arg)
                else:
                    txt_args.append(type(arg))
            txt_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str | float | int | bool | tuple | Exception | None):
                    txt_kwargs[k] = v
                else:
                    txt_kwargs[k] = type(v)

            if event in self._disabled_events:
                msg = f"{tab}Disabled Event: {event}."
                self.execution_log.append(msg)
                with self.execution_output.output:
                    display(HTML(f"<pre>{msg}</pre>"))
                return

            msg = f"{tab}{event}. Args: {tuple(txt_args)}. Kwargs: {txt_kwargs}"
            self.execution_log.append(msg)
            with self.execution_output.output:
                display(HTML(f"<pre>{msg}</pre>"))

            preempt = self.preemptions.get(event, [])
            with self.disable(preempt):
                if event in self._subscribers:
                    for callback in self._subscribers[event]:
                        if hasattr(callback, "__self__"):
                            parent_class = callback.__self__.__class__.__name__
                        else:
                            parent_class = "<local context>"
                        msg = f"{tab}{event} --> {parent_class}.{callback.__name__}()"
                        self.execution_log.append(msg)
                        with self.execution_output.output:
                            display(HTML(f"<pre>{msg}</pre>"))

                        callback(*args, **kwargs)
        except Exception as e:
            self._log_exception(e, tab)

        finally:
            self.depth -= 1

    def _log_exception(self, e: Exception, tab: str = ""):
        msg = f"{tab}    Exception: {e}"
        self.execution_log.append(msg)
        with self.execution_output.output:
            display(HTML(f"<pre>{msg}</pre>"))

        msg = f"{msg}{tab}    Exception: {e}"
        msg += f"\n\n{traceback.format_exc()}"
        self.exceptions_log.append(msg)

        with self.exceptions_output.output:
            display(HTML(f"<pre>{traceback.format_exc()}</pre>"))

    @contextmanager
    def batch(self):
        """Defer calls to coalesced functions until the outermost batch exits.

        Re-entrant. Every :meth:`publish` runs inside a batch. On exit of the
        outermost batch, each pending coalesced function runs once with the
        arguments of its latest call, in the order it was first called. Calls
        made while flushing (directly or through events) are queued and run in
        the same flush.
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            try:
                if self._batch_depth == 1:
                    self.flush()
            finally:
                self._batch_depth -= 1

    def coalesce(self, fn: callable) -> callable:
        """Wrap ``fn`` so that, inside a :meth:`batch`, it only records its call.

        Outside a batch (or with ``coalescing = False``) the wrapper calls ``fn``
        immediately. Inside one, it returns None and ``fn`` runs when the batch
        ends (see :meth:`batch`), or earlier through :meth:`flush`. Exceptions
        raised by deferred calls go to ``exceptions_log``.
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.coalescing or not self._batch_depth:
                return fn(*args, **kwargs)
            self._pending[wrapper] = (fn, args, kwargs)  # keeps first-seen position

        if hasattr(fn, "__self__"):
            wrapper.__self__ = fn.__self__  # for the event log
        return wrapper

    def flush(self, coalesced: callable = None):
        """Run pending coalesced calls now.

        With ``coalesced`` (a function returned by :meth:`coalesce`), run only its
        pending call, if any. Use it before code that depends on that call's
        result, e.g. before modifying a figure a pending call would replace.
        """
        if coalesced is not None:
            if coalesced in self._pending:
                self._run_pending(coalesced)
            return

        for _ in range(self.max_flush_calls):
            if not self._pending:
                return
            self._run_pending(next(iter(self._pending)))
        if not self._pending:
            return

        pending = [getattr(f, "__name__", f) for f in self._pending]
        self._pending.clear()
        try:
            raise RuntimeError(f"Coalesced calls did not settle; dropped: {pending}")
        except RuntimeError as e:
            self._log_exception(e)

    def _run_pending(self, key: callable):
        fn, args, kwargs = self._pending.pop(key)
        try:
            fn(*args, **kwargs)
        except Exception as e:
            self._log_exception(e)

    def disable_event(self, event: str):
        self._disabled_events.append(event)

    def enable_event(self, event: str):
        if event in self._disabled_events:
            self._disabled_events.remove(event)

    @contextmanager
    def disable(self, events: str | list[str]):
        if isinstance(events, str):
            events = [events]

        try:
            for event in events:
                self.disable_event(event)
            yield
        finally:
            for event in events:
                self.enable_event(event)
