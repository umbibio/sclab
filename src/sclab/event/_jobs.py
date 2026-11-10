"""Run long processor steps off the main thread, keeping the dashboard responsive.

A job is a generator (see ``ProcessorStepBase._run``). Its own code runs on
the "main" thread, the one running widget callbacks (the kernel's main thread,
or a subshell thread with ipykernel 7 and a frontend using kernel subshells
for widgets), each part up to a ``yield`` in one
:meth:`EventBroker.batch`. Each value it yields is a zero-argument callable,
the heavy work, which runs in a worker thread; its result (or exception) is
sent back into the generator. Without a kernel (plain Python, tests), and
always for :meth:`JobRunner.run`, the work runs inline.

In a kernel, :meth:`JobRunner.start` returns once the work is started, so the
kernel serves other widget events (tab switches, plot pan/zoom) meanwhile.
The job resumes on the main thread when a hidden ``Play`` widget ticks (the
"beat", 2 per second; see :meth:`JobRunner._poll`). A tick is a frontend
request, like a click, so the step's code runs as in a synchronous click; in
particular ``with output:``
captures into Output widgets (step messages, Logs, the obs/var and results
tables) work, which they do not in an IOLoop callback (there is no request
for the frontend to route their messages to). Without beats (the dashboard
not displayed), a timer on the kernel's IOLoop resumes the job instead.

The work must not touch widgets: comm messages from other threads are unsafe
in ipykernel. Its ``broker.publish`` calls are queued and replayed on the main
thread, in the batch of the job's next main-thread part, and its prints go to
the Logs tab's Standard Output (its last line shows in the job panel).

Cancel (:meth:`JobRunner.cancel`) sets a flag checked at checkpoints:
:meth:`JobRunner.progress`, each write of the work to stdout/stderr (e.g. by
tqdm), and when the work returns, whose result is then discarded: the step
raises :class:`StepCancelled` instead of applying it. Interrupt
(:meth:`JobRunner.interrupt`) also raises :class:`StepCancelled` in the worker
thread with ``PyThreadState_SetAsyncExc``. That fires only between Python
bytecodes: inside a long C or numba call it waits until the call returns.
"""

import ctypes
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from html import escape

from ipywidgets import HTML, Button, HBox, Layout, Play, VBox
from traitlets import CInt

from .. import _compat


class _Beat(Play):
    """The job's beat. The kernel does not echo its ticks back to the frontend.

    Each tick already costs a comm message and two kernel status messages; an
    echo is a third, sent to every connection, also to a stalled one that the
    server later drops with every unsent message failing (WebSocketClosedError).
    """

    value = CInt(0).tag(sync=True, echo_update=False)


@contextmanager
def capture(output):
    """``with output:`` (an ``Output`` widget) that lets exceptions propagate.

    In a kernel ``Output.__exit__`` shows the traceback in the widget and
    swallows the exception: a failing step would carry on as if it succeeded.
    """
    error = None
    with output:
        try:
            yield
        except BaseException as e:  # noqa: BLE001 - re-raised below
            error = e
    if error is not None:
        raise error


class StepCancelled(KeyboardInterrupt):
    """Raised in a job the user cancelled.

    A ``KeyboardInterrupt`` so that ``except Exception`` in library code does
    not swallow it.
    """


def _kernel_loop():
    """The running kernel's IOLoop, or None (plain Python, a terminal IPython).

    None under WebAssembly (Pyodide), where threads cannot start: jobs run inline.
    """
    if _compat.IS_WASM:
        return None
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return getattr(getattr(get_ipython(), "kernel", None), "io_loop", None)


def _async_raise(thread_id: int, exc: type | None) -> int:
    """Raise ``exc`` in thread ``thread_id`` at its next bytecode; None clears it."""
    return ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), None if exc is None else ctypes.py_object(exc)
    )


class _WorkerStream:
    """``sys.stdout``/``sys.stderr`` during a job: the worker's writes go to it."""

    def __init__(self, stream, runner: "JobRunner"):
        self.stream = stream
        self.runner = runner

    def write(self, text):
        if threading.current_thread() is not self.runner._thread:
            return self.stream.write(text)
        self.runner._output.append(text)
        self.runner.progress()  # a cancel checkpoint, e.g. in tqdm
        return len(text)

    def flush(self):
        if threading.current_thread() is not self.runner._thread:
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


_BUSY_BAR = (
    "<style>@keyframes sclab-job{to{background-position:28px 0}}</style>"
    "<div style='height:8px;border-radius:4px;background:repeating-linear-gradient"
    "(-45deg,#1976d2 0 7px,#90caf9 7px 14px);background-size:28px 8px;"
    "animation:sclab-job .8s linear infinite'></div>"
)


def _bar_html(fraction: float | None) -> str:
    if fraction is None:  # no progress reported: an indeterminate bar
        return _BUSY_BAR
    percent = 100 * min(max(fraction, 0.0), 1.0)
    return (
        "<div style='height:8px;border-radius:4px;background:#e0e0e0'>"
        f"<div style='height:8px;border-radius:4px;width:{percent:.0f}%;"
        "background:#1976d2'></div></div>"
    )


class JobRunner:
    """Runs one job at a time for a dashboard; see the module doc.

    ``panel`` (hidden while idle) shows the running job: its name, elapsed
    time and last printed line, a progress bar, and Cancel and Interrupt
    buttons. While a job runs, the widgets passed as ``lock`` to
    :meth:`start` are disabled, and :meth:`run` / :meth:`start` refuse to run
    another job.
    """

    tick_seconds = 0.5
    beat_timeout = 3.0  # no beat for this long: the IOLoop timer resumes the job
    threaded = True  # False: start() is run(), see the Settings tab "Background steps"

    def __init__(self, broker):
        self.broker = broker
        # IOLoop-like (call_later); None: the kernel's
        self.loop = None
        self.name = None
        self._job = None
        self._loop = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # beats and the timer may come on different threads (kernel subshells)
        self._poll_lock = threading.Lock()
        self._in_work = False
        self._interrupted = False
        self._cancel = threading.Event()
        self._lock_targets = []
        self._locked: dict[int, tuple] = {}  # id -> (widget, disabled before)
        self.deferred: list[tuple[str, tuple, dict]] = []  # the work's publishes
        self._output: list[str] = []
        self._last_line = ""
        self._fraction = None
        self._t0 = 0.0
        self._outcome = None  # (result, exception) of the finished work
        self._last_beat = 0.0

        self.status = HTML()
        self.bar = HTML()
        self.cancel_button = Button(
            description="Cancel",
            tooltip="Stop at the next checkpoint; the data is left unchanged",
            layout=Layout(width="auto"),
        )
        self.interrupt_button = Button(
            description="Interrupt",
            button_style="danger",
            tooltip="Also raise in the running computation (not inside C code)",
            layout=Layout(width="auto"),
        )
        self.cancel_button.on_click(lambda _: self.cancel())
        self.interrupt_button.on_click(lambda _: self.interrupt())
        self.beat = _Beat(
            interval=500,
            max=10**9,
            repeat=True,
            show_repeat=False,
            layout=Layout(display="none"),
        )
        self.beat.observe(self._on_beat, "value")
        buttons = HBox([self.cancel_button, self.interrupt_button, self.beat])
        self.panel = VBox(
            [self.status, self.bar, buttons],
            layout=Layout(display="none", width="98%", margin="0 0 8px"),
        )

    @property
    def busy(self) -> bool:
        return self._job is not None

    def is_worker(self) -> bool:
        """Whether the caller runs in the job's worker thread."""
        return self._thread is not None and threading.current_thread() is self._thread

    def output(self):
        """Context for the work's printed output: the Logs tab's Standard Output.

        On the main thread (:meth:`run`) the Output widget itself; in the
        worker nothing, as the runner already collects what it prints.
        """
        return nullcontext() if self.is_worker() else capture(self.broker.std_output)

    def run(self, job):
        """Drive ``job`` to its end on this thread, in one batch."""
        self._check_idle(job)
        with self.broker.batch():
            value, error = None, None
            while True:
                try:
                    work = job.send(value) if error is None else job.throw(error)
                except StopIteration:
                    return
                try:
                    value, error = work(), None
                except BaseException as e:  # noqa: BLE001 - sent into the job
                    value, error = None, e

    def start(self, job, name: str, lock=()):
        """Drive ``job`` with its work in a worker thread (see the module doc).

        Without a kernel (and no ``loop`` set), or with ``threaded`` off, this
        is :meth:`run`. The widgets in ``lock`` are disabled while the work runs.
        """
        loop = self.loop or _kernel_loop()
        if loop is None or not self.threaded:
            return self.run(job)
        self._check_idle(job)
        self._job, self._loop, self.name = job, loop, name
        self._lock_targets = list(lock)
        self._cancel.clear()
        self.deferred.clear()
        self._output.clear()
        self._last_line, self._fraction = "", None
        self._t0 = self._last_beat = time.monotonic()
        self._advance(None, None)

    def cancel(self):
        """Stop the job at its next checkpoint, see the module doc."""
        if self.busy:
            self._cancel.set()
            self._render()

    def interrupt(self):
        """Cancel, and raise :class:`StepCancelled` in the running work.

        It fires between Python bytecodes only, so not inside a long C call.
        """
        self.cancel()
        with self._lock:
            if self._in_work and not self._interrupted:
                self._interrupted = True
                _async_raise(self._thread.ident, StepCancelled)

    def progress(self, fraction: float | None = None):
        """Report the work's progress (0 to 1); a cancel checkpoint.

        Call it from the work; in the worker it raises :class:`StepCancelled`
        once the job is cancelled.
        """
        if fraction is not None:
            self._fraction = fraction
        if self._cancel.is_set() and self.is_worker():
            raise StepCancelled

    def _check_idle(self, job):
        if self.busy:
            job.close()
            raise RuntimeError(
                f"{self.name!r} is running; wait for it to finish or cancel it."
            )

    def _advance(self, value, error):
        """Run the job's next main-thread part, then start its next work, if any."""
        first = self._thread is None  # set from the first work on
        self._thread = None
        self._unlock()  # first: the step's own code may change the controls
        self._drain_output()
        if error is None and self._cancel.is_set():
            error = StepCancelled()  # discard the result: nothing is applied
        try:
            with self.broker.batch():
                try:
                    job = self._job
                    work = job.send(value) if error is None else job.throw(error)
                finally:
                    deferred, self.deferred = self.deferred, []
                    for event, args, kwargs in deferred:
                        self.broker.publish(event, *args, **kwargs)
        except StopIteration:
            self._finish()
            return
        except BaseException as e:  # noqa: BLE001 - the job's own code failed
            self._finish()
            self.broker._log_exception(e)
            return

        self._lock_widgets()
        self._interrupted = False
        self._thread = threading.Thread(
            target=self._work, args=(work,), name=f"sclab job: {self.name}", daemon=True
        )
        if first:
            self._install_streams()
            self.panel.layout.display = ""
            self._render()
            self.beat.playing = True
            self._loop.call_later(self.tick_seconds, self._tick, self._job)
        self._outcome = None
        with self._lock:
            self._in_work = True
        self._thread.start()

    def _work(self, work):
        """Worker thread: run ``work``, then hand its outcome to the main thread."""
        try:
            try:
                outcome = (work(), None)
            finally:
                with self._lock:
                    self._in_work = False
                    _async_raise(threading.get_ident(), None)  # drop a late interrupt
        except BaseException as e:  # noqa: BLE001 - sent into the job
            outcome = (None, e)
        self._outcome = outcome  # picked up on the main thread by _poll

    def _finish(self):
        self.beat.playing = False
        self._unlock()
        self._uninstall_streams()
        self._drain_output()
        self._job = self._thread = None
        self.panel.layout.display = "none"

    def _on_beat(self, _):
        self._last_beat = time.monotonic()
        self._poll()

    def _tick(self, job):
        if self._job is not job:  # finished
            return
        if time.monotonic() - self._last_beat > self.beat_timeout:
            self._poll()
        if self._job is job:
            self._loop.call_later(self.tick_seconds, self._tick, job)

    def _poll(self):
        """Main thread: resume the job if its work finished, else show its state."""
        if not self.busy or not self._poll_lock.acquire(blocking=False):
            return
        try:
            if self._outcome is not None:
                outcome, self._outcome = self._outcome, None
                self._advance(*outcome)
                return
            self._drain_output()
            self._render()
        finally:
            self._poll_lock.release()

    def _render(self):
        state = "cancelling" if self._cancel.is_set() else "running"
        elapsed = time.monotonic() - self._t0
        html = f"<b>{escape(self.name or '')}</b> {state} · {elapsed:.0f} s"
        if self._last_line:
            line = escape(self._last_line[-100:])
            html += f"<br><code style='font-size:11px'>{line}</code>"
        self.status.value = html
        self.bar.value = _bar_html(self._fraction)  # unchanged: nothing is sent

    def _drain_output(self):
        n = len(self._output)
        if not n:
            return
        text = "".join(self._output[:n])
        del self._output[:n]
        lines = text.replace("\r", "\n").strip().splitlines()
        if lines:
            self._last_line = lines[-1]
        self.broker.std_output.append_stdout(text)

    def _lock_widgets(self):
        for widget in self._lock_targets:
            if hasattr(widget, "disabled") and id(widget) not in self._locked:
                self._locked[id(widget)] = (widget, widget.disabled)
                widget.disabled = True

    def set_disabled(self, widget, disabled: bool):
        """Set ``widget.disabled``; while the job locks it, the state it gets back."""
        if id(widget) in self._locked:
            self._locked[id(widget)] = (widget, disabled)
        else:
            widget.disabled = disabled

    def _unlock(self):
        for widget, disabled in self._locked.values():
            widget.disabled = disabled
        self._locked.clear()

    def _install_streams(self):
        sys.stdout = _WorkerStream(sys.stdout, self)
        sys.stderr = _WorkerStream(sys.stderr, self)

    def _uninstall_streams(self):
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            if isinstance(stream, _WorkerStream) and stream.runner is self:
                setattr(sys, name, stream.stream)
