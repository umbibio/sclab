"""Background steps: work in a worker thread, everything else on the main thread.

A ``FakeLoop`` stands in for the kernel's IOLoop: the job's main-thread parts
run when the test thread pumps its timers (the fallback when no frontend
beats), or on simulated beats.
"""

import threading
import time

import numpy as np
import pandas as pd
import pytest
import scanpy

from sclab import SCLabDashboard
from sclab._workspace import Workspace
from sclab.dataset.plotter import Plotter
from sclab.dataset.processor.step._processor_step_base import (
    _apply_changes,
    _shallow_copy,
)
from sclab.event import StepCancelled

from .test_utils import run_step, simple_loop_adata


class RenderCalls(list):
    fail = False


class FakeLoop:
    def __init__(self):
        self.timers = []

    def call_later(self, delay, fn, *args):
        self.timers.append((fn, args))

    def run_timers(self):
        timers, self.timers = self.timers, []
        for fn, args in timers:
            fn(*args)

    def pump(self, jobs, timeout=30):
        deadline = time.monotonic() + timeout
        while jobs.busy:
            assert time.monotonic() < deadline, "the job did not finish"
            time.sleep(0.01)
            self.run_timers()


@pytest.fixture
def renders(monkeypatch):
    """Threads of the figure renders; ``renders.fail = True`` makes them raise."""
    calls = RenderCalls()
    make_new_figure = Plotter.make_new_figure

    def counting(self, *args, **kwargs):
        calls.append(threading.current_thread())
        if calls.fail:
            raise RuntimeError("render failed")
        return make_new_figure(self, *args, **kwargs)

    monkeypatch.setattr(Plotter, "make_new_figure", counting)
    return calls


@pytest.fixture
def dash(renders):
    d = SCLabDashboard(simple_loop_adata(n_obs=300))
    run_step(d, "QC")
    run_step(d, "Preprocess")
    d.broker.jobs.loop = FakeLoop()
    d.broker.jobs.beat_timeout = 0  # no frontend: resume on the timer
    return d


@pytest.fixture
def leiden(monkeypatch):
    """Fake ``sc.tl.leiden``: runs ``leiden.body(adata)`` (in the worker)."""

    class Fake:
        started = threading.Event()
        release = threading.Event()
        threads = []

        def body(self, adata):
            self.release.wait(10)

        def __call__(self, adata, **kwargs):
            self.threads.append(threading.current_thread())
            self.started.set()
            self.body(adata)
            adata.obs["leiden"] = pd.Categorical(["0"] * adata.n_obs)
            adata.uns["leiden"] = {"params": kwargs}

    fake = Fake()
    monkeypatch.setattr(scanpy.tl, "leiden", fake)
    return fake


def click_run(dash, step: str):
    step = dash.processor.steps[step]
    step.run_button.click()
    return step


def test_background_step_works_in_a_thread_and_redraws_once(dash, renders, monkeypatch):
    import sclab.examples.processor_steps._pca as mod

    work_threads = []
    real_pca = mod.pca

    def pca(adata, **kwargs):
        work_threads.append(threading.current_thread())
        assert "X_pca" not in dash.dataset.adata.obsm  # works on a copy
        return real_pca(adata, **kwargs)

    monkeypatch.setattr(mod, "pca", pca)
    renders.clear()
    step = click_run(dash, "PCA")
    jobs = dash.broker.jobs
    assert jobs.busy and jobs.panel.layout.display == ""
    assert step.run_button.disabled

    jobs.loop.pump(jobs)

    assert work_threads and work_threads[0] is not threading.main_thread()
    assert renders == [threading.main_thread()]  # one redraw, on the main thread
    assert "X_pca" in dash.dataset.adata.obsm
    assert step.run_button.button_style == "success" and not step.run_button.disabled
    assert not step.plot_variance_ratio_button.disabled  # set after the lock
    entry = dash.processor.step_history[-1]
    assert (entry["step"], entry["status"]) == ("PCA", "success")
    assert entry["params"]["n_comps"] == 30
    assert entry["shape_after"] == list(dash.dataset.adata.shape)
    assert not jobs.busy and jobs.panel.layout.display == "none"
    assert not dash.broker.exceptions_log


def test_frontend_beats_resume_the_job(dash, renders, leiden):
    jobs = dash.broker.jobs
    jobs.beat_timeout = 1e9  # the frontend beats: the timer only waits
    leiden.release.set()
    step = click_run(dash, "Cluster")
    assert jobs.beat.playing
    for _ in range(500):
        jobs.loop.run_timers()
        if not jobs.busy:
            break
        jobs.beat.value += 1  # a frontend tick, a comm message in a kernel
        time.sleep(0.01)
    assert not jobs.busy and not jobs.beat.playing
    assert jobs.loop.timers == []  # the timer chain ended with the job
    assert step.run_button.button_style == "success"


def test_worker_publishes_are_replayed_on_the_main_thread(dash, renders, leiden):
    received = []
    dash.broker.subscribe(
        "dset_anndata_neighbors_change",
        lambda: received.append(threading.current_thread()),
    )

    def body(adata):
        dash.broker.publish("dset_anndata_neighbors_change")
        assert not received  # queued

    leiden.body = body
    renders.clear()
    click_run(dash, "Cluster")
    dash.broker.jobs.loop.pump(dash.broker.jobs)

    assert leiden.threads[0] is not threading.main_thread()
    assert received == [threading.main_thread()]
    assert renders == [threading.main_thread()]  # in one batch: one redraw
    assert "leiden" in dash.dataset.adata.obs
    assert not dash.broker.exceptions_log


def assert_cancelled(dash, step, obs_before, obsm_before, uns_before):
    adata = dash.dataset.adata
    assert list(adata.obs.columns) == obs_before
    assert list(adata.obsm) == obsm_before and list(adata.uns) == uns_before
    entry = dash.processor.step_history[-1]
    assert (entry["step"], entry["status"]) == (step.description, "cancelled")
    assert step.run_button.button_style == "" and not step.run_button.disabled
    assert not dash.broker.jobs.busy
    assert not dash.broker.exceptions_log


def snapshot(dash):
    adata = dash.dataset.adata
    return list(adata.obs.columns), list(adata.obsm), list(adata.uns)


def test_cancel_after_the_work_applies_nothing(dash, leiden):
    before = snapshot(dash)
    step = click_run(dash, "Cluster")
    jobs = dash.broker.jobs
    assert leiden.started.wait(10)
    jobs.cancel()
    leiden.release.set()  # the work finishes; its result is discarded
    jobs.loop.pump(jobs)
    assert_cancelled(dash, step, *before)
    assert dash.processor.step_history[-1]["params"]["resolution"] == 1.0


def test_cancel_at_a_print_or_progress_checkpoint(dash, leiden):
    seen = []

    def body(adata):
        try:
            while True:
                print("iteration")
                dash.broker.jobs.progress(0.5)
                time.sleep(0.01)
        except StepCancelled:
            seen.append("cancelled")
            raise

    leiden.body = body
    before = snapshot(dash)
    step = click_run(dash, "Cluster")
    jobs = dash.broker.jobs
    assert leiden.started.wait(10)
    time.sleep(0.05)
    jobs.loop.run_timers()  # a tick: renders the progress
    assert "50%" in jobs.bar.value
    jobs.cancel()
    jobs.loop.pump(jobs)
    assert seen == ["cancelled"]
    assert_cancelled(dash, step, *before)
    printed = "".join(o.get("text", "") for o in dash.broker.std_output.outputs)
    assert "iteration" in printed  # the worker's prints go to the Logs tab


def test_interrupt_raises_in_the_worker(dash, leiden):
    seen = []

    def body(adata):
        try:
            while True:  # pure Python: no checkpoint, but bytecodes
                sum(range(100))
        except StepCancelled:
            seen.append(threading.current_thread())
            raise

    leiden.body = body
    before = snapshot(dash)
    step = click_run(dash, "Cluster")
    jobs = dash.broker.jobs
    assert leiden.started.wait(10)
    jobs.interrupt()
    jobs.loop.pump(jobs)
    assert seen == leiden.threads  # raised there, in the worker
    assert_cancelled(dash, step, *before)


def test_worker_error_reaches_the_logs_and_the_button(dash, leiden):
    def body(adata):
        raise ValueError("boom")

    leiden.body = body
    step = click_run(dash, "Cluster")
    dash.broker.jobs.loop.pump(dash.broker.jobs)
    assert step.run_button.button_style == "danger"
    entry = dash.processor.step_history[-1]
    assert (entry["status"], entry["error"]) == ("failed", "ValueError: boom")
    assert any("ValueError: boom" in e for e in dash.broker.exceptions_log)
    assert "leiden" not in dash.dataset.adata.obs


@pytest.mark.parametrize("background", [False, True])
def test_failed_redraw_turns_the_button_red(dash, renders, background):
    renders.fail = True
    step = dash.processor.steps["PCA"]
    if background:
        click_run(dash, "PCA")
        dash.broker.jobs.loop.pump(dash.broker.jobs)
    else:
        step.run()
    assert renders  # the deferred redraw ran, and failed
    assert any("render failed" in e for e in dash.broker.exceptions_log)
    assert step.run_button.button_style == "danger"
    assert dash.processor.step_history[-1]["status"] == "success"  # the data step


def test_controls_are_locked_while_a_job_runs(dash, leiden):
    controls = [c for c in dash.processor.lockable_controls() if hasattr(c, "disabled")]
    before = {id(c): c.disabled for c in controls}
    assert any(before.values()) and not all(before.values())
    keep, drop = dash.processor.selection_buttons_dict["keep_drop_buttons"].children
    assert keep in controls and drop in controls

    step = click_run(dash, "Cluster")
    jobs = dash.broker.jobs
    assert leiden.started.wait(10)
    assert all(c.disabled for c in controls)
    with pytest.raises(RuntimeError, match="'Cluster' is running"):
        dash.processor.steps["QC"].run()
    with pytest.raises(RuntimeError, match="'Cluster' is running"):
        dash.processor.steps["filter_rows"].run(index=dash.dataset.adata.obs_names)
    clear = dash.processor.selection_buttons_dict["clear_selection"]
    assert not clear.disabled  # selecting does not change the data

    leiden.release.set()
    jobs.loop.pump(jobs)
    after = {id(c): c.disabled for c in controls}
    assert after == before
    assert step.run_button.button_style == "success"
    assert dash.dataset.adata.n_obs == 300


def test_workspace_refuses_to_switch_or_merge_during_a_job(renders, leiden):
    ws = Workspace(
        {"a": simple_loop_adata(n_obs=200), "b": simple_loop_adata(n_obs=100)}
    )
    jobs = ws.dash.broker.jobs
    jobs.loop, jobs.beat_timeout = FakeLoop(), 0
    click_run(ws.dash, "Cluster")
    assert leiden.started.wait(10)

    with pytest.raises(RuntimeError, match="'Cluster' is running in 'a'"):
        ws.activate("b")
    with pytest.raises(RuntimeError, match="is running"):
        ws.merge(["a", "b"], "ab")
    ws._ui(ws.activate, "b")  # from the selector: the error is shown
    assert "is running" in ws.message.value
    assert ws.active == "a" and ws.selector.value == "a"
    with pytest.raises(RuntimeError, match="is running"):
        ws.remove("a")

    leiden.release.set()
    jobs.loop.pump(jobs)
    ws.activate("b")
    assert ws.active == "b"


def test_without_a_kernel_the_button_runs_synchronously(dash, leiden):
    dash.broker.jobs.loop = None  # and no kernel in pytest
    leiden.release.set()
    step = click_run(dash, "Cluster")
    assert not dash.broker.jobs.busy
    assert leiden.threads == [threading.main_thread()]
    assert step.run_button.button_style == "success"
    assert "leiden" in dash.dataset.adata.obs


def test_shallow_copy_changes_apply_only_on_request():
    adata = simple_loop_adata(n_obs=50)
    adata.obsm["old"] = np.zeros((50, 2))
    adata.uns["keep"] = keep = {"a": 1}
    adata.obs["gone"] = 1
    work, entries = _shallow_copy(adata)
    assert work.X is adata.X and work.uns["keep"] is keep

    work.obsm["new"] = np.ones((50, 2))
    del work.obsm["old"]
    work.uns["u"] = 1
    work.obs["c"] = "x"
    del work.obs["gone"]
    assert "new" not in adata.obsm and "old" in adata.obsm and "u" not in adata.uns
    assert "c" not in adata.obs and "gone" in adata.obs

    adata.uns["main"] = 2  # set meanwhile on the main thread: kept
    _apply_changes(adata, work, entries)
    assert "new" in adata.obsm and "old" not in adata.obsm
    assert adata.uns["u"] == 1 and adata.uns["main"] == 2 and adata.uns["keep"] is keep
    assert list(adata.obs["c"]) == ["x"] * 50 and "gone" not in adata.obs
