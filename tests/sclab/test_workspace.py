"""Workspace: switching datasets keeps edits safe and frees what it can."""

from types import SimpleNamespace

import h5py
import pytest
from anndata import AnnData
from ipywidgets import Widget

import sclab._sclab as sclab_module
import sclab._workspace as workspace_module
from sclab import SCLabDashboard, Workspace
from sclab._workspace import _open_lazy

from .test_utils import live_widget_ids, run_step, simple_loop_adata


@pytest.fixture
def files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    paths = {}
    for name, n_obs in [("a", 300), ("b", 200)]:
        paths[name] = data / f"{name}.h5ad"
        simple_loop_adata(n_obs=n_obs, n_vars=50).write_h5ad(paths[name])
    return paths


def edit(ws: Workspace) -> str:
    """Run QC and pin a view on the active dataset; return the view id."""
    run_step(ws.dash, "QC")
    return ws.dash.plotter.views.pin("mine")["id"]


def test_history_and_views_survive_switching(files, tmp_path):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    assert ws.active == "a"
    view_id = edit(ws)
    assert ws.is_dirty("a")

    ws.activate("b")
    a = ws.datasets["a"]
    assert a.adata is None and a.dash is None
    assert a.cache_path == tmp_path / "ws" / "a.h5ad" and a.cache_path.exists()
    assert ws.dash.ds.adata.n_obs == 200

    ws.activate("a")
    assert [e["step"] for e in ws.dash.processor.step_history] == ["load", "QC"]
    assert [v["id"] for v in ws.dash.plotter.views.views] == [view_id]
    assert not ws.is_dirty("a")  # matches its cache
    assert "cached" not in ws._label(a) and "modified" in ws._label(a)


def test_dirty_switch_without_workspace_dir_raises(files, tmp_path):
    ws = Workspace(files)
    edit(ws)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(RuntimeError, match="workspace_dir"):
        ws.activate("b")
    assert set(tmp_path.rglob("*")) == before
    assert ws.active == "a" and ws.dash is not None
    assert ws.datasets["a"].adata is not None

    ws._ui(ws.activate, "b")  # the UI shows the error instead
    assert "workspace_dir" in ws.message.value
    assert ws.active == "a" and ws.selector.value == "a"


def test_in_memory_dataset_is_kept_without_workspace_dir(files):
    adata = simple_loop_adata(n_obs=100, n_vars=50)
    ws = Workspace({"mem": adata, "b": files["b"]})
    run_step(ws.dash, "QC")
    ws.activate("b")
    mem = ws.datasets["mem"]
    assert mem.adata is adata and mem.cache_path is None
    ws.activate("mem")
    assert [e["step"] for e in ws.dash.processor.step_history] == ["QC"]


def test_clean_file_dataset_is_freed_and_reread(files, monkeypatch):
    reads = []
    read_adata = workspace_module.read_adata
    monkeypatch.setattr(
        workspace_module, "read_adata", lambda p: reads.append(p) or read_adata(p)
    )
    ws = Workspace(files)
    ws.activate("b")
    assert ws.datasets["a"].adata is None
    assert "on disk" in ws._label(ws.datasets["a"])
    ws.activate("a")
    assert reads == [files["a"], files["b"], files["a"]]
    assert ws.dash.ds.adata.shape == (300, 50)


def test_failed_write_refuses_switch(files, tmp_path, monkeypatch):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    edit(ws)
    dash = ws.dash

    def fail(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(AnnData, "write_h5ad", fail)
    with pytest.raises(OSError, match="disk full"):
        ws.activate("b")
    assert ws.active == "a" and ws.dash is dash and dash.comm is not None
    assert ws.datasets["a"].adata is not None and ws.datasets["a"].cache_path is None
    assert list((tmp_path / "ws").iterdir()) == []  # no partial file left behind


def test_switching_closes_the_dashboard_widgets(files):
    ws = Workspace(files)
    old = ws.dash
    figure, button = old.plotter.g, old.processor.steps["QC"].run_button
    ws.activate("b")
    assert old.comm is None and figure.comm is None and button.comm is None

    counts = []
    for name in ["a", "b", "a", "b"]:
        ws.activate(name)
        counts.append(len(live_widget_ids()))
    assert counts[2:] == counts[:2]

    widgets = live_widget_ids()
    ws.close()
    assert ws.comm is None and ws.datasets["b"].dash.comm is None
    assert len(live_widget_ids()) < len(widgets)


@pytest.mark.parametrize("make", ["dashboard", "workspace"])
def test_garbage_collection_closes_only_the_widget_itself(make, files, monkeypatch):
    if make == "dashboard":
        root = SCLabDashboard(simple_loop_adata(n_obs=100, n_vars=50))
    else:
        root = Workspace(files)
    figure = (root if make == "dashboard" else root.dash).plotter.g  # in use elsewhere

    def shutdown():  # the registry lookup at interpreter shutdown
        raise ImportError("sys.meta_path is None, Python is likely shutting down")

    monkeypatch.setattr(sclab_module, "_widget_registry", shutdown)
    # GC calls __del__ (also at shutdown): no error, and the widgets it holds,
    # which may be displayed elsewhere, stay open
    root.__del__()
    assert root.comm is None and figure.comm is not None
    Widget.close(figure)


def test_backed_listing_reads_shape_without_x(files, monkeypatch):
    monkeypatch.setattr(workspace_module, "_lazy_reader", lambda: None)
    opened = []
    read_h5ad = workspace_module.read_h5ad
    monkeypatch.setattr(
        workspace_module,
        "read_h5ad",
        lambda p, **kw: opened.append(read_h5ad(p, **kw)) or opened[-1],
    )
    assert _open_lazy(files["a"]) == (300, 50)
    (adata,) = opened
    assert adata.isbacked and not adata.file.is_open

    ws = Workspace(files)
    assert ws.datasets["b"].adata is None and ws.datasets["b"].shape == (200, 50)
    assert "200×50 · on disk · clean" in ws._label(ws.datasets["b"])


def test_lazy_listing_uses_read_lazy(files, monkeypatch):
    seen = []

    def fake_read_lazy(f):
        seen.append(isinstance(f, h5py.Group))
        return SimpleNamespace(shape=(300, 50))

    monkeypatch.setattr(workspace_module, "_lazy_reader", lambda: fake_read_lazy)
    monkeypatch.setattr(workspace_module, "read_h5ad", None)  # never the fallback
    assert _open_lazy(files["a"]) == (300, 50)
    assert seen == [True]


def test_read_lazy_shape(files):
    pytest.importorskip("xarray")
    pytest.importorskip("dask")
    assert workspace_module._lazy_reader() is not None
    assert _open_lazy(files["b"]) == (200, 50)


def test_tenx_listing_is_deferred(tmp_path):
    assert _open_lazy(tmp_path / "x.h5") is None
    assert _open_lazy(tmp_path) is None


def test_add_rename_remove(files, tmp_path):
    ws = Workspace({"a": files["a"]})
    with pytest.raises(ValueError, match="already exists"):
        ws.add("a", files["b"])
    with pytest.raises(ValueError, match="Invalid"):
        ws.add("../b", files["b"])
    ws.add("b", files["b"])
    ws.rename("a", "first")
    assert list(ws.datasets) == ["first", "b"] and ws.active == "first"
    ws.remove("first")
    assert list(ws.datasets) == ["b"] and ws.active is None and ws.dash is None
    assert ws.selector.value is None


def test_load_button_adds_a_defined_adata(files):
    extra = simple_loop_adata(n_obs=80, n_vars=50)  # noqa: F841 - listed by the loader
    ws = Workspace({"a": files["a"]})
    ws.load_button.click()
    (loader,) = ws.loader_box.children
    loader.defined_adatas.value = "extra"
    loader.continue_button.click()
    assert ws.active == "extra" and ws.dash.ds.adata.shape == (80, 50)
    assert ws.loader_box.children == () and loader.comm is None


def labels(ws: Workspace) -> list[str]:
    return [label for label, _ in ws.selector.options]


def test_label_follows_history_and_views_of_the_active_dataset(files, tmp_path):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    assert "clean" in labels(ws)[0]
    broker = ws.dash.broker
    ws.dash.plotter.views.pin("mine")  # not a top-bar action
    assert "modified" in labels(ws)[0] and ws.active == "a"
    assert ws.selector.value == "a"
    run_step(ws.dash, "QC")
    assert "modified" in labels(ws)[0] and "clean" in labels(ws)[1]
    watcher = ws._watcher
    ws.activate("b")
    assert watcher not in broker._subscribers.get("dset_state_change", [])
    run_step(ws.dash, "QC")
    assert "modified" in labels(ws)[1]


def test_label_is_refreshed_after_a_background_step(files, tmp_path):
    from .test_background_jobs import FakeLoop

    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    jobs = ws.dash.broker.jobs
    jobs.loop, jobs.beat_timeout = FakeLoop(), 0
    ws.dash.processor.steps["QC"].run_button.click()
    jobs.loop.pump(jobs)
    assert ws.dash.processor.step_history[-1]["step"] == "QC"
    assert "modified" in labels(ws)[0]


def test_rename_updates_the_dashboard_and_the_cache_file(files, tmp_path):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    edit(ws)
    ws.activate("b")  # a is written to its cache
    old = ws.datasets["a"].cache_path
    ws.rename("a", "first")
    assert not old.exists() and ws.datasets["first"].cache_path == old.with_name(
        "first.h5ad"
    )
    assert ws.datasets["first"].cache_path.exists()
    ws.activate("first")
    assert [e["step"] for e in ws.dash.processor.step_history] == ["load", "QC"]
    ws.rename("first", "second")
    assert ws.dash.name == ws.dash.dataset.name == "second"
    assert ws.datasets["second"].cache_path.name == "second.h5ad"
    assert "second" in labels(ws)[0]


def test_rename_refuses_an_existing_cache_file(files, tmp_path):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    edit(ws)
    ws.activate("b")
    (tmp_path / "ws" / "taken.h5ad").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        ws.rename("a", "taken")
    assert list(ws.datasets) == ["a", "b"]
    assert ws.datasets["a"].cache_path.name == "a.h5ad"
    assert ws.datasets["a"].cache_path.exists()
