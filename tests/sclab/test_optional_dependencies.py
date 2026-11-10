"""sclab without its optional packages (as under Pyodide), and the WASM guard."""

import subprocess
import sys
import threading
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import sclab
from sclab import SCLabDashboard, _compat
from sclab.dataset.processor._processor import _UnavailableStep
from sclab.event._jobs import _kernel_loop
from sclab.tools.differential_expression import pylimma_is_available

from .test_utils import run_step, simple_loop_adata

OPTIONAL = [
    "scanpy",
    "umap",
    "pynndescent",
    "numba",
    "rpy2",
    "anndata2ri",
    "scrublet",
    "doubletdetection",
    "bbknn",
    "edgepython",
    "scanorama",
]
SCANPY_STEPS = ["QC", "Preprocess", "PCA", "Neighbors", "UMAP", "Cluster"]


def test_import_sclab_without_optional_packages():
    code = f"""
import sys
for m in {OPTIONAL!r}:
    sys.modules[m] = None  # import m raises ImportError
import sclab
loaded = [m for m in {OPTIONAL!r} if sys.modules.get(m) is not None]
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def without_optional(monkeypatch):
    for m in OPTIONAL:
        monkeypatch.setitem(sys.modules, m, None)


def make_adata():
    adata = simple_loop_adata(n_obs=240, n_vars=120)
    adata.layers["counts"] = adata.X.copy()
    adata.obs["grp"] = pd.cut(adata.obs["time"], bins=3, labels=["A", "B", "C"])
    return adata


def test_dashboard_builds_without_optional_packages(without_optional):
    d = SCLabDashboard(make_adata(), counts_layer="counts")
    steps = d.processor.steps
    for name in SCANPY_STEPS:  # shown, but cannot run
        assert steps[name].run_button.disabled, name
        assert steps[name]._reason == "Needs scanpy: pip install 'sclab[scanpy]'"
        assert "sclab[scanpy]" in steps[name].status.value
    with pytest.raises(ValueError, match="Needs scanpy"):
        steps["QC"].run()

    # its constructor needs a backend: a placeholder saying what to install
    dd = steps["Doublet Detection"]
    assert isinstance(dd, _UnavailableStep)
    assert "sclab[doublets]" in dd.controls.children[0].value
    processing = d.processor.step_groups["Processing"]
    assert "Doublet Detection" in processing.titles
    with pytest.raises(ImportError, match="sclab\\[doublets\\]"):
        dd.run()

    de = steps["Differential Expression"]
    c = de.variable_controls
    c["groupby"].value = "grp"
    assert de._reason == "wilcoxon needs scanpy: pip install 'sclab[scanpy]'"
    c["method"].value = "pseudobulk_limma"
    if pylimma_is_available():
        assert de._reason is None


@pytest.mark.skipif(not pylimma_is_available(), reason="pylimma not installed")
def test_pseudobulk_de_runs_without_scanpy(without_optional):
    d = SCLabDashboard(make_adata(), counts_layer="counts")
    run_step(
        d,
        "Differential Expression",
        method="pseudobulk_limma",
        groupby="grp",
        groups=(),
        reference="A",
        cell_identity=None,
        sample_key=None,
        replicas_per_group=3,
        override_layer=False,
        layer=None,
        name="",
    )
    res = d.dataset.adata.uns["differential_expression"]
    assert res["params"]["method"] == "pseudobulk_limma"
    assert set(res["table"]["contrast"]) <= {"B_vs_A", "C_vs_A"}
    assert sys.modules["scanpy"] is None  # not imported meanwhile


def test_wasm_runs_jobs_on_the_main_thread(monkeypatch):
    import IPython
    import scanpy

    class FakeLoop:
        def call_later(self, *args):
            raise AssertionError("no timers under wasm")

    kernel = type("Kernel", (), {"io_loop": FakeLoop()})()
    shell = type("Shell", (), {"kernel": kernel})()
    monkeypatch.setattr(IPython, "get_ipython", lambda: shell)
    assert _kernel_loop() is kernel.io_loop  # a kernel: background jobs

    threads = []

    def leiden(adata, **kwargs):
        threads.append(threading.current_thread())
        adata.obs["leiden"] = pd.Categorical(["0"] * adata.n_obs)

    monkeypatch.setattr(scanpy.tl, "leiden", leiden)
    monkeypatch.setattr(_compat, "IS_WASM", True)
    assert _kernel_loop() is None

    d = SCLabDashboard(make_adata())
    adata = d.dataset.adata  # a stand-in graph: Cluster needs one to run
    adata.uns["neighbors"] = {}
    adata.obsp["connectivities"] = scipy.sparse.identity(adata.n_obs, format="csr")
    d.broker.publish("dset_anndata_neighbors_change")
    step = d.processor.steps["Cluster"]
    step.run_button.click()
    assert not d.broker.jobs.busy
    assert threads == [threading.main_thread()]
    assert step.run_button.button_style == "success"

    assert d.background_checkbox.disabled
    assert d.background_checkbox.value is False


def flavors(d):
    return d.processor.steps["Integration"].variable_controls["flavor"].options


def test_scanorama_offered_only_when_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "scanorama", None)
    assert flavors(SCLabDashboard(make_adata())) == ("cca", "harmony")

    monkeypatch.setattr(_compat, "missing", lambda feature: None)
    assert flavors(SCLabDashboard(make_adata()))[-1] == "scanorama"
    monkeypatch.setattr(_compat, "IS_WASM", True)
    assert "scanorama" not in flavors(SCLabDashboard(make_adata()))


def test_optional_dependencies_table(capsys, monkeypatch):
    df = sclab.optional_dependencies()
    assert list(df["feature"]) == list(_compat.FEATURES)
    for row in df.itertuples():
        modules = _compat.FEATURES[row.feature].modules
        assert row.available == all(find_spec(m) is not None for m in modules)
    out = capsys.readouterr().out
    assert "pip install 'sclab[de-edger]'" in out and "scanorama" in out

    monkeypatch.setattr(_compat, "IS_WASM", True)
    df = sclab.optional_dependencies(print_table=False)
    install = dict(zip(df["feature"], df["install"]))
    assert install["jscatter"] == "await micropip.install(['jupyter-scatter'])"
    assert install["R"].startswith("not available under Pyodide")
    assert "micropip.install('edgepython', deps=False)" in install["edgepython"]
    assert capsys.readouterr().out == ""


def test_messages_come_from_the_registry(monkeypatch):
    from sclab.preprocess import neighbors

    monkeypatch.setitem(sys.modules, "bbknn", None)
    reason = _compat.missing("bbknn")
    assert reason == "Needs bbknn: pip install 'sclab[bbknn]'"
    adata = make_adata()
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 10))
    with pytest.raises(ImportError) as error:
        neighbors(adata, batch_key="grp")
    assert str(error.value) == reason

    d = SCLabDashboard(make_adata(), batch_key="grp")  # Group By Batch: on
    step = d.processor.steps["Neighbors"]
    step.variable_controls["use_rep"].value = "Y_array2"
    assert step._reason == f"Group By Batch: {reason}"
