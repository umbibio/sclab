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
SCANPY_STEPS: list[str] = []  # Neighbors, Cluster, UMAP: sclab ports


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
    assert steps["QC"]._reason is steps["Preprocess"]._reason is None
    assert steps["Neighbors"]._reason.startswith("Choose a 'Use rep.'")
    assert steps["Cluster"]._reason == "Run the Neighbors step first"
    assert steps["PCA"]._reason.endswith("Run the Preprocess step first")
    assert steps["UMAP"]._reason == "Run the Neighbors step first"  # sclab's port

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


@pytest.mark.skipif(find_spec("skmisc") is None, reason="scikit-misc not installed")
def test_qc_preprocess_pca_run_without_scanpy(without_optional):
    d = SCLabDashboard(make_adata(), counts_layer="counts", batch_key="grp")
    run_step(d, "QC", min_counts=0, min_genes=0, min_cells=0)
    run_step(d, "Preprocess", hvg_methods=("cell_ranger", "seurat_v3"))
    run_step(d, "PCA", n_comps=5, mask_var="highly_variable", reference_batch="A")
    adata = d.dataset.adata
    var = adata.var
    assert (
        var["highly_variable"]
        == var["highly_variable_cell_ranger"] | var["highly_variable_seurat_v3"]
    ).all()
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)
    history = d.processor.step_history
    assert [e["params"]["backend"] for e in history] == ["native"] * 3
    code = d.generate_code()
    assert code.count('backend="native"') == 3 and "import scanpy" not in code
    assert sys.modules["scanpy"] is None  # not imported meanwhile


def test_umap_runs_without_scanpy(without_optional):
    from sklearn.neighbors import kneighbors_graph

    d = SCLabDashboard(make_adata(), counts_layer="counts")
    adata = d.dataset.adata
    graph = kneighbors_graph(adata.obsm["Y_array8"], 10)  # a stand-in for Neighbors
    adata.obsp["connectivities"] = graph.maximum(graph.T).tocsr()
    adata.uns["neighbors"] = {"params": {"method": "umap", "use_rep": "Y_array8"}}
    run_step(d, "UMAP", n_components=2)
    assert adata.obsm["X_2Dumap"].shape == (adata.n_obs, 2)
    assert d.processor.step_history[-1]["params"]["backend"] == "native"
    code = d.generate_code()
    assert "sclab.tools.embedding.umap(" in code
    assert 'backend="native"' in code and "import scanpy" not in code
    assert sys.modules["scanpy"] is None


def test_neighbors_cluster_run_without_scanpy(without_optional):
    d = SCLabDashboard(make_adata(), counts_layer="counts")
    run_step(d, "QC", min_counts=0, min_genes=0, min_cells=0)
    run_step(d, "Preprocess", hvg_methods=("cell_ranger",))
    run_step(d, "PCA", n_comps=10, mask_var="highly_variable")
    run_step(d, "Neighbors", use_rep="X_pca", dims=(1, 10), n_neighbors=15)
    run_step(d, "Cluster", resolution=0.5)
    adata = d.dataset.adata
    assert adata.uns["neighbors"]["params"]["n_neighbors"] == 15
    assert adata.obsp["connectivities"].nnz > 0
    assert adata.obs["leiden"].nunique() > 1
    history = d.processor.step_history
    assert [e["params"]["backend"] for e in history] == ["native"] * 5
    code = d.generate_code()
    assert (
        "sclab.preprocess.neighbors(" in code and "sclab.tools.cluster.leiden(" in code
    )
    assert code.count('backend="native"') == 5 and "import scanpy" not in code
    assert sys.modules["scanpy"] is None  # not imported meanwhile


def test_seurat_v3_hvg_needs_skmisc(monkeypatch):
    monkeypatch.setitem(sys.modules, "skmisc", None)
    reason = "HVG seurat_v3_paper needs skmisc: pip install scikit-misc"
    d = SCLabDashboard(make_adata(), counts_layer="counts", batch_key="grp")
    step = d.processor.steps["Preprocess"]
    assert step._reason == reason  # the default methods
    step.variable_controls["hvg_methods"].value = ("seurat", "cell_ranger")
    assert step._reason is None
    step.variable_controls["hvg_methods"].value = ()
    assert step._reason == "Choose at least one 'HVG methods'"

    pca = d.processor.steps["PCA"]
    pca._X_is_raw, pca._checked_X = False, d.dataset.adata.X
    pca.variable_controls["reference_batch"].value = "A"
    d.dataset.adata.var["highly_variable"] = True
    d.broker.publish("dset_var_dataframe_change", d.dataset.adata.var)
    pca.variable_controls["mask_var"].value = "highly_variable"
    assert pca._reason.endswith("need skmisc: pip install scikit-misc"), pca._reason
    with pytest.raises(ImportError, match="Needs skmisc"):
        sclab.preprocess.highly_variable_genes(make_adata(), backend="native")


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

    with monkeypatch.context() as m:
        m.setattr(_compat, "missing", lambda feature: None)
        assert flavors(SCLabDashboard(make_adata()))[-1] == "scanorama"


def test_scanorama_disabled_under_wasm(monkeypatch):
    monkeypatch.setitem(sys.modules, "scanorama", None)  # no wasm build
    monkeypatch.setattr(_compat, "IS_WASM", True)
    adata = make_adata()
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 5))
    step = SCLabDashboard(adata, batch_key="grp").processor.steps["Integration"]
    c = step.variable_controls
    assert c["flavor"].options[-1] == "scanorama"
    c["use_rep"].value = "X_pca"
    c["flavor"].value = "harmony"
    assert step._reason is None
    c["flavor"].value = "scanorama"
    reason = "Needs scanorama, scanpy: not available under Pyodide"
    assert step._reason.startswith(reason), step._reason
    assert step.run_button.disabled
    params = dict(use_rep="X_pca", group_by="grp", flavor="scanorama")
    with pytest.raises(ImportError, match=reason):
        step.function(**params, reference_batch=None, max_iters=10)


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
    assert "micropip.install('scrublet', deps=False)" in install["scrublet"]
    assert install["scanorama"].startswith("not available under Pyodide")
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
