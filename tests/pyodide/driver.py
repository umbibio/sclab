"""Run by smoke.mjs inside Pyodide, after sclab and its dependencies are installed.

Builds the dashboard without scanpy and runs the steps that work natively
(QC through Neighbors, Cluster and UMAP), pseudobulk limma and edgeR
(edgepython without numba), and scrublet (installed without its annoy,
umap-learn and numba dependencies).
Any failed assertion fails the smoke test.
"""

import sys
import time
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sclab
import sclab.tools.cluster
from sclab import SCLabDashboard, _compat
from sclab.tools.differential_expression import _edgepython_kernels, _pseudobulk_pylimma


def ok(label, value=""):
    print(f"[OK] {label} {value}".rstrip())


assert sys.platform == "emscripten" and _compat.IS_WASM
assert find_spec("scanpy") is None, "the smoke test is meant to run without scanpy"
ok("import sclab", sclab.__version__)
sclab.optional_dependencies()

rng = np.random.default_rng(0)
n_obs, n_vars = 400, 200
obs = pd.DataFrame(
    {
        "batch": pd.Categorical(rng.choice(["b1", "b2"], n_obs)),
        "grp": pd.Categorical(rng.choice(["A", "B"], n_obs)),
    },
    index=[f"c{i}" for i in range(n_obs)],
)
X = sp.csr_matrix(rng.poisson(1.0, (n_obs, n_vars)).astype(np.float32))
adata = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]))
adata.layers["counts"] = adata.X.copy()
adata.obsm["X_pca"] = rng.normal(size=(n_obs, 20))

d = SCLabDashboard(adata, counts_layer="counts", batch_key="batch")
steps = d.processor.steps
assert steps["UMAP"]._reason == "Run the Neighbors step first"  # sclab's port
assert steps["QC"]._reason is steps["Preprocess"]._reason is None  # sclab's port
assert "Doublet Detection" in steps
assert d.background_checkbox.disabled and not d.background_checkbox.value
flavors = steps["Integration"].variable_controls["flavor"].options
assert flavors[-1] == "scanorama", flavors  # shown, but cannot run
ok("dashboard built", f"({len(steps)} steps)")


def run(name, **params):
    n_exc = len(d.broker.exceptions_log)
    t0 = time.perf_counter()
    steps[name].run(**params)
    assert d.broker.exceptions_log[n_exc:] == [], d.broker.exceptions_log[n_exc:]
    assert steps[name].run_button.button_style == "success", steps[name].status.value
    return f"({time.perf_counter() - t0:.2f} s)"


ok("QC", run("QC"))
c = steps["Preprocess"].variable_controls
c["hvg_methods"].value = ("seurat", "seurat_v3")
c["hvg_combine"].value = "union"
elapsed = run("Preprocess")
var = d.dataset.adata.var
hvg = var["highly_variable_seurat"] | var["highly_variable_seurat_v3"]
assert (var["highly_variable"] == hvg).all() and hvg.any()
ok("Preprocess seurat | seurat_v3", elapsed)
elapsed = run("PCA", n_comps=10, mask_var="highly_variable")
assert d.dataset.adata.obsm["X_pca"].shape == (d.dataset.adata.n_obs, 10)
ok("PCA", elapsed)
neighbors_kw = dict(use_rep="X_pca", dims=(1, 10), n_neighbors=15, group_by_batch=False)
elapsed = run("Neighbors", **neighbors_kw)  # Group By Batch needs bbknn
assert d.dataset.adata.obsp["connectivities"].nnz > 0
ok("Neighbors", elapsed)
elapsed = run("Cluster", resolution=1.0)
assert d.dataset.adata.obs["leiden"].nunique() > 0
ok("Cluster", elapsed)
backends = [e["params"]["backend"] for e in d.processor.step_history]
assert backends == ["native"] * 5, backends
assert d.generate_code().count('backend="native"') == 5
ok("history and code: backend native")

# timings of the native neighbors and Leiden on 10k cells (30 dims)
big = ad.AnnData(np.zeros((10_000, 1), np.float32))
centers = rng.normal(0, 3, (15, 30))
big.obsm["X_pca"] = (
    centers[rng.integers(0, 15, big.n_obs)] + rng.normal(size=(big.n_obs, 30))
).astype(np.float32)
t0 = time.perf_counter()
sclab.preprocess.neighbors(big, dims=(1, 30), n_neighbors=20)
t1 = time.perf_counter()
sclab.tools.cluster.leiden(big)
t2 = time.perf_counter()
n_clusters = big.obs["leiden"].nunique()
ok(
    "10k cells: neighbors, leiden",
    f"({t1 - t0:.2f} s, {t2 - t1:.2f} s; {n_clusters} clusters)",
)


elapsed = run("UMAP", n_components=2)
xy = d.dataset.adata.obsm["X_2Dumap"]
assert xy.shape == (d.dataset.adata.n_obs, 2) and np.isfinite(xy).all()
assert d.processor.step_history[-1]["params"]["backend"] == "native"
assert d.generate_code().count('backend="native"') == 6
ok("UMAP (sclab's port of umap-learn)", elapsed)


integration = dict(group_by="batch", reference_batch=None, max_iters=10)
run("Integration", use_rep=None, flavor="cca", **integration)
assert "X_cca" in d.dataset.adata.obsm
ok("Integration cca")
run("Integration", use_rep="X_pca", flavor="harmony", **integration)
assert "X_pca_harmony" in d.dataset.adata.obsm
ok("Integration harmony")

c = steps["Gene Expression"].variable_controls
c["layer"].value = "counts"
c["gene_input"].value = c["gene_input"].options[0]
run("Gene Expression")
assert d.plotter.state["source"] == "gene"
ok("Gene Expression")

steps["Integration"].variable_controls["flavor"].value = "scanorama"
assert "not available under Pyodide" in steps["Integration"]._reason
ok("Integration scanorama disabled", f"({steps['Integration']._reason})")

# scrublet installed with deps=False: no annoy, umap-learn or numba; exact neighbors
assert not any(find_spec(m) for m in ("annoy", "umap", "numba"))
# more genes than above: scrublet keeps the top 15% variable genes for 30 PCs
X2 = sp.csr_matrix(rng.negative_binomial(2, 0.5, (n_obs, 1000)).astype(np.float32))
d2 = SCLabDashboard(ad.AnnData(X2, layers={"counts": X2.copy()}), counts_layer="counts")
doublets = d2.processor.steps["Doublet Detection"]
assert doublets.variable_controls["flavor"].options == ("scrublet",)
scores = []
for _ in range(2):
    t0 = time.perf_counter()
    doublets.run(flavor="scrublet")
    assert d2.broker.exceptions_log == [], d2.broker.exceptions_log
    assert doublets.run_button.button_style == "success", doublets.status.value
    scores.append(d2.dataset.adata.obs["scrublet_score"].to_numpy())
    elapsed = f"({time.perf_counter() - t0:.2f} s)"
assert (scores[0] == scores[1]).all() and ((0 <= scores[0]) & (scores[0] <= 1)).all()
assert sp.issparse(d2.dataset.adata.layers["counts"])
ok("Doublet Detection scrublet (exact neighbors, reproducible)", elapsed)

de_kw = dict(reference_group="A", layer="counts", replicas_per_group=3, random_state=0)
assert find_spec("edgepython") is not None and find_spec("numba") is None
table = sclab.tools.de(adata.copy(), "grp", method="pseudobulk_limma", **de_kw)
assert {"gene", "logFC", "pval", "padj"} <= set(table.columns), table.columns
ok(
    "sclab.tools.de pseudobulk_limma (edgepython filterByExpr, TMM)",
    f"({len(table)} rows)",
)

# sclab's own filterByExpr and TMM, used when edgepython is not installed
_pseudobulk_pylimma.edgepython_is_available = lambda: False
port = sclab.tools.de(adata.copy(), "grp", method="pseudobulk_limma", **de_kw)
_pseudobulk_pylimma.edgepython_is_available = lambda: True
pd.testing.assert_frame_equal(port, table)
ok("sclab.tools.de pseudobulk_limma (sclab filterByExpr, TMM)", "(same table)")

table = sclab.tools.de(adata.copy(), "grp", method="pseudobulk_edger", **de_kw)
assert {"gene", "logFC", "pval", "padj", "F"} <= set(table.columns), table.columns
assert len(table) > 0 and table["pval"].between(0, 1).all()
import edgepython.glm_kernels  # noqa: E402  (imported by sclab without numba)

assert edgepython.glm_kernels.fit_leven_mat is _edgepython_kernels.fit_leven_mat
ok(
    "sclab.tools.de pseudobulk_edger (edgepython, NumPy kernels)",
    f"({len(table)} rows)",
)

de = steps["Differential Expression"]
c = de.variable_controls
c["groupby"].value = "grp"
assert "scanpy" not in (de._reason or ""), de._reason  # sclab's port of it
c["method"].value = "pseudobulk_limma"
c["reference"].value = "A"
assert de._reason is None, de._reason
de.run_button.click()  # a background step: runs inline under wasm
assert not d.broker.jobs.busy
assert de.run_button.button_style == "success", de.status.value
table = d.dataset.adata.uns["differential_expression"]["table"]
assert set(table["contrast"]) == {"B_vs_A"} and len(table) > 0
ok("Differential Expression step pseudobulk_limma", f"({len(table)} rows)")

lognorm = adata.copy()
lognorm.X = lognorm.X.log1p()
for method in ["wilcoxon", "t-test", "logreg"]:
    t0 = time.perf_counter()
    table = sclab.tools.de(lognorm, "grp", method=method, reference_group="A")
    backend = lognorm.uns["differential_expression"]["params"]["backend"]
    assert backend.startswith("sclab port"), backend
    assert set(table["contrast"]) == {"B_vs_A"} and len(table) == n_vars
    elapsed = f"({time.perf_counter() - t0:.2f} s)"
    ok(f"sclab.tools.de {method} (port of rank_genes_groups)", elapsed)

print("SMOKE TEST PASSED")
