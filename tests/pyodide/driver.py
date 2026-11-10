"""Run by smoke.mjs inside Pyodide, after sclab and its dependencies are installed.

Builds the dashboard without scanpy and runs the steps that work natively,
and pseudobulk limma and edgeR (edgepython without numba).
Any failed assertion fails the smoke test.
"""

import sys
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sclab
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
for name in ["QC", "Preprocess", "PCA", "Neighbors", "UMAP", "Cluster"]:
    assert steps[name].run_button.disabled, name
    assert steps[name]._reason.startswith("Needs scanpy"), steps[name]._reason
assert type(steps["Doublet Detection"]).__name__ == "_UnavailableStep"
assert d.background_checkbox.disabled and not d.background_checkbox.value
flavors = steps["Integration"].variable_controls["flavor"].options
assert "scanorama" not in flavors, flavors
ok("dashboard built", f"({len(steps)} steps)")


def run(name, **params):
    n_exc = len(d.broker.exceptions_log)
    steps[name].run(**params)
    assert d.broker.exceptions_log[n_exc:] == [], d.broker.exceptions_log[n_exc:]
    assert steps[name].run_button.button_style == "success", steps[name].status.value


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

de_kw = dict(reference_group="A", layer="counts", replicas_per_group=3, random_state=0)
assert find_spec("edgepython") is not None and find_spec("numba") is None
table = sclab.tools.de(adata.copy(), "grp", method="pseudobulk_limma", **de_kw)
assert {"gene", "logFC", "pval", "padj"} <= set(table.columns), table.columns
ok("sclab.tools.de pseudobulk_limma (edgepython filterByExpr, TMM)", f"({len(table)} rows)")

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
ok("sclab.tools.de pseudobulk_edger (edgepython, NumPy kernels)", f"({len(table)} rows)")

de = steps["Differential Expression"]
c = de.variable_controls
c["groupby"].value = "grp"
assert de._reason.startswith("wilcoxon needs scanpy"), de._reason
c["method"].value = "pseudobulk_limma"
c["reference"].value = "A"
assert de._reason is None, de._reason
de.run_button.click()  # a background step: runs inline under wasm
assert not d.broker.jobs.busy
assert de.run_button.button_style == "success", de.status.value
table = d.dataset.adata.uns["differential_expression"]["table"]
assert set(table["contrast"]) == {"B_vs_A"} and len(table) > 0
ok("Differential Expression step pseudobulk_limma", f"({len(table)} rows)")

print("SMOKE TEST PASSED")
