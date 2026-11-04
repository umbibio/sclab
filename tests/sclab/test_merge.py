"""Merging datasets: the preview, the merge, depth thinning, the Workspace merge."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from sclab import Workspace
from sclab._workspace import MergeDialog, workspace_to_code
from sclab.dataset.processor._history import loads_uns
from sclab.preprocess import downsample_counts, merge_datasets, merge_preview
from sclab.utils import looks_like_raw_counts

from .test_utils import run_step, simple_loop_adata


def dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def make(n_obs=200, n_vars=50, scale=1, genes=None):
    adata = simple_loop_adata(n_obs=n_obs, n_vars=n_vars)
    adata.X = adata.X * scale
    if genes is not None:
        adata.var_names = genes
    adata.layers["counts"] = adata.X.copy()
    return adata


def totals(X):
    return np.asarray(X.sum(axis=1)).ravel()


# --------------------------------------------------------------------------
# merge_preview
# --------------------------------------------------------------------------
def test_preview_ok_with_depth_stats():
    preview = merge_preview({"A": make(scale=1), "B": make(scale=4)})
    assert preview["errors"] == [] and preview["warnings"] == []
    assert preview["n_obs"] == 400 and preview["n_vars"] == 50
    a, b = preview["datasets"]["A"], preview["datasets"]["B"]
    assert a["counts"] == b["counts"] == "ok"
    assert a["id_kind"] == "symbol" and a["genes_dropped"] == 0
    assert b["median_counts"] == pytest.approx(4 * a["median_counts"])
    assert a["depth_ratio"] == 1 and b["depth_ratio"] == pytest.approx(4)
    assert b["median_genes"] == a["median_genes"]


def test_preview_disjoint_genes():
    a = make(genes=[f"a{i}" for i in range(50)])
    b = make(genes=[f"b{i}" for i in range(50)])
    inner = merge_preview({"A": a, "B": b})
    assert inner["shared_genes"] == 0 and inner["n_vars"] == 0
    assert any("share no genes" in e for e in inner["errors"])
    assert inner["datasets"]["A"]["genes_dropped"] == 50

    outer = merge_preview({"A": a, "B": b}, join="outer")
    assert outer["errors"] == [] and outer["n_vars"] == 100
    assert outer["datasets"]["A"]["genes_filled"] == 50


def test_preview_ensembl_vs_symbols():
    a = make(genes=[f"ENSG{i:011d}" for i in range(50)])
    b = make()
    preview = merge_preview({"A": a, "B": b})
    assert preview["datasets"]["A"]["id_kind"] == "ensembl"
    assert preview["datasets"]["B"]["id_kind"] == "symbol"
    (error,) = [e for e in preview["errors"] if "Ensembl" in e]
    assert "gene_ids" not in error

    b.var["gene_ids"] = a.var_names.to_numpy()
    (error,) = [e for e in merge_preview({"A": a, "B": b})["errors"] if "Ensembl" in e]
    assert "B has a var 'gene_ids' column" in error


def test_preview_counts_checks():
    a, b = make(), make()
    del b.layers["counts"]
    preview = merge_preview({"A": a, "B": b})
    assert preview["datasets"]["B"]["counts"] == "missing"
    assert any("B: no 'counts' layer" in e for e in preview["errors"])

    b.layers["counts"] = b.X * 0.5
    preview = merge_preview({"A": a, "B": b})
    assert preview["datasets"]["B"]["counts"] == "not integer"
    assert preview["errors"] == [] and any(
        "not integer" in w for w in preview["warnings"]
    )
    preview = merge_preview({"A": a, "B": b}, equalize_depth=True)
    assert any("cannot be equalized" in e for e in preview["errors"])


def test_preview_obs_conflicts_and_label_clash():
    a, b = make(), make()
    a.obs["kind"] = pd.Categorical(["x"] * a.n_obs)
    b.obs["kind"] = pd.Categorical(["y"] * b.n_obs)
    b.obs["poisson"] = b.obs["poisson"].astype(str)
    preview = merge_preview({"A": a, "B": b})
    assert preview["obs_conflicts"]["kind"] == "categories differ"
    assert "A: int64" in preview["obs_conflicts"]["poisson"]
    assert "quadrant" not in preview["obs_conflicts"]
    assert preview["errors"] == [] and len(preview["warnings"]) == 2

    a.obs["sample"] = "s1"
    preview = merge_preview({"A": a, "B": b})
    assert any("A: obs already has a 'sample' column" in e for e in preview["errors"])
    assert merge_preview({"A": a, "B": b}, label="dataset")["errors"] == []


def test_preview_needs_two_datasets():
    assert any("at least two" in e for e in merge_preview({"A": make()})["errors"])


# --------------------------------------------------------------------------
# merge_datasets
# --------------------------------------------------------------------------
@pytest.mark.parametrize("join", ["inner", "outer"])
def test_merge(join):
    a = make(n_obs=200, genes=[f"g{i}" for i in range(50)])
    b = make(n_obs=100, genes=[f"g{i}" for i in range(25, 75)])
    a.obsm["X_pca"] = np.zeros((a.n_obs, 2))
    a.layers["log1p"] = a.X.copy()
    a.X = a.X * 0.1  # normalized: the merge takes the counts layer
    before = {
        n: (ad.shape, dense(ad.X).copy(), dict(ad.uns))
        for n, ad in [("A", a), ("B", b)]
    }

    merged = merge_datasets({"A": a, "B": b}, join=join)
    assert merged.shape == (300, 25 if join == "inner" else 75)
    assert merged.obs["sample"].tolist() == ["A"] * 200 + ["B"] * 100
    assert merged.obs_names[0] == f"{a.obs_names[0]}-A"
    assert (
        set(merged.layers) - {None} == {"counts"} and len(merged.obsm) == 0
    )  # 0.13 lists X as None
    np.testing.assert_array_equal(dense(merged.X), dense(merged.layers["counts"]))
    assert merged.X is not merged.layers["counts"]
    g = merged.var_names.get_loc("g30")
    expected = dense(a.layers["counts"][:, a.var_names.get_loc("g30")]).ravel()
    np.testing.assert_array_equal(dense(merged.X[:200, g]).ravel(), expected)
    if join == "outer":
        assert dense(merged.X[200:, merged.var_names.get_loc("g0")]).sum() == 0

    for name, ad in [("A", a), ("B", b)]:  # parents unchanged
        shape, X, uns = before[name]
        assert ad.shape == shape and dict(ad.uns) == uns
        np.testing.assert_array_equal(dense(ad.X), X)

    (entry,) = loads_uns(merged)
    params = entry["params"]
    assert entry["step"] == "merge" and entry["shape_after"] == list(merged.shape)
    assert [p["name"] for p in params["parents"]] == ["A", "B"]
    assert params["keys"] == ["A", "B"] and params["join"] == join
    assert params["index_unique"] == "-" and params["equalize_depth"] is False


def test_merge_refuses_blocking_errors():
    a = make()
    a.obs["sample"] = "x"
    with pytest.raises(ValueError, match="already has a 'sample' column"):
        merge_datasets({"A": a, "B": make()})


def test_merge_equalize_depth():
    a, b = make(n_obs=300, scale=1), make(n_obs=300, scale=4)
    merged = merge_datasets({"A": a, "B": b}, equalize_depth=True, random_state=3)
    t = totals(merged.layers["counts"])
    ratio = np.median(t[300:]) / np.median(t[:300])
    assert ratio == pytest.approx(1, abs=0.05)
    assert looks_like_raw_counts(merged.X)
    np.testing.assert_array_equal(dense(merged.X), dense(merged.layers["counts"]))
    np.testing.assert_array_equal(dense(merged.X[:300]), dense(a.layers["counts"]))
    np.testing.assert_array_equal(
        dense(b.layers["counts"]), dense(make(300, scale=4).X)
    )
    params = loads_uns(merged)[0]["params"]
    assert params["equalize_depth"] is True and params["random_state"] == 3

    again = merge_datasets({"A": a, "B": b}, equalize_depth=True, random_state=3)
    np.testing.assert_array_equal(dense(again.X), dense(merged.X))


# --------------------------------------------------------------------------
# downsample_counts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sparse", [True, False])
def test_downsample_counts_ratio(sparse):
    adata = make(n_obs=400, scale=10)
    if not sparse:
        adata.X = dense(adata.X)
    before = totals(adata.X)
    downsample_counts(adata, 0.25, random_state=0)
    assert issparse(adata.X) == sparse
    assert looks_like_raw_counts(adata.X)
    assert totals(adata.X).sum() / before.sum() == pytest.approx(0.25, rel=0.02)
    assert np.all(dense(adata.X) <= dense(adata.layers["counts"]))


def test_downsample_counts_by_group_and_layer():
    adata = make(n_obs=400, scale=10)
    adata.obs["grp"] = np.where(np.arange(400) < 200, "deep", "shallow")
    X = dense(adata.X).copy()
    downsample_counts(adata, {"deep": 0.5}, by="grp", layer="counts")
    np.testing.assert_array_equal(dense(adata.X), X)  # X untouched
    t = totals(adata.layers["counts"])
    assert t[:200].sum() / X[:200].sum() == pytest.approx(0.5, rel=0.03)
    np.testing.assert_array_equal(dense(adata.layers["counts"])[200:], X[200:])

    adata.X = adata.X * 0.01
    with pytest.raises(ValueError, match="raw counts"):
        downsample_counts(adata, 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        downsample_counts(adata, 2.0, layer="counts")


# --------------------------------------------------------------------------
# Workspace.merge
# --------------------------------------------------------------------------
@pytest.fixture
def files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    paths = {}
    for name, n_obs, scale in [("a", 300, 1), ("b", 200, 3)]:
        paths[name] = data / f"{name}.h5ad"
        adata = simple_loop_adata(n_obs=n_obs, n_vars=50)
        adata.X = adata.X * scale
        adata.write_h5ad(paths[name])
    return paths


def test_workspace_merge_keeps_parents(files, tmp_path):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    run_step(ws.dash, "QC")
    n_a = ws.dash.ds.adata.n_obs
    a_history = ws.history("a")
    assert [e["step"] for e in a_history] == ["load", "QC"]
    assert a_history[0]["params"] == {"source": str(files["a"]), "name": "a"}

    entry = ws.merge(["a", "b"], "ab")
    assert ws.datasets["b"].adata is None  # read for the merge, freed again
    assert ws.history("a") == a_history and ws.active == "a"
    assert entry.parents == ["a", "b"] and entry.source is None
    assert entry.batch_key == "sample" and entry.shape == (n_a + 200, 50)
    merge = loads_uns(entry.adata)[0]["params"]
    assert [p["source"] for p in merge["parents"]] == [str(f) for f in files.values()]

    ws.activate("ab")
    assert ws.dash.batch_key == "sample" and ws.dash.processor.batch_key == "sample"
    assert ws.datasets["a"].cache_path is not None  # a was edited: written
    ws.activate("a")
    assert ws.datasets["ab"].adata is not None  # clean merge: kept in memory
    ws.activate("ab")
    run_step(ws.dash, "Preprocess")
    ws.activate("b")
    ab = ws.datasets["ab"]
    assert ab.adata is None and ab.cache_path == tmp_path / "ws" / "ab.h5ad"
    ws.activate("ab")
    assert [e["step"] for e in ws.dash.processor.step_history] == [
        "merge",
        "Preprocess",
    ]


def test_workspace_merge_without_workspace_dir_keeps_it(files):
    ws = Workspace(files)
    ws.merge(["a", "b"], "ab")
    ws.activate("ab")
    run_step(ws.dash, "Preprocess")
    ws.activate("b")  # dirty, but kept in memory: no refusal
    assert ws.datasets["ab"].adata is not None
    ws.rename("a", "first")
    assert ws.datasets["ab"].parents == ["first", "b"]


def exec_script(code: str):
    namespace = {}
    exec(code, namespace)
    return namespace["adata"]


@pytest.mark.parametrize("equalize_depth", [False, True])
def test_workspace_codegen_roundtrip(files, tmp_path, equalize_depth):
    ws = Workspace(files, workspace_dir=tmp_path / "ws")
    run_step(ws.dash, "QC")
    ws.merge(["a", "b"], "ab", equalize_depth=equalize_depth)
    ws.activate("a")
    run_step(ws.dash, "QC", min_counts=10)  # after the merge: not in its script
    ws.activate("ab")
    run_step(ws.dash, "Preprocess")

    code = ws.dash.generate_code()
    assert code == workspace_to_code(ws, "ab")
    assert ws.dash.processor.code_actions.text == code  # the History tab's script
    assert 'adatas["a"] = adata' in code and 'adatas["b"] = adata' in code
    assert code.count("# QC\n") == 1 and "# a changed" not in code
    assert f"equalize_depth={equalize_depth}" in code

    result = exec_script(code)
    adata = ws.dash.ds.adata
    assert result.shape == adata.shape
    assert result.obs_names.equals(adata.obs_names)
    assert result.obs["sample"].tolist() == adata.obs["sample"].tolist()
    np.testing.assert_allclose(dense(result.X), dense(adata.X), rtol=1e-6)
    np.testing.assert_array_equal(
        dense(result.layers["counts"]), dense(adata.layers["counts"])
    )


def test_merge_dialog_preview_and_create(files):
    ws = Workspace(files)
    ws.merge_button.click()
    (dialog,) = ws.loader_box.children
    assert isinstance(dialog, MergeDialog)
    assert dialog.create_button.disabled

    dialog.checks["a"].value = True
    dialog.preview_button.click()
    assert "at least two" in dialog.output.value and dialog.create_button.disabled

    dialog.checks["b"].value = True
    dialog.label.value = "quadrant"  # an obs column of both: blocked
    dialog.preview_button.click()
    assert "color:#b91c1c" in dialog.output.value and dialog.create_button.disabled

    dialog.label.value = "batch"
    dialog.equalize_depth.value = True
    dialog.preview_button.click()
    assert "<table" in dialog.output.value and "Depth ratio" in dialog.output.value
    assert not dialog.create_button.disabled
    dialog.join.value = "outer"  # a changed choice needs a new preview
    assert dialog.create_button.disabled
    dialog.preview_button.click()

    dialog.new_name.value = "a"  # taken: the error is shown, nothing changes
    dialog.create_button.click()
    assert "already exists" in dialog.output.value and list(ws.datasets) == ["a", "b"]

    dialog.new_name.value = "ab"
    dialog.create_button.click()
    assert ws.active == "ab" and ws.dash.batch_key == "batch"
    assert ws.loader_box.children == () and dialog.comm is None
    params = loads_uns(ws.dash.ds.adata)[0]["params"]
    assert params["join"] == "outer" and params["equalize_depth"] is True
