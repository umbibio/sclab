import inspect

import anndata
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

import sclab.tools
from sclab.tools.differential_expression import (
    differential_expression,
    edgepython_is_available,
    pseudobulk_edgepython,
    pseudobulk_edger,
    pseudobulk_limma,
    pseudobulk_pylimma,
    pylimma_is_available,
)
from sclab.tools.differential_expression._differential_expression import COLUMNS

try:
    import scanpy  # noqa: F401

    has_scanpy = True
except ImportError:
    has_scanpy = False

needs_scanpy = pytest.mark.skipif(not has_scanpy, reason="scanpy not installed")
needs_pylimma = pytest.mark.skipif(
    not (pylimma_is_available() and edgepython_is_available()),
    reason="pylimma/edgepython not installed",
)
needs_edgepython = pytest.mark.skipif(
    not edgepython_is_available(), reason="edgepython not installed"
)

SCANPY = [pytest.param(m, marks=needs_scanpy) for m in ("wilcoxon", "t-test", "logreg")]
PSEUDOBULK = [
    pytest.param("pseudobulk_limma", marks=needs_pylimma),
    pytest.param("pseudobulk_edger", marks=needs_edgepython),
]
EXTRAS = {
    "wilcoxon": [],
    "t-test": [],
    "logreg": [],
    "pseudobulk_limma": ["AveExpr", "t", "B"],
    "pseudobulk_edger": ["logCPM", "F"],
}
DE_GENES = {f"g{i}" for i in range(10)}


@pytest.fixture
def adata() -> AnnData:
    """3 groups x 2 cell types, 2 samples per group; g0-g9 up 4x in B."""
    rng = np.random.default_rng(0)
    n_obs, n_vars = 480, 200
    grp = np.repeat(["A", "B", "C"], n_obs // 3)
    ct = np.tile(["T1", "T2"], n_obs // 2)
    sample = np.char.add(grp, np.tile(np.repeat(["1", "2"], 2), n_obs // 4))
    mu = np.full((n_obs, n_vars), 2.0)
    mu[grp == "B", :10] *= 4
    counts = rng.poisson(mu).astype(float)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    obs = pd.DataFrame(
        {"grp": grp, "ct": ct, "sample": sample},
        index=[f"c{i}" for i in range(n_obs)],
    )
    adata = AnnData(
        csr_matrix(lognorm),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    adata.layers["counts"] = csr_matrix(counts)
    return adata


def _contrast(df: pd.DataFrame, contrast: str) -> pd.DataFrame:
    return df[df["contrast"] == contrast].set_index("gene")


@pytest.mark.parametrize("method", SCANPY + PSEUDOBULK)
def test_each_method_recovers_planted_genes(adata, method):
    df = differential_expression(adata, "grp", method=method)

    assert (
        list(df.columns)
        == [c for c in COLUMNS if c != "cell_identity"] + EXTRAS[method]
    )
    assert set(df["contrast"]) == {"A_vs_rest", "B_vs_rest", "C_vs_rest"}
    b = _contrast(df, "B_vs_rest")
    top = set(b.sort_values("score", ascending=False).index[:10])
    assert top == DE_GENES
    assert (b.loc[list(DE_GENES), "logFC"] > 1).all()
    assert (b.loc[list(DE_GENES), "mean_cnd"] > b.loc[list(DE_GENES), "mean_ref"]).all()
    if method == "logreg":
        assert df["pval"].isna().all() and df["padj"].isna().all()
    else:
        assert (b.loc[list(DE_GENES), "padj"] < 0.01).all()
    numeric = df.columns.drop(["contrast", "gene"])
    assert all(df[c].dtype == np.float64 for c in numeric)


@pytest.mark.parametrize("method", SCANPY[:1] + PSEUDOBULK)
def test_reference_group(adata, method):
    df = differential_expression(adata, "grp", method=method, reference_group="A")

    assert set(df["contrast"]) == {"B_vs_A", "C_vs_A"}
    b = _contrast(df, "B_vs_A")
    a_cells = (adata.obs["grp"] == "A").to_numpy()
    counts = adata.layers["counts"].toarray()
    np.testing.assert_allclose(
        b["pct_ref"],
        (counts[a_cells] > 0).mean(axis=0)[adata.var_names.get_indexer(b.index)],
    )
    assert set(b.sort_values("score", ascending=False).index[:10]) == DE_GENES


@needs_scanpy
def test_condition_group_subset_vs_rest(adata):
    df = differential_expression(adata, "grp", condition_group="B")

    assert set(df["contrast"]) == {"B_vs_rest"}
    rest = (adata.obs["grp"] != "B").to_numpy()
    b = _contrast(df, "B_vs_rest")
    X = adata.X.toarray()[:, adata.var_names.get_indexer(b.index)]
    np.testing.assert_allclose(b["mean_ref"], X[rest].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(b["pct_ref"], (X[rest] > 0).mean(axis=0))


@pytest.mark.parametrize("method", SCANPY[:1] + PSEUDOBULK[:1])
def test_cell_identity_key(adata, method):
    df = differential_expression(
        adata, "grp", method=method, reference_group="A", cell_identity_key="ct"
    )

    assert list(df.columns[:3]) == ["contrast", "cell_identity", "gene"]
    pairs = set(zip(df["cell_identity"], df["contrast"]))
    assert pairs == {(ct, c) for ct in ("T1", "T2") for c in ("B_vs_A", "C_vs_A")}
    t1 = df[(df["cell_identity"] == "T1") & (df["contrast"] == "B_vs_A")].set_index(
        "gene"
    )
    cells = ((adata.obs["grp"] == "B") & (adata.obs["ct"] == "T1")).to_numpy()
    counts = adata.layers["counts"].toarray()
    expected = (counts[cells] > 0).mean(axis=0)[adata.var_names.get_indexer(t1.index)]
    np.testing.assert_allclose(t1["pct_cnd"], expected)
    assert set(t1.sort_values("score", ascending=False).index[:10]) == DE_GENES


@needs_pylimma
def test_cell_identity_rest_is_within_identity(adata):
    df = differential_expression(
        adata, "grp", method="pseudobulk_limma", cell_identity_key="ct"
    )

    t1 = df[(df["cell_identity"] == "T1") & (df["contrast"] == "B_vs_rest")]
    cells = ((adata.obs["grp"] != "B") & (adata.obs["ct"] == "T1")).to_numpy()
    counts = adata.layers["counts"].toarray()
    expected = (counts[cells] > 0).mean(axis=0)[adata.var_names.get_indexer(t1["gene"])]
    np.testing.assert_allclose(t1["pct_ref"], expected)


@pytest.mark.parametrize("method", PSEUDOBULK)
def test_sample_key(adata, method):
    df = differential_expression(
        adata, "grp", method=method, reference_group="A", sample_key="sample"
    )

    assert set(df["contrast"]) == {"B_vs_A", "C_vs_A"}
    params = adata.uns["differential_expression"]["params"]
    assert params["sample_key"] == "sample"
    b = _contrast(df, "B_vs_A")
    assert set(b.sort_values("pval").index[:10]) == DE_GENES


@needs_pylimma
def test_sample_key_needs_two_samples_per_side(adata):
    adata.obs["sample"] = adata.obs["grp"]  # one sample per group

    with pytest.raises(ValueError, match="at least 2 samples"):
        differential_expression(
            adata, "grp", method="pseudobulk_limma", sample_key="sample"
        )


@needs_scanpy
def test_sample_key_rejected_for_scanpy_methods(adata):
    with pytest.raises(ValueError, match="pseudobulk"):
        differential_expression(adata, "grp", sample_key="sample")


@needs_scanpy
def test_scanpy_methods_default_to_X_and_refuse_counts(adata):
    differential_expression(adata, "grp")
    assert adata.uns["differential_expression"]["params"]["layer"] == "X"

    with pytest.raises(ValueError, match="raw counts"):
        differential_expression(adata, "grp", layer="counts")

    adata.X = adata.layers["counts"].copy()
    with pytest.raises(ValueError, match="raw counts"):
        differential_expression(adata, "grp", method="t-test")


@needs_pylimma
def test_pseudobulk_defaults_to_counts_and_refuses_lognorm(adata):
    differential_expression(adata, "grp", method="pseudobulk_limma")
    assert adata.uns["differential_expression"]["params"]["layer"] == "counts"

    with pytest.raises(ValueError, match="raw counts"):
        differential_expression(adata, "grp", method="pseudobulk_limma", layer="X")


def test_unknown_method_and_group(adata):
    with pytest.raises(ValueError, match="method must be one of"):
        differential_expression(adata, "grp", method="deseq2")
    with pytest.raises(ValueError, match="not found"):
        differential_expression(adata, "grp", reference_group="Z")


@needs_scanpy
def test_logreg_score_is_coefficient_without_pvalues(adata):
    df = differential_expression(adata, "grp", method="logreg", reference_group="A")

    b = _contrast(df, "B_vs_A")
    assert b["pval"].isna().all() and b["padj"].isna().all()
    assert (b.loc[list(DE_GENES), "score"] > 0).all()  # sign follows the condition
    a = _contrast(
        differential_expression(
            adata, "grp", method="logreg", condition_group="A", reference_group="B"
        ),
        "A_vs_B",
    )
    assert (a.loc[list(DE_GENES), "score"] < 0).all()


@needs_scanpy
def test_leaves_adata_untouched_apart_from_key_added(adata):
    obs = adata.obs.copy()

    differential_expression(adata, "grp", key_added="de")

    assert list(adata.uns) == ["de"]
    pd.testing.assert_frame_equal(adata.obs, obs)


@pytest.mark.parametrize(
    "method, kwargs",
    [
        pytest.param("wilcoxon", {}, marks=needs_scanpy),
        pytest.param(
            "pseudobulk_limma", {"cell_identity_key": "ct"}, marks=needs_pylimma
        ),
    ],
)
def test_uns_h5ad_round_trip(adata, tmp_path, method, kwargs):
    df = differential_expression(adata, "grp", method=method, **kwargs)
    path = tmp_path / "de.h5ad"

    adata.write_h5ad(path)
    back = anndata.read_h5ad(path).uns["differential_expression"]

    pd.testing.assert_frame_equal(
        back["table"].reset_index(drop=True),
        df,
        check_dtype=False,
        check_index_type=False,
    )
    assert all(back["table"][c].dtype == np.float64 for c in ("logFC", "score", "pval"))
    params = back["params"]
    assert params["method"] == method
    assert params["reference_group"] == "rest"
    assert "sclab" in params["versions"]


@pytest.mark.parametrize("method", PSEUDOBULK[:1] + SCANPY[:1])
def test_reproducible_with_random_state(adata, method):
    a = differential_expression(adata, "grp", method=method)
    b = differential_expression(adata, "grp", method=method)
    pd.testing.assert_frame_equal(a, b)


def test_de_is_the_function_and_subpackage_still_imports(adata):
    import sclab.tools.differential_expression as subpackage

    assert sclab.tools.de is differential_expression
    assert subpackage.differential_expression is differential_expression
    assert subpackage.pseudobulk_pylimma.__name__ == "pseudobulk_pylimma"
    assert not callable(subpackage)
    assert list(inspect.signature(sclab.tools.de).parameters)[:2] == [
        "adata",
        "group_key",
    ]
    if has_scanpy:
        pd.testing.assert_frame_equal(
            sclab.tools.de(adata, "grp"), differential_expression(adata, "grp")
        )


def test_default_pseudo_replicas():
    def default(f):
        return inspect.signature(f).parameters["replicas_per_group"].default

    assert default(pseudobulk_pylimma) == default(pseudobulk_edgepython) == 3
    assert default(pseudobulk_limma) == default(pseudobulk_edger) == 5  # deprecated
