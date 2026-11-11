"""sclab's port of scanpy.tl.rank_genes_groups: its kernels against scipy, and
the whole function against scanpy (the results are bit-identical)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from scipy import stats

import sclab
from sclab.tools.differential_expression import _rank_genes_groups as native


def make_lognorm(n_obs=300, n_vars=120, sparse=True, negative=False, seed=0):
    """Log-normalized NB counts of 4 cell types (A-D), 5% of cells unlabeled,
    5 unexpressed genes; optionally negative values (e.g. after regression)."""
    rng = np.random.default_rng(seed)
    mu = rng.gamma(0.6, 2.0, n_vars)
    types = rng.integers(0, 4, n_obs)
    effect = rng.lognormal(0, 1.0, (4, n_vars)) * (rng.random((4, n_vars)) < 0.1) + 1
    lam = rng.lognormal(0, 0.5, n_obs)[:, None] * mu * effect[types]
    X = rng.negative_binomial(2, 2 / (2 + lam)).astype(np.float32)
    X[:, :5] = 0
    X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4).astype(np.float32)
    if negative:
        X[:, 5:30] -= 0.3 * (X[:, 5:30] > 0)
    grp = np.array(list("ABCD"))[types].astype(object)
    grp[rng.random(n_obs) < 0.05] = None
    return AnnData(
        sp.csr_matrix(X) if sparse else X,
        obs=pd.DataFrame(
            {"g": pd.Categorical(grp)}, index=[f"c{i}" for i in range(n_obs)]
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


@pytest.mark.parametrize("fmt", [sp.csr_matrix, sp.csc_matrix, np.asarray])
def test_rank_sums_match_scipy(fmt):
    rng = np.random.default_rng(1)
    dense = rng.choice([0, 0, 0, -1.5, 0.5, 1, 2.25], size=(57, 9)).astype(np.float32)
    dense[:, 0] = 0  # all tied
    dense[:, 1] = rng.normal(size=57)  # no zeros
    X = fmt(dense)
    if sp.issparse(X):
        X.data[:3] = 0  # explicit zeros
        dense = X.toarray()
    codes = rng.integers(0, 3, 57)
    sums, tc = native._rank_sums(X, codes, 3, tie_correct=True)
    ranks = stats.rankdata(dense, axis=0)
    for c in range(3):
        np.testing.assert_array_equal(sums[c], ranks[codes == c].sum(axis=0))
    expected = [stats.tiecorrect(ranks[:, j]) for j in range(dense.shape[1])]
    np.testing.assert_allclose(tc, expected, rtol=1e-14)
    _, no_tc = native._rank_sums(X, codes, 3, tie_correct=False)
    np.testing.assert_array_equal(no_tc, 1)


def test_rank_sums_in_gene_blocks(monkeypatch):
    X = make_lognorm(200, 50).X
    codes = np.arange(200) % 2
    whole = native._rank_sums(X, codes, 2, tie_correct=True)
    monkeypatch.setattr(native, "_NNZ_BLOCK", 37)  # blocks of 1 or a few genes
    for got, want in zip(native._rank_sums(X, codes, 2, True), whole, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("sparse", [True, False])
def test_group_mean_var(sparse):
    adata = make_lognorm(sparse=sparse, negative=True)
    codes = adata.obs["g"].cat.codes.to_numpy() + 1  # 0: unlabeled
    mean, var = native._group_mean_var(adata.X, codes, 5, need_var=True)
    dense = np.asarray(adata.X.toarray() if sparse else adata.X, dtype=np.float64)
    for c in range(5):
        np.testing.assert_allclose(mean[c], dense[codes == c].mean(0), atol=1e-14)
        np.testing.assert_allclose(var[c], dense[codes == c].var(0, ddof=1), atol=1e-13)
    sums_mean, _ = native._group_mean_var(adata.X, codes, 5, need_var=False)
    np.testing.assert_allclose(sums_mean, mean, atol=1e-14)


def test_fdr_bh_matches_scipy():
    p = np.random.default_rng(0).random(500) ** 3
    p[:20] = p[0]  # ties
    np.testing.assert_allclose(
        native._fdr_bh(p), stats.false_discovery_control(p), rtol=1e-14
    )


def test_unsupported_arguments():
    adata = make_lognorm()
    with pytest.raises(TypeError, match="pts"):
        native.rank_genes_groups(adata, "g", method="wilcoxon", pts=True)
    with pytest.raises(ValueError, match="one sample"):
        adata.obs["g"] = pd.Categorical(["A"] + ["B"] * (adata.n_obs - 1))
        native.rank_genes_groups(adata, "g", method="t-test")


def test_de_native_backend_records_citations():
    adata = make_lognorm()
    table = sclab.tools.de(adata, "g", method="wilcoxon", backend="native")
    assert set(table["contrast"]) == {f"{g}_vs_rest" for g in "ABCD"}
    params = adata.uns["differential_expression"]["params"]
    assert params["backend"].startswith("sclab port of scanpy.tl.rank_genes_groups")
    keys = list(adata.uns["sclab"]["citations"])
    assert {"scanpy", "wilcoxon1945", "mann1947", "benjamini1995"} <= set(keys)
    sclab.tools.de(adata, "g", method="t-test", backend="native", key_added="t")
    assert "welch1947" in adata.uns["sclab"]["citations"]


# --- against scanpy ----------------------------------------------------------

CASES = [
    dict(method="wilcoxon"),
    dict(method="wilcoxon", tie_correct=True),
    dict(method="wilcoxon", groups=["B", "C"], reference="A", tie_correct=True),
    dict(method="t-test"),
    dict(method="t-test_overestim_var", groups=["B", "D"]),
    dict(method="t-test", groups=["B"], reference="A"),
    dict(method="t-test", corr_method="bonferroni", n_genes=20, rankby_abs=True),
    dict(method="logreg", random_state=0, max_iter=500),
    dict(method="logreg", groups=["B"], reference="A", random_state=0),
]


def _exact(sc) -> bool:
    """scanpy >= 1.12.5, the version ported: bit-identical results. Older
    versions compute the t-test variances differently (~1e-8 apart), so near
    ties may swap and results are compared per gene."""
    from importlib.metadata import version

    from packaging.version import Version

    return Version(version("scanpy")) >= Version("1.12.5")


def _assert_tables_close(a: pd.DataFrame, b: pd.DataFrame, key: list[str]):
    key = [k for k in key if k in a]  # rank_genes_groups_df: "group" if several
    a, b = a.sort_values(key, ignore_index=True), b.sort_values(key, ignore_index=True)
    assert list(a.columns) == list(b.columns)
    for c in a.columns:
        if pd.api.types.is_float_dtype(a[c]):
            np.testing.assert_allclose(a[c], b[c], rtol=1e-5, atol=1e-7, err_msg=c)
        else:
            np.testing.assert_array_equal(a[c], b[c], err_msg=c)


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize(
    "kwargs", CASES, ids=lambda kw: "-".join(map(str, kw.values()))
)
def test_matches_scanpy(kwargs, sparse, negative):
    sc = pytest.importorskip("scanpy")
    adata = make_lognorm(sparse=sparse, negative=negative)
    ours, theirs = adata.copy(), adata.copy()
    native.rank_genes_groups(ours, "g", use_raw=False, **kwargs)
    sc.tl.rank_genes_groups(theirs, "g", use_raw=False, **kwargs)
    a, b = ours.uns["rank_genes_groups"], theirs.uns["rank_genes_groups"]
    assert a.keys() == b.keys()
    assert a["params"] == b["params"]
    for field in a.keys() - {"params"}:
        assert a[field].dtype == b[field].dtype, field
    df_a, df_b = (sc.get.rank_genes_groups_df(x, None) for x in (ours, theirs))
    if not _exact(sc):
        _assert_tables_close(df_a, df_b, ["group", "names"])
        return
    for field in a.keys() - {"params"}:
        for group in a[field].dtype.names:  # bit-identical, also the gene order
            np.testing.assert_array_equal(a[field][group], b[field][group])
    pd.testing.assert_frame_equal(df_a, df_b)


@pytest.mark.parametrize("method", ["wilcoxon", "t-test", "logreg"])
def test_de_table_matches_scanpy_backend(method):
    sc = pytest.importorskip("scanpy")
    adata = make_lognorm()
    kw = dict(method=method, cell_identity_key=None, reference_group="A")
    ours = sclab.tools.de(adata.copy(), "g", backend="native", **kw)
    theirs = sclab.tools.de(adata.copy(), "g", backend="scanpy", **kw)
    if _exact(sc):
        pd.testing.assert_frame_equal(ours, theirs, check_exact=True)
    else:
        _assert_tables_close(ours, theirs, ["contrast", "gene"])
