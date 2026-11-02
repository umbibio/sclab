"""Layer / raw-count validation shared by the three pseudobulk functions."""

import pytest

from sclab.tools.differential_expression import (
    _pseudobulk_edger as edger_mod,
)
from sclab.tools.differential_expression import (
    _pseudobulk_limma as limma_mod,
)
from sclab.tools.differential_expression import (
    pseudobulk_edger,
    pseudobulk_limma,
    pseudobulk_pylimma,
    pylimma_is_available,
)

from ._fake_r import install_fake_r

KWARGS = dict(
    group_key="cond",
    condition_group="A",
    reference_group="B",
    replicas_per_group=2,
)


@pytest.fixture(params=["limma", "edger", "pylimma"])
def run(request, monkeypatch):
    """Call one pseudobulk function; the R ones run against a fake R."""
    name = request.param
    if name == "pylimma":
        if not pylimma_is_available():
            pytest.skip("pylimma not installed")

        def call(adata, **kw):
            return pseudobulk_pylimma(adata, edger_preprocessing=False, **KWARGS, **kw)

        return call

    pytest.importorskip("rpy2")
    pytest.importorskip("anndata2ri")
    func, mod, cols = {
        "limma": (pseudobulk_limma, limma_mod, ["logFC", "P.Value"]),
        "edger": (pseudobulk_edger, edger_mod, ["logFC", "PValue"]),
    }[name]
    fake = install_fake_r(monkeypatch, mod, cols)

    def call(adata, **kw):
        res = func(adata, **KWARGS, **kw)
        # expose what R received, to inspect the aggregated counts
        res["aggr"] = fake.assigned["aggr_adata"]
        return res

    return call


def _totals(res):
    """Total count of gene0 in condition A as seen by the aggregation."""
    if "aggr" in res:
        return res["aggr"].var["tot_expr_A"]["gene0"]
    return res["A_vs_B"].loc["gene0", "tot_expr_cnd"]


def test_default_uses_counts_layer_not_X(run, counts_adata):
    counts_adata.layers["counts"] = counts_adata.X.copy()
    expected = counts_adata.layers["counts"][
        (counts_adata.obs["cond"] == "A").values, 0
    ].sum()
    counts_adata.X = counts_adata.X.toarray() * 0.5 + 0.25  # normalized-looking X

    res = run(counts_adata)

    assert _totals(res) == expected


def test_layer_X_uses_adata_X(run, counts_adata):
    expected = counts_adata.X[(counts_adata.obs["cond"] == "A").values, 0].sum()
    counts_adata.layers["counts"] = counts_adata.X * 2

    res = run(counts_adata, layer="X")

    assert _totals(res) == expected


def test_normalized_layer_raises(run, counts_adata):
    counts_adata.layers["counts"] = counts_adata.X.toarray() * 0.5 + 0.25

    with pytest.raises(ValueError, match="does not look like raw counts"):
        run(counts_adata)


def test_check_counts_false_bypasses(run, counts_adata):
    counts_adata.layers["counts"] = counts_adata.X.toarray() * 0.5 + 0.25

    res = run(counts_adata, check_counts=False)

    assert res


def test_layer_none_raises(run, counts_adata):
    with pytest.raises(ValueError, match="layer='counts'.*layer='X'"):
        run(counts_adata, layer=None)


def test_missing_layer_raises(run, counts_adata):
    del counts_adata.layers["counts"]

    with pytest.raises(ValueError, match="layer 'counts' not found.*layer='X'"):
        run(counts_adata)
