import random
import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.tools.differential_expression import (
    edgepython_is_available,
    limma_is_available,
    pseudobulk_limma,
    pseudobulk_pylimma,
    pylimma_is_available,
)
from sclab.tools.utils import aggregate_and_filter

pytestmark = pytest.mark.skipif(
    not pylimma_is_available(), reason="pylimma not installed"
)
needs_edgepython = pytest.mark.skipif(
    not edgepython_is_available(), reason="edgepython not installed"
)

N_DE = 10
DE_GENES = {f"g{i}" for i in range(N_DE)}


@pytest.fixture(scope="module")
def adata() -> AnnData:
    rng = np.random.default_rng(0)
    n_obs, n_vars = 600, 200
    grp = np.repeat(["A", "B", "C"], n_obs // 3)
    ct = np.tile(["T1", "T2"], n_obs // 2)
    mu = np.full((n_obs, n_vars), 2.0)
    mu[grp == "B", :N_DE] *= 4  # first genes up in B
    X = rng.poisson(mu)
    X[:, -1] = 0  # never expressed
    return AnnData(
        csr_matrix(X),
        obs=pd.DataFrame({"grp": grp, "ct": ct}, index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


def _check_recovers_de(tt: pd.DataFrame):
    assert set(tt.index[:N_DE]) == DE_GENES
    assert (tt.loc[list(DE_GENES), "logFC"] > 1).all()
    assert (tt.loc[list(DE_GENES), "adj.P.Val"] < 0.01).all()


@needs_edgepython
def test_one_vs_rest(adata: AnnData):
    res = pseudobulk_pylimma(adata, "grp")

    assert set(res) == {"A_vs_rest", "B_vs_rest", "C_vs_rest"}
    tt = res["B_vs_rest"]
    assert list(tt.columns[:6]) == [
        "logFC",
        "AveExpr",
        "t",
        "P.Value",
        "adj.P.Val",
        "B",
    ]
    assert tt.index.name == "gene_ids"
    _check_recovers_de(tt)
    assert "g199" not in tt.index  # filtered out by filterByExpr


@needs_edgepython
def test_condition_vs_reference(adata: AnnData):
    res = pseudobulk_pylimma(adata, "grp", condition_group="B", reference_group="A")

    assert list(res) == ["B_vs_A"]
    tt = res["B_vs_A"]
    _check_recovers_de(tt)
    np.testing.assert_allclose(tt["mean_cnd"], tt["tot_expr_cnd"] / tt["num_expr_cnd"])


@needs_edgepython
def test_cell_identity_stratifies_contrasts(adata: AnnData):
    res = pseudobulk_pylimma(adata, "grp", reference_group="A", cell_identity_key="ct")

    assert set(res) == {f"ct:{ct}|{c}_vs_A" for ct in ("T1", "T2") for c in ("B", "C")}
    _check_recovers_de(res["ct:T2|B_vs_A"])


def test_without_edgepython(adata: AnnData, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "edgepython", None)  # import fails

    res = pseudobulk_pylimma(
        adata,
        "grp",
        condition_group="B",
        reference_group="A",
        edger_preprocessing=False,
    )

    tt = res["B_vs_A"]
    _check_recovers_de(tt)
    assert "g199" not in tt.index  # zero-count genes still dropped


def test_edger_preprocessing_requires_edgepython(
    adata: AnnData, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "sclab.tools.differential_expression._pseudobulk_pylimma.edgepython_is_available",
        lambda: False,
    )
    with pytest.raises(ImportError, match="edger_preprocessing=False"):
        pseudobulk_pylimma(adata, "grp")


@needs_edgepython
def test_no_residual_df_is_skipped(adata: AnnData, capsys):
    # one replica per group and one coefficient per group -> no residual df
    res = pseudobulk_pylimma(
        adata, "grp", condition_group="B", reference_group="A", replicas_per_group=1
    )

    assert res == {}
    assert "Skipping" in capsys.readouterr().out


@needs_edgepython
@pytest.mark.skipif(not limma_is_available(), reason="R limma not available")
@pytest.mark.parametrize(
    "kwargs", [{}, {"cell_identity_key": "ct", "reference_group": "A"}]
)
def test_matches_r_limma(adata: AnnData, kwargs: dict):
    random.seed(0)
    aggr = aggregate_and_filter(adata, "grp", kwargs.get("cell_identity_key"))

    expected = pseudobulk_limma(aggr, "grp", aggregate=False, **kwargs)
    result = pseudobulk_pylimma(aggr, "grp", aggregate=False, **kwargs)

    assert sorted(result) == sorted(expected)
    for key, tt_r in expected.items():
        tt_py = result[key].loc[tt_r.index]
        for col in ["logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B"]:
            np.testing.assert_allclose(tt_py[col], tt_r[col], rtol=1e-6, atol=1e-8)
