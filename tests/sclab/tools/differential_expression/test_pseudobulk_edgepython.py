import random
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

from sclab.tools.differential_expression import (
    edgepython_is_available,
    edger_is_available,
    pseudobulk_edgepython,
    pseudobulk_edger,
)
from sclab.tools.utils import aggregate_and_filter

pytestmark = pytest.mark.skipif(
    not edgepython_is_available(), reason="edgepython not installed"
)

N_DE = 10
DE_GENES = {f"g{i}" for i in range(N_DE)}
TT_COLUMNS = ["logFC", "logCPM", "F", "PValue", "FDR"]


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
    adata = AnnData(
        csr_matrix(X),
        obs=pd.DataFrame({"grp": grp, "ct": ct}, index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _check_recovers_de(tt: pd.DataFrame):
    assert set(tt.index[:N_DE]) == DE_GENES
    assert (tt.loc[list(DE_GENES), "logFC"] > 1).all()
    assert (tt.loc[list(DE_GENES), "FDR"] < 0.01).all()


def test_one_vs_rest(adata):
    res = pseudobulk_edgepython(adata, "grp", random_state=0)

    assert set(res) == {"A_vs_rest", "B_vs_rest", "C_vs_rest"}
    tt = res["B_vs_rest"]
    assert list(tt.columns[:5]) == TT_COLUMNS
    assert tt.index.name == "gene_ids"
    _check_recovers_de(tt)
    assert "g199" not in tt.index  # filtered out by filterByExpr
    assert tt["PValue"].is_monotonic_increasing


def test_condition_vs_reference(adata):
    res = pseudobulk_edgepython(adata, "grp", condition_group="B", reference_group="A")

    assert list(res) == ["B_vs_A"]
    _check_recovers_de(res["B_vs_A"])


def test_cell_identity_stratifies_contrasts(adata):
    res = pseudobulk_edgepython(
        adata, "grp", reference_group="A", cell_identity_key="ct"
    )

    assert set(res) == {f"ct:{ct}|{c}_vs_A" for ct in ("T1", "T2") for c in ("B", "C")}
    _check_recovers_de(res["ct:T2|B_vs_A"])


def test_random_state_is_reproducible(adata):
    a = pseudobulk_edgepython(adata, "grp", condition_group="B", random_state=3)
    b = pseudobulk_edgepython(adata, "grp", condition_group="B", random_state=3)

    pd.testing.assert_frame_equal(a["B_vs_rest"], b["B_vs_rest"])


def test_reproducible_by_default(adata):
    a = pseudobulk_edgepython(adata, "grp", condition_group="B")
    b = pseudobulk_edgepython(adata, "grp", condition_group="B")

    pd.testing.assert_frame_equal(a["B_vs_rest"], b["B_vs_rest"])


@pytest.mark.skipif(not edger_is_available(), reason="R edgeR not available")
@pytest.mark.parametrize(
    "kwargs", [{}, {"cell_identity_key": "ct", "reference_group": "A"}]
)
def test_matches_r_edger(adata, kwargs: dict):
    random.seed(0)
    aggr = aggregate_and_filter(adata, "grp", kwargs.get("cell_identity_key"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        expected = pseudobulk_edger(aggr, "grp", aggregate=False, **kwargs)
    result = pseudobulk_edgepython(aggr, "grp", aggregate=False, **kwargs)

    assert sorted(result) == sorted(expected)
    for key, tt_r in expected.items():
        tt_py = result[key]
        assert set(tt_py.index) == set(tt_r.index)
        assert set(tt_py.index[:N_DE]) == set(tt_r.index[:N_DE])
        tt_py = tt_py.loc[tt_r.index]
        # agreement is ~1e-6: small differences in the dispersion trend fit
        for col in TT_COLUMNS:
            np.testing.assert_allclose(tt_py[col], tt_r[col], rtol=1e-3, atol=1e-5)
        for col in ["logFC", "F"]:
            assert pearsonr(tt_py[col], tt_r[col])[0] > 0.99999
        log_p = -np.log10(tt_py["PValue"]), -np.log10(tt_r["PValue"])
        assert pearsonr(*log_p)[0] > 0.99999
