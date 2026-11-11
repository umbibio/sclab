"""sclab's batched weighted fit against pylimma's per-gene _lm_series_slow."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab.tools.differential_expression import _pylimma_lm_series as F
from sclab.tools.differential_expression import pseudobulk_pylimma

pylimma = pytest.importorskip("pylimma")
lmfit = pytest.importorskip("pylimma.lmfit")
TESTED = pytest.mark.skipif(
    pylimma.__version__ not in F.TESTED_VERSIONS,
    reason="installed pylimma is not one the batched fit was checked against",
)


def _design(n_groups=4, n_reps=6):
    """~ 0 + group + replica, replicas crossed with groups."""
    group = np.repeat(np.arange(n_groups), n_reps)
    rep = np.tile(np.arange(n_reps), n_groups)
    return np.column_stack(
        [(group == g).astype(float) for g in range(n_groups)]
        + [(rep == r).astype(float) for r in range(1, n_reps)]
    )


def _compare(design, M, W):
    expected = lmfit.lm_series(M, design, W)
    seen = []
    slow = lmfit._lm_series_slow

    def spy(M, *args):
        seen.append(len(M))
        return slow(M, *args)

    lmfit._lm_series_slow = spy
    try:
        with F.fast_lm_series():
            got = lmfit.lm_series(M, design, W)
    finally:
        lmfit._lm_series_slow = slow
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], value, rtol=1e-10, atol=0, err_msg=key)
    return seen  # genes left to pylimma's loop


@TESTED
def test_matches_pylimma_with_missing_and_zero_weights():
    rng = np.random.default_rng(0)
    design = _design()
    M = rng.normal(5, 2, (200, len(design)))
    W = rng.uniform(0.05, 5, M.shape)
    M[3, 5] = np.nan  # missing value
    W[4, 7] = 0  # zero weight: dropped observation
    W[6] = 0  # no observations at all
    M[7] = np.inf
    assert _compare(design, M, W) == [4]


@TESTED
def test_rank_deficient_gene_goes_to_pylimma():
    rng = np.random.default_rng(1)
    x = rng.normal(size=12)
    design = np.column_stack([np.ones(12), x])
    M = rng.normal(size=(50, 12))
    W = rng.uniform(0.5, 2, M.shape)
    W[2, 0] = 1e20  # one sample dominates: R's dqrdc2 drops the slope column
    assert _compare(design, M, W) == [1]
    assert np.isnan(lmfit.lm_series(M, design, W)["coefficients"][2, 1])


@TESTED
def test_rank_deficient_design_and_no_df():
    rng = np.random.default_rng(2)
    design = _design(2, 2)
    _compare(design, rng.normal(size=(30, 4)), rng.uniform(0.5, 2, (30, 4)))
    collinear = np.column_stack([design, design[:, 0]])
    _compare(collinear, rng.normal(size=(30, 4)), rng.uniform(0.5, 2, (30, 4)))


def test_binding_is_scoped(monkeypatch: pytest.MonkeyPatch):
    slow = lmfit._lm_series_slow
    with pytest.raises(RuntimeError):
        with F.fast_lm_series():
            if pylimma.__version__ in F.TESTED_VERSIONS:
                assert lmfit._lm_series_slow is not slow
            raise RuntimeError
    assert lmfit._lm_series_slow is slow

    monkeypatch.setattr(F, "TESTED_VERSIONS", ())  # unknown version: unchanged
    with F.fast_lm_series():
        assert lmfit._lm_series_slow is slow


@TESTED
def test_top_tables_match_pylimma(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(3)
    n_genes, design = 500, _design()
    mu = np.exp(rng.normal(3, 2, n_genes))[:, None] * rng.uniform(0.5, 2, len(design))
    mu[:20, :6] *= 4
    counts = rng.negative_binomial(5, 5 / (5 + mu)).astype(float)
    counts[30] = 0  # dropped by filterByExpr
    obs = pd.DataFrame(
        {
            "grp": pd.Categorical(np.repeat([f"g{i}" for i in range(4)], 6)),
            "replica": pd.Categorical(np.tile([f"r{i}" for i in range(6)], 4)),
        },
        index=[f"s{i}" for i in range(len(design))],
    )
    adata = AnnData(counts.T, obs=obs)
    for g in obs["grp"].cat.categories:
        for c in ("pct_expr", "num_expr", "tot_expr"):
            adata.var[f"{c}_{g}"] = 1.0
    kw = dict(condition_group="g0", reference_group="g1", aggregate=False)

    fast = pseudobulk_pylimma(adata, "grp", **kw)["g0_vs_g1"]
    monkeypatch.setattr(F, "TESTED_VERSIONS", ())
    ref = pseudobulk_pylimma(adata, "grp", **kw)["g0_vs_g1"]
    cols = ["logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B"]
    pd.testing.assert_frame_equal(
        fast[cols], ref.loc[fast.index, cols], rtol=1e-9, atol=1e-12
    )
