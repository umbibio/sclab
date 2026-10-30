import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.tools.cellflow import expression_dynamics

GENES = ["up_early", "up_mid", "down_late", "up_late", "flat"]


def _sigmoid(t, onset, rate=30.0, sign=1.0):
    return 1.0 / (1.0 + np.exp(-sign * (t - onset) * rate))


@pytest.fixture
def times():
    return np.random.default_rng(0).uniform(0, 1, 400)


@pytest.fixture
def adata(times) -> AnnData:
    X = np.column_stack(
        [
            _sigmoid(times, 0.25),
            _sigmoid(times, 0.50),
            _sigmoid(times, 0.75, sign=-1.0),
            _sigmoid(times, 0.80),
            np.full_like(times, 0.5),
        ]
    ).astype(np.float32)
    obs = pd.DataFrame({"pt": times}, index=[f"c{i}" for i in range(times.size)])
    var = pd.DataFrame(index=GENES)
    return AnnData(X, obs=obs, var=var)


def test_writes_expected_columns(adata):
    expression_dynamics(adata, "pt", t_range=(0, 1))

    assert {"n_activation", "n_repression", "transcriptional_flux"} <= set(
        adata.obs.columns
    )
    assert {
        "peak_activation_t",
        "peak_repression_t",
        "acceleration_onset_t",
        "deceleration_onset_t",
    } <= set(adata.var.columns)
    assert adata.obs["n_activation"].between(0, adata.n_vars).all()
    assert adata.obs["n_repression"].between(0, adata.n_vars).all()
    assert (adata.obs["transcriptional_flux"] >= 0).all()
    assert adata.var.index.tolist() == GENES


def test_peak_activation_matches_sigmoid_onsets(adata):
    expression_dynamics(adata, "pt", t_range=(0, 1))

    peak = adata.var["peak_activation_t"]
    assert peak["up_early"] == pytest.approx(0.25, abs=0.06)
    assert peak["up_mid"] == pytest.approx(0.50, abs=0.06)
    assert peak["up_late"] == pytest.approx(0.80, abs=0.06)


def test_peak_repression_matches_downward_sigmoid(adata):
    expression_dynamics(adata, "pt", t_range=(0, 1))

    assert adata.var.loc["down_late", "peak_repression_t"] == pytest.approx(
        0.75, abs=0.06
    )


def test_second_derivative_extrema_flank_the_inflection(adata):
    expression_dynamics(adata, "pt", t_range=(0, 1))

    row = adata.var.loc["up_mid"]
    assert row["acceleration_onset_t"] < row["peak_activation_t"]
    assert row["peak_activation_t"] < row["deceleration_onset_t"]
    # a logistic with rate 30 has curvature extrema ~ +-0.577/30 from the onset
    assert row["peak_activation_t"] - row["acceleration_onset_t"] < 0.15


def test_timing_columns_lie_within_t_range(adata):
    expression_dynamics(adata, "pt", t_range=(0, 1), n_grid=201)

    cols = [
        "peak_activation_t",
        "peak_repression_t",
        "acceleration_onset_t",
        "deceleration_onset_t",
    ]
    vals = adata.var[cols].values
    assert np.isfinite(vals).all()
    assert (vals >= 0).all() and (vals <= 1).all()
    # values are drawn from the evaluation grid
    grid = np.linspace(0, 1, 201)
    assert np.isin(np.round(vals, 10), np.round(grid, 10)).all()


@pytest.fixture
def step_adata() -> AnnData:
    """Three genes switching on and two switching off at t=0.5, plus a flat gene."""
    t = np.random.default_rng(0).uniform(0, 1, 600)
    X = np.column_stack(
        [_sigmoid(t, 0.5, rate=15.0)] * 3
        + [_sigmoid(t, 0.5, rate=15.0, sign=-1.0)] * 2
        + [np.full_like(t, 0.5)]
    ).astype(np.float32)
    obs = pd.DataFrame({"pt": t}, index=[f"c{i}" for i in range(t.size)])
    return AnnData(X, obs=obs)


def test_flux_is_highest_where_expression_changes(step_adata):
    expression_dynamics(step_adata, "pt", t_range=(0, 1))

    flux = step_adata.obs["transcriptional_flux"]
    t = step_adata.obs["pt"]
    plateau = flux[(t < 0.1) | (t > 0.9)].mean()
    transition = flux[(t > 0.4) & (t < 0.6)].mean()
    assert transition > 20 * plateau


def test_activation_and_repression_counts_follow_transitions(step_adata):
    expression_dynamics(step_adata, "pt", t_range=(0, 1))

    t = step_adata.obs["pt"]
    mid = (t > 0.4) & (t < 0.6)
    flat = (t < 0.1) | (t > 0.9)
    # all three up-switching genes are "highly activating" mid-transition, and
    # both down-switching genes are "highly repressing"
    assert (step_adata.obs.loc[mid, "n_activation"] == 3).all()
    assert (step_adata.obs.loc[mid, "n_repression"] == 2).all()
    assert step_adata.obs.loc[flat, "n_activation"].max() == 0


def test_flat_gene_has_no_dynamics(adata):
    expression_dynamics(adata, "pt", t_range=(0, 1))
    # drop the dynamic genes; with only a constant gene, flux is ~0
    sub = adata[:, ["flat"]].copy()
    expression_dynamics(sub, "pt", t_range=(0, 1))

    assert sub.obs["transcriptional_flux"].max() < 1e-3


def test_sparse_input_matches_dense(adata):
    sparse = adata.copy()
    sparse.X = csr_matrix(sparse.X)

    expression_dynamics(adata, "pt", t_range=(0, 1))
    expression_dynamics(sparse, "pt", t_range=(0, 1))

    pd.testing.assert_frame_equal(adata.obs, sparse.obs)
    pd.testing.assert_frame_equal(adata.var, sparse.var)


def test_float64_input_supported(adata):
    adata.X = adata.X.astype(np.float64)

    expression_dynamics(adata, "pt", t_range=(0, 1))

    assert adata.var["peak_activation_t"].notna().all()


def test_layer_is_used_instead_of_X(adata):
    adata.layers["lyr"] = adata.X.copy()
    adata.X = np.zeros_like(adata.X)
    expression_dynamics(adata, "pt", t_range=(0, 1), layer="lyr")

    assert adata.var.loc["up_mid", "peak_activation_t"] == pytest.approx(0.5, abs=0.06)


def test_does_not_modify_expression(adata):
    X_before = adata.X.copy()

    expression_dynamics(adata, "pt", t_range=(0, 1))

    np.testing.assert_array_equal(adata.X, X_before)


class TestGeneMask:
    def test_prefixed_outputs_and_nan_for_unmasked(self, adata):
        adata.var["sel"] = [True, True, False, False, False]

        expression_dynamics(adata, "pt", t_range=(0, 1), gene_mask="sel")

        assert {"sel_up", "sel_dw", "sel_flux"} <= set(adata.obs.columns)
        assert "n_activation" not in adata.obs.columns
        assert "transcriptional_flux" not in adata.obs.columns
        col = adata.var["sel_peak_activation_t"]
        assert col[["up_early", "up_mid"]].notna().all()
        assert col[["down_late", "up_late", "flat"]].isna().all()
        assert col["up_early"] == pytest.approx(0.25, abs=0.06)
        assert col["up_mid"] == pytest.approx(0.50, abs=0.06)
        assert "peak_activation_t" not in adata.var.columns

    def test_masked_run_equals_subsetted_run(self, adata):
        adata.var["sel"] = [True, False, True, False, True]
        sub = adata[:, adata.var["sel"]].copy()

        expression_dynamics(adata, "pt", t_range=(0, 1), gene_mask="sel")
        expression_dynamics(sub, "pt", t_range=(0, 1))

        np.testing.assert_allclose(
            adata.obs["sel_flux"].values, sub.obs["transcriptional_flux"].values
        )
        np.testing.assert_allclose(
            adata.var.loc[sub.var_names, "sel_peak_activation_t"].values,
            sub.var["peak_activation_t"].values,
        )

    def test_missing_mask_column_raises(self, adata):
        with pytest.raises(AssertionError, match="gene_mask must be a column"):
            expression_dynamics(adata, "pt", t_range=(0, 1), gene_mask="nope")

    def test_non_boolean_mask_raises(self, adata):
        adata.var["sel"] = np.arange(adata.n_vars)

        with pytest.raises(AssertionError, match="boolean"):
            expression_dynamics(adata, "pt", t_range=(0, 1), gene_mask="sel")


class TestRangeAndPeriodicity:
    def test_t_range_inferred_from_uns(self, adata):
        adata.uns["pt"] = {"t_range": [0, 1], "periodic": False}
        explicit = adata.copy()

        expression_dynamics(adata, "pt")
        expression_dynamics(explicit, "pt", t_range=(0, 1), periodic=False)

        pd.testing.assert_frame_equal(adata.var, explicit.var)

    def test_t_range_inferred_from_data_without_uns(self, adata):
        expression_dynamics(adata, "pt")

        t = adata.obs["pt"]
        peaks = adata.var["peak_activation_t"]
        assert (peaks >= t.min()).all() and (peaks <= t.max()).all()
        assert adata.var.loc["up_mid", "peak_activation_t"] == pytest.approx(
            0.5, abs=0.06
        )

    def test_periodic_data_peak_near_seam(self):
        # a periodic gene peaking at t=0 (the seam) on [0, 1)
        rng = np.random.default_rng(1)
        t = rng.uniform(0, 1, 500)
        d = np.minimum(t, 1 - t)  # circular distance to 0
        X = np.column_stack([np.exp(-(d**2) / (2 * 0.1**2))]).astype(np.float32)
        adata = AnnData(
            X,
            obs=pd.DataFrame({"pt": t}, index=[f"c{i}" for i in range(500)]),
            var=pd.DataFrame(index=["g"]),
        )
        adata.uns["pt"] = {"t_range": [0, 1], "periodic": True}

        expression_dynamics(adata, "pt")

        row = adata.var.loc["g"]
        # fastest rise happens just before the seam, fastest fall just after it
        assert row["peak_activation_t"] > 0.8
        assert row["peak_repression_t"] < 0.2

    def test_missing_time_key_raises(self, adata):
        with pytest.raises(KeyError):
            expression_dynamics(adata, "does_not_exist")
