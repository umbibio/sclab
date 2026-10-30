import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_array, csr_matrix

from sclab.tools.cellflow.utils.periodic_genes import (
    _assert_integer_counts,
    infer_dt_from_index,
    pct_power_below,
    periodic_genes,
    periodic_genes_cosinor,
    power_spectrum_df,
)

PERIOD = 24.0
N_BINS = 12
# antiphase pairs with equal amplitude keep the total library size constant, so
# that per-bin normalisation does not induce artefacts in the null genes
PEAKS = {
    "per_2": 2.0,
    "per_14": 14.0,
    "per_6": 6.0,
    "per_18": 18.0,
    "per_10": 10.0,
    "per_22": 22.0,
}
AMPS = {
    "per_2": 0.6,
    "per_14": 0.6,
    "per_6": 0.8,
    "per_18": 0.8,
    "per_10": 0.5,
    "per_22": 0.5,
}
NOISE = [f"noise{i}" for i in range(12)]


def _circ_dist(a, b, period=PERIOD):
    d = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


def _make_adata(t, extra_cols=None, extra_names=None, seed=4, offset=0.0) -> AnnData:
    rng = np.random.default_rng(seed)
    n = t.size
    cols, names = [], []
    for name, peak in PEAKS.items():
        mu = 5 * (1 + AMPS[name] * np.cos(2 * np.pi * (t - offset - peak) / PERIOD))
        cols.append(rng.poisson(mu))
        names.append(name)
    for name in NOISE:
        cols.append(rng.poisson(5, n))
        names.append(name)
    if extra_cols is not None:
        cols.extend(extra_cols(rng, t))
        names.extend(extra_names)
    X = np.column_stack(cols).astype(float)
    obs = pd.DataFrame({"t": t}, index=[f"c{i}" for i in range(n)])
    return AnnData(csr_matrix(X), obs=obs, var=pd.DataFrame(index=names))


@pytest.fixture(scope="module")
def times():
    return np.random.default_rng(0).uniform(0, PERIOD, 3000)


@pytest.fixture(scope="module")
def _fitted(times):
    adata = _make_adata(times)
    periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, alpha=0.01)
    return adata


@pytest.fixture
def adata(times):
    return _make_adata(times)


@pytest.fixture
def fitted(_fitted):
    return _fitted.copy()


class TestCosinorDetection:
    def test_periodic_genes_detected_and_noise_rejected(self, fitted):
        periodic = fitted.var["periodic"]

        assert periodic[list(PEAKS)].all()
        assert not periodic[NOISE].any()

    def test_pvalues_separate_signal_from_noise(self, fitted):
        v = fitted.var
        assert v.loc[list(PEAKS), "cosinor_padj"].max() < 1e-3
        assert v.loc[NOISE, "cosinor_padj"].min() > 0.05
        assert v.loc[list(PEAKS), "cosinor_r_squared"].min() > 0.9
        assert v.loc[NOISE, "cosinor_r_squared"].max() < 0.7

    def test_peak_time_recovered(self, fitted):
        for name, peak in PEAKS.items():
            got = fitted.var.loc[name, "cosinor_peak_time"]
            assert _circ_dist(got, peak) < 0.5, name

    def test_phase_is_peak_time_in_radians(self, fitted):
        v = fitted.var.loc[list(PEAKS)]

        np.testing.assert_allclose(
            v["cosinor_phase"], 2 * np.pi * v["cosinor_peak_time"] / PERIOD, atol=1e-8
        )
        assert ((v["cosinor_phase"] >= 0) & (v["cosinor_phase"] < 2 * np.pi)).all()

    def test_antiphase_pairs_differ_by_half_a_cycle(self, fitted):
        v = fitted.var
        for a, b in [("per_2", "per_14"), ("per_6", "per_18"), ("per_10", "per_22")]:
            assert _circ_dist(
                v.loc[a, "cosinor_peak_time"], v.loc[b, "cosinor_peak_time"]
            ) == pytest.approx(12.0, abs=0.7)
            assert v.loc[a, "cosinor_amplitude"] == pytest.approx(
                v.loc[b, "cosinor_amplitude"], rel=0.15
            )

    def test_amplitude_orders_with_true_modulation_depth(self, fitted):
        amp = fitted.var["cosinor_amplitude"]

        assert amp["per_6"] > amp["per_2"] > amp["per_10"]
        assert amp[NOISE].max() < 0.05
        assert amp[list(PEAKS)].min() > 0.3

    def test_amplitude_is_hypot_of_first_harmonic(self, fitted):
        np.testing.assert_allclose(
            fitted.var["cosinor_amplitude"],
            fitted.varm["harmonic_amplitudes"]["h1"],
        )

    def test_padj_is_bh_adjusted(self, fitted):
        v = fitted.var
        assert (v["cosinor_padj"] >= v["cosinor_pvalue"] - 1e-12).all()
        assert ((v["cosinor_pvalue"] >= 0) & (v["cosinor_pvalue"] <= 1)).all()
        assert (v["cosinor_padj"] <= 1).all()
        np.testing.assert_array_equal(v["periodic"], v["cosinor_padj"] <= 0.01)

    def test_alpha_controls_calls(self, adata):
        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, alpha=1e-300)
        strict = adata.var["periodic"].sum()
        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, alpha=1.0)

        assert strict <= 6
        assert adata.var["periodic"].all()

    def test_constant_gene_is_never_called(self, times):
        adata = _make_adata(
            times,
            extra_cols=lambda rng, t: [np.zeros(t.size)],
            extra_names=["silent"],
        )

        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS)

        row = adata.var.loc["silent"]
        assert row["cosinor_r_squared"] == 0
        assert row["cosinor_pvalue"] == 1
        assert not row["periodic"]


class TestCosinorOutputs:
    def test_stored_keys_and_shapes(self, fitted):
        assert {
            "cosinor_intercept",
            "cosinor_amplitude",
            "cosinor_phase",
            "cosinor_peak_time",
            "cosinor_r_squared",
            "cosinor_pvalue",
            "cosinor_padj",
            "periodic",
        } <= set(fitted.var.columns)
        assert {"profile", "cosinor_fit", "harmonic_amplitudes"} <= set(fitted.varm)
        assert fitted.varm["profile"].shape == (fitted.n_vars, N_BINS)
        assert fitted.varm["cosinor_fit"].shape == (fitted.n_vars, N_BINS)
        assert fitted.varm["harmonic_amplitudes"].shape == (fitted.n_vars, 1)
        assert list(fitted.varm["profile"].index) == list(fitted.var_names)

    def test_profile_bins_are_centered_labels(self, fitted):
        labels = fitted.varm["profile"].columns.astype(float)

        np.testing.assert_allclose(labels, np.arange(N_BINS) * 2.0)

    def test_fit_tracks_profile_for_periodic_genes(self, fitted):
        prof = fitted.varm["profile"].loc["per_6"].values
        fit = fitted.varm["cosinor_fit"].loc["per_6"].values

        assert np.corrcoef(prof, fit)[0, 1] > 0.95

    def test_r_squared_matches_fit_and_profile(self, fitted):
        prof = fitted.varm["profile"].values
        fit = fitted.varm["cosinor_fit"].values
        rss = ((prof - fit) ** 2).sum(axis=1)
        tss = ((prof - prof.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)

        np.testing.assert_allclose(
            fitted.var["cosinor_r_squared"], 1 - rss / tss, atol=1e-8
        )

    def test_intercept_is_profile_mean(self, fitted):
        # cos/sin regressors are orthogonal over the 12 evenly spaced bins
        np.testing.assert_allclose(
            fitted.var["cosinor_intercept"],
            fitted.varm["profile"].values.mean(axis=1),
            atol=1e-8,
        )

    def test_does_not_modify_counts(self, adata):
        X_before = adata.X.copy()

        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS)

        assert (adata.X != X_before).nnz == 0

    def test_layer_argument(self, adata):
        adata.layers["counts"] = adata.X.copy()
        adata.X = adata.X * 0.5  # non-integer X must be ignored

        periodic_genes_cosinor(
            adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, layer="counts"
        )

        assert adata.var.loc["per_6", "periodic"]

    def test_layer_X_means_adata_X(self, adata):
        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, layer="X")

        assert adata.var.loc["per_6", "periodic"]


class TestCosinorParameters:
    def test_default_range_inferred_from_data(self, adata):
        periodic_genes_cosinor(adata, "t", n=N_BINS)

        assert adata.var.loc[list(PEAKS), "periodic"].all()
        # inferred range is [~0, ~24]; peaks should still be close
        for name, peak in PEAKS.items():
            assert _circ_dist(adata.var.loc[name, "cosinor_peak_time"], peak) < 1.0

    def test_two_harmonics_capture_nonsinusoidal_gene(self, times):
        def extra(rng, t):
            # twice-per-period oscillation: invisible to the fundamental
            return [rng.poisson(5 * (1 + 0.8 * np.cos(4 * np.pi * (t - 3) / PERIOD)))]

        adata = _make_adata(times, extra_cols=extra, extra_names=["harm2"])
        adata1 = adata.copy()

        periodic_genes_cosinor(
            adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, n_harmonics=2, alpha=0.01
        )
        periodic_genes_cosinor(
            adata1, "t", tmin=0, tmax=PERIOD, n=N_BINS, n_harmonics=1, alpha=0.01
        )

        amps = adata.varm["harmonic_amplitudes"]
        assert list(amps.columns) == ["h1", "h2"]
        assert amps.loc["harm2", "h2"] > 5 * amps.loc["harm2", "h1"]
        assert amps.loc["harm2", "h2"] > 0.4
        assert adata.var.loc["harm2", "periodic"]
        assert not adata1.var.loc["harm2", "periodic"]
        assert (
            adata.var.loc["harm2", "cosinor_r_squared"]
            > adata1.var.loc["harm2", "cosinor_r_squared"] + 0.5
        )

    def test_too_few_bins_for_harmonics_raises(self, adata):
        with pytest.raises(ValueError, match="Not enough bins"):
            periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=4, n_harmonics=2)

    def test_minimum_number_of_bins_is_accepted(self, adata):
        # 1 + 2K parameters need at least 2K + 2 bins
        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=4, n_harmonics=1)

        assert np.isfinite(adata.var["cosinor_pvalue"]).all()

    def test_non_count_data_is_rejected(self, adata):
        adata.X = adata.X * 0.5

        with pytest.raises(AssertionError, match="raw integer counts"):
            periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS)

    def test_explicit_period_changes_the_model(self, adata):
        # forcing a 12 h period on 24 h data removes the fit quality
        periodic_genes_cosinor(
            adata, "t", tmin=0, tmax=PERIOD, n=N_BINS, period=PERIOD / 2
        )

        assert adata.var.loc[list(PEAKS), "cosinor_r_squared"].max() < 0.5

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: wrap-around subtracts tmax instead of the period, so with "
        "tmin != 0 cells in the last half bin are dropped (NaN bin)",
    )
    def test_time_shift_invariance(self, times):
        base = _make_adata(times)
        shifted = _make_adata(times + 30.0, offset=30.0)

        periodic_genes_cosinor(base, "t", tmin=0, tmax=PERIOD, n=N_BINS)
        periodic_genes_cosinor(shifted, "t", tmin=30, tmax=30 + PERIOD, n=N_BINS)

        np.testing.assert_allclose(
            shifted.varm["profile"].values, base.varm["profile"].values
        )

    def test_shifted_time_axis_still_recovers_peaks(self, times):
        shifted = _make_adata(times + 30.0, offset=30.0)

        periodic_genes_cosinor(shifted, "t", tmin=30, tmax=30 + PERIOD, n=N_BINS)

        assert shifted.var.loc[list(PEAKS), "periodic"].all()
        for name, peak in PEAKS.items():
            got = shifted.var.loc[name, "cosinor_peak_time"]
            assert _circ_dist(got - 30.0, peak) < 1.0

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: bins with <30 cells are dropped by aggregate_and_filter but the "
        "profile frame is built with all bin labels -> ValueError on gaps in time",
    )
    def test_gap_in_time_coverage(self, times):
        keep = ~((times > 6) & (times < 12))
        adata = _make_adata(times[keep])

        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS)

        assert adata.var.loc["per_6", "periodic"]


class TestCountValidation:
    def test_sparse_integer_counts_pass(self):
        _assert_integer_counts(csr_matrix(np.array([[0, 1], [2, 3]], dtype=float)))

    def test_sparse_fractional_counts_fail(self):
        with pytest.raises(AssertionError, match="raw integer counts"):
            _assert_integer_counts(csr_matrix(np.array([[0, 1.5], [2, 3]])))

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: builtin all() iterates rows of a 2-D dense array -> ValueError "
        "(dense adata.X is unsupported)",
    )
    def test_dense_integer_counts_pass(self):
        _assert_integer_counts(np.array([[0.0, 1.0], [2.0, 3.0]]))

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: isinstance(X, spmatrix) is False for sparse arrays (csr_array)",
    )
    def test_sparse_array_integer_counts_pass(self):
        _assert_integer_counts(csr_array(np.array([[0.0, 1.0], [2.0, 3.0]])))

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: dense adata.X crashes in _assert_integer_counts",
    )
    def test_cosinor_with_dense_X(self, adata):
        adata.X = adata.X.toarray()

        periodic_genes_cosinor(adata, "t", tmin=0, tmax=PERIOD, n=N_BINS)

        assert adata.var.loc["per_6", "periodic"]


class TestSpectralPeriodicGenes:
    def test_flags_periodic_genes(self, adata):
        periodic_genes(adata, "t", 0, PERIOD, PERIOD, N_BINS)

        assert adata.var.loc[list(PEAKS), "periodic"].all()
        assert not adata.var.loc[NOISE, "periodic"].any()
        pct = adata.var["pct_power_below"]
        assert pct[list(PEAKS)].min() > 0.9
        assert pct[NOISE].max() < 0.75

    def test_stores_profiles_and_periodograms(self, adata):
        periodic_genes(adata, "t", 0, PERIOD, PERIOD, N_BINS)

        assert adata.varm["profile"].shape == (adata.n_vars, N_BINS)
        # one-sided periodogram of N_BINS samples has N_BINS//2 + 1 frequencies
        assert adata.varm["periodogram"].shape == (adata.n_vars, N_BINS // 2 + 1)

    def test_threshold_controls_calls(self, adata):
        periodic_genes(adata, "t", 0, PERIOD, PERIOD, N_BINS, min_pct_power_below=0.999)

        assert not adata.var["periodic"].any()

    def test_agrees_with_cosinor(self, adata):
        other = adata.copy()
        periodic_genes(adata, "t", 0, PERIOD, PERIOD, N_BINS)
        periodic_genes_cosinor(other, "t", 0, PERIOD, N_BINS, alpha=0.01)

        np.testing.assert_array_equal(adata.var["periodic"], other.var["periodic"])

    def test_rejects_non_counts(self, adata):
        adata.X = adata.X * 0.5

        with pytest.raises(AssertionError, match="raw integer counts"):
            periodic_genes(adata, "t", 0, PERIOD, PERIOD, N_BINS)


class TestSpectrumHelpers:
    def test_infer_dt_numeric(self):
        assert infer_dt_from_index(pd.Index([0.0, 2.0, 4.0, 6.0])) == 2.0
        assert infer_dt_from_index(pd.Index([0, 1, 2, 10])) == 1.0  # median, not mean

    def test_infer_dt_datetime_is_seconds(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="h")

        assert infer_dt_from_index(idx) == 3600.0

    def test_power_spectrum_peaks_at_signal_frequency(self):
        t = np.arange(12) * 2.0  # dt = 2 -> frequencies are multiples of 1/24
        X = pd.DataFrame({"a": np.sin(2 * np.pi * t / 12.0)}, index=t)

        ps = power_spectrum_df(X)

        assert ps.index.name == "frequency"
        assert ps.index[np.argmax(ps["a"].values)] == pytest.approx(1 / 12)
        # amplitude 1 sinusoid -> power 1/2 ("spectrum" scaling)
        assert ps["a"].max() == pytest.approx(0.5)
        assert ps.index.max() == pytest.approx(0.25)  # Nyquist for dt=2

    def test_power_spectrum_ignores_constant_offset(self):
        t = np.arange(16.0)
        X = pd.DataFrame({"a": 10 + np.cos(2 * np.pi * t / 8)}, index=t)

        ps = power_spectrum_df(X)

        assert ps["a"].iloc[0] == pytest.approx(0, abs=1e-12)

    def test_pct_power_below_pure_tones(self):
        t = np.arange(24.0)
        X = pd.DataFrame(
            {
                "slow": np.sin(2 * np.pi * t / 24),
                "fast": np.sin(2 * np.pi * t / 3),
            },
            index=t,
        )

        pct = pct_power_below(power_spectrum_df(X), max_freq=0.1)

        assert pct["slow"] == pytest.approx(1.0)
        assert pct["fast"] == pytest.approx(0.0, abs=1e-9)
        assert pct.name == "pct_power_at_low_freq"

    def test_pct_power_below_all_frequencies_is_one(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(16, 3)), index=np.arange(16.0))
        ps = power_spectrum_df(X)

        pct = pct_power_below(ps, max_freq=ps.index.max())

        np.testing.assert_allclose(pct, 1.0)

    def test_pct_power_below_single_row_is_nan(self):
        ps = pd.DataFrame({"a": [1.0]}, index=pd.Index([0.0], name="frequency"))

        assert np.isnan(pct_power_below(ps, 0.1)["a"])
