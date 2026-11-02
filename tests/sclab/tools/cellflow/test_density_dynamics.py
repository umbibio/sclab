import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from anndata import AnnData  # noqa: E402
from scipy.interpolate import BSpline  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from sclab.tools.cellflow import density_dynamics, real_time  # noqa: E402
from sclab.tools.cellflow._density_dynamics import (  # noqa: E402
    _area_under_curve,
    density,
    get_pseudotimes,
    get_realtimes,
)

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


def _make_adata(times, uns=None, key="pseudotime") -> AnnData:
    obs = pd.DataFrame({key: times}, index=[f"c{i}" for i in range(len(times))])
    adata = AnnData(np.zeros((len(times), 1)), obs=obs)
    if uns is not None:
        adata.uns[key] = dict(uns)
    return adata


@pytest.fixture
def bimodal_times():
    rng = np.random.default_rng(0)
    t = np.concatenate([rng.normal(0.25, 0.04, 1000), rng.normal(0.70, 0.04, 1000)])
    return t[(t > 0) & (t < 1)]


@pytest.fixture
def overlapping_bimodal_times():
    rng = np.random.default_rng(0)
    t = np.concatenate([rng.normal(0.3, 0.1, 1500), rng.normal(0.7, 0.1, 1500)])
    return t[(t > 0) & (t < 1)]


@pytest.fixture
def shallow_and_deep_valleys():
    # modes at 0.15 / 0.5 / 0.72: deep valley near 0.30, shallow one near 0.61
    rng = np.random.default_rng(0)
    t = np.concatenate(
        [
            rng.normal(0.15, 0.04, 1000),
            rng.normal(0.50, 0.05, 1000),
            rng.normal(0.72, 0.05, 1000),
        ]
    )
    return _make_adata(t[(t > 0) & (t < 1)], {"t_range": [0, 1]})


@pytest.fixture
def bimodal(bimodal_times):
    return _make_adata(bimodal_times, {"t_range": [0, 1], "periodic": False})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestDensity:
    def test_guesses_range_and_stores_results(self):
        rng = np.random.default_rng(1)
        adata = _make_adata(rng.uniform(0.1, 0.8, 600))

        density(adata)

        uns = adata.uns["pseudotime"]
        assert uns["periodic"] is False
        tmin, tmax = uns["t_range"]
        assert tmin <= 0.1 and tmax >= 0.8
        assert set(uns["density"]) == {"params", "density_bspline_tck"}
        params = uns["density"]["params"]
        assert params["bandwidth"] == 1 / 64
        assert params["max_grid_size"] == 2**8 + 1
        tck = uns["density"]["density_bspline_tck"]
        assert set(tck) == {"t", "c", "k"} and tck["k"] == 3

    def test_density_integrates_to_one(self, bimodal):
        density(bimodal)

        tck = bimodal.uns["pseudotime"]["density"]["density_bspline_tck"]
        bspl = BSpline(**tck)
        area, _ = quad(bspl, 0, 1, limit=200)
        assert area == pytest.approx(1.0, abs=0.02)

    def test_density_has_modes_at_data_modes(self, bimodal):
        density(bimodal)

        tck = bimodal.uns["pseudotime"]["density"]["density_bspline_tck"]
        bspl = BSpline(**tck)
        x = np.linspace(0, 1, 1001)
        y = bspl(x)
        assert y[np.argmax(y[:500])] > 5 * y[500]  # peak vs. middle valley
        assert x[np.argmax(y[:500])] == pytest.approx(0.25, abs=0.03)
        assert x[500 + np.argmax(y[500:])] == pytest.approx(0.70, abs=0.03)

    def test_uses_stored_periodic_flag(self, bimodal_times):
        adata = _make_adata(bimodal_times, {"t_range": [0, 1], "periodic": True})

        density(adata)

        tck = adata.uns["pseudotime"]["density"]["density_bspline_tck"]
        bspl = BSpline(**tck)
        assert bspl(0.0) == pytest.approx(bspl(1.0), rel=1e-2, abs=1e-3)
        # stored values must not be clobbered
        assert adata.uns["pseudotime"]["periodic"] is True
        assert adata.uns["pseudotime"]["t_range"] == [0, 1]

    def test_preserves_existing_uns_entries(self, bimodal):
        bimodal.uns["pseudotime"]["custom"] = "keep"

        density(bimodal)

        assert bimodal.uns["pseudotime"]["custom"] == "keep"
        assert "density" in bimodal.uns["pseudotime"]

    def test_custom_time_key(self, bimodal_times):
        adata = _make_adata(bimodal_times, key="phase")

        density(adata, time_key="phase")

        assert "density" in adata.uns["phase"]
        assert "pseudotime" not in adata.uns

    def test_ignores_non_finite_pseudotimes_when_guessing(self, bimodal_times):
        times = np.concatenate([bimodal_times, [np.nan, np.nan]])
        adata = _make_adata(times)

        density(adata)

        tmin, tmax = adata.uns["pseudotime"]["t_range"]
        assert np.isfinite([tmin, tmax]).all()

    def test_explicit_t_range_is_respected(self, bimodal_times):
        adata = _make_adata(bimodal_times)

        density(adata, t_range=(0.0, 1.0))

        assert list(adata.uns["pseudotime"]["t_range"]) == [0.0, 1.0]

    def test_explicit_periodic_is_respected(self, bimodal_times):
        adata = _make_adata(bimodal_times)

        density(adata, t_range=(0.0, 1.0), periodic=True)

        assert adata.uns["pseudotime"]["periodic"] is True
        assert adata.uns["pseudotime"]["density"]["params"]["periodic"] is True

    def test_explicit_args_do_not_overwrite_existing_metadata(self, bimodal):
        # bimodal carries t_range=[0, 1], periodic=False from "pseudotime()"
        density(bimodal, t_range=(0.1, 0.9), periodic=True)

        uns = bimodal.uns["pseudotime"]
        assert uns["t_range"] == [0, 1]
        assert uns["periodic"] is False
        assert uns["density"]["params"]["t_range"] == [0.1, 0.9]
        assert uns["density"]["params"]["periodic"] is True

    def test_params_record_stored_values_when_args_omitted(self):
        rng = np.random.default_rng(0)
        adata = _make_adata(
            rng.uniform(0.2, 0.9, 600), {"t_range": [0, 1], "periodic": True}
        )

        density(adata)

        params = adata.uns["pseudotime"]["density"]["params"]
        assert params["t_range"] == [0, 1]
        assert params["periodic"] is True

    def test_params_record_guessed_values(self):
        rng = np.random.default_rng(1)
        adata = _make_adata(rng.uniform(0.1, 0.8, 600))

        density(adata)

        params = adata.uns["pseudotime"]["density"]["params"]
        assert params["t_range"] == list(adata.uns["pseudotime"]["t_range"])
        assert params["periodic"] is False

    def test_explicit_args_override_stored_params(self, bimodal):
        density(bimodal, t_range=(0.0, 2.0))
        density(bimodal)  # stored top-level t_range is still [0, 1]

        assert bimodal.uns["pseudotime"]["density"]["params"]["t_range"] == [0, 1]

    def test_plotting_options_do_not_fail(self, bimodal):
        density(
            bimodal,
            plot_density=True,
            plot_density_fit=True,
            plot_density_fit_derivative=True,
            plot_histogram=True,
        )


class TestDensityDynamics:
    def test_requires_t_range(self, bimodal_times):
        adata = _make_adata(bimodal_times)

        with pytest.raises(ValueError, match="t_range must be provided"):
            density_dynamics(adata)

    def test_finds_both_modes(self, bimodal):
        density_dynamics(bimodal, find_peaks_kwargs={"height": 0.5})

        res = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]
        np.testing.assert_allclose(res["times"], [0.25, 0.70], atol=0.03)
        assert (res["heights"] > 0).all()
        assert res["times"].shape == res["deltas"].shape == res["heights"].shape

    def test_non_periodic_deltas_measure_from_range_start(self, bimodal):
        density_dynamics(bimodal, find_peaks_kwargs={"height": 0.5})

        res = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]
        times, deltas = res["times"], res["deltas"]
        # first delta is distance to tmin; others are gaps between peaks
        assert deltas[0] == pytest.approx(times[0] - 0.0)
        assert deltas[1] == pytest.approx(times[1] - times[0])
        assert deltas.sum() == pytest.approx(times[-1])

    def test_periodic_deltas_sum_to_range_span(self, bimodal):
        density_dynamics(bimodal, periodic=True, find_peaks_kwargs={"height": 0.5})

        res = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]
        np.testing.assert_allclose(res["times"], [0.25, 0.70], atol=0.03)
        assert res["deltas"].sum() == pytest.approx(1.0)
        # wrap-around delta: from peak at ~0.70 to peak at ~0.25 (+1)
        assert res["deltas"][0] == pytest.approx(0.25 - 0.70 + 1.0, abs=0.06)

    def test_periodic_inferred_from_uns(self, bimodal_times):
        adata = _make_adata(bimodal_times, {"t_range": [0, 1], "periodic": True})

        density_dynamics(adata, find_peaks_kwargs={"height": 0.5})

        res = adata.uns["pseudotime"]["density_dynamics_d0_peaks"]
        assert res["deltas"].sum() == pytest.approx(1.0)

    def test_periodic_peak_across_boundary(self):
        # Unimodal peak straddling the 0/1 boundary is one peak when periodic,
        # but only two half-peaks (at the edges) otherwise.
        rng = np.random.default_rng(0)
        t = np.concatenate([rng.normal(0.0, 0.05, 1500), rng.normal(1.0, 0.05, 1500)])
        t = t % 1.0
        adata = _make_adata(t, {"t_range": [0, 1]})

        density_dynamics(adata, periodic=True, find_peaks_kwargs={"height": 0.5})

        res = adata.uns["pseudotime"]["density_dynamics_d0_peaks"]
        assert len(res["times"]) == 1
        assert min(res["times"][0], 1 - res["times"][0]) < 0.03

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: with a single periodic peak the cyclic delta is (t - t) % span "
        "= 0 instead of the full period",
    )
    def test_periodic_single_peak_delta_is_full_period(self):
        rng = np.random.default_rng(0)
        t = np.concatenate([rng.normal(0.0, 0.05, 1500), rng.normal(1.0, 0.05, 1500)])
        adata = _make_adata(t % 1.0, {"t_range": [0, 1]})

        density_dynamics(adata, periodic=True, find_peaks_kwargs={"height": 0.5})

        res = adata.uns["pseudotime"]["density_dynamics_d0_peaks"]
        assert res["deltas"][0] == pytest.approx(1.0)

    def test_height_is_fraction_of_maximum(self, bimodal):
        density_dynamics(bimodal, find_peaks_kwargs={"height": 0.5})

        res = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]
        assert res["heights"].min() >= res["params"]["find_peaks_kwargs"]["height"]
        assert res["params"]["find_peaks_kwargs"]["height"] > 0

    def test_high_height_threshold_removes_smaller_peak(self):
        rng = np.random.default_rng(0)
        t = np.concatenate([rng.normal(0.25, 0.04, 2000), rng.normal(0.70, 0.04, 400)])
        adata = _make_adata(t[(t > 0) & (t < 1)], {"t_range": [0, 1]})

        density_dynamics(adata, find_peaks_kwargs={"height": 0.6})

        times = adata.uns["pseudotime"]["density_dynamics_d0_peaks"]["times"]
        assert len(times) == 1
        assert times[0] == pytest.approx(0.25, abs=0.03)

    def test_extra_find_peaks_kwargs_are_forwarded(self, bimodal):
        density_dynamics(bimodal, find_peaks_kwargs={"height": 0.5, "distance": 20000})

        res = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]
        assert res["params"]["find_peaks_kwargs"]["distance"] == 20000
        assert len(res["times"]) == 2

    def test_peaks_outside_t_range_are_excluded(self, bimodal):
        density_dynamics(bimodal, t_range=(0.0, 0.5), find_peaks_kwargs={"height": 0.5})

        times = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]["times"]
        assert ((times >= 0.0) & (times < 0.5)).all()
        assert times[0] == pytest.approx(0.25, abs=0.03)

    def test_result_key_depends_on_derivative_and_mode(self, bimodal):
        density_dynamics(bimodal, derivative=1, mode="peaks")
        density_dynamics(bimodal, derivative=1, mode="valleys")

        uns = bimodal.uns["pseudotime"]
        assert "density_dynamics_d1_peaks" in uns
        assert "density_dynamics_d1_valleys" in uns

    def test_first_derivative_extrema_flank_density_peaks(self, bimodal):
        # density rises fastest before a mode and falls fastest after it;
        # valley height is relative to max(signal), so falls need a negative one
        density_dynamics(
            bimodal, derivative=1, mode="peaks", find_peaks_kwargs={"height": 0.5}
        )
        density_dynamics(
            bimodal, derivative=1, mode="valleys", find_peaks_kwargs={"height": -0.5}
        )

        uns = bimodal.uns["pseudotime"]
        rise = uns["density_dynamics_d1_peaks"]["times"]
        fall = uns["density_dynamics_d1_valleys"]["times"]
        assert len(rise) >= 2 and len(fall) >= 2
        assert rise[0] < 0.25 < fall[0]
        assert rise[-1] < 0.70 < fall[-1]

    def test_stored_spline_matches_density(self, bimodal):
        density_dynamics(bimodal)

        tck = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"][
            "density_bspline_tck"
        ]
        assert tck["k"] == 3
        assert len(tck["t"]) == len(tck["c"]) + tck["k"] + 1

    def test_stores_params(self, bimodal):
        density_dynamics(bimodal, bandwidth=0.05, kernel="epanechnikov")

        params = bimodal.uns["pseudotime"]["density_dynamics_d0_peaks"]["params"]
        assert params["bandwidth"] == 0.05
        assert params["kernel"] == "epanechnikov"

    def test_creates_uns_entry_when_t_range_given(self, bimodal_times):
        adata = _make_adata(bimodal_times)

        density_dynamics(adata, t_range=(0.0, 1.0))

        assert "density_dynamics_d0_peaks" in adata.uns["pseudotime"]

    def test_valleys_finds_density_minimum(self, overlapping_bimodal_times):
        adata = _make_adata(overlapping_bimodal_times, {"t_range": [0, 1]})

        density_dynamics(adata, mode="valleys", bandwidth=1 / 16)

        times = adata.uns["pseudotime"]["density_dynamics_d0_valleys"]["times"]
        assert len(times) == 1
        assert times[0] == pytest.approx(0.5, abs=0.05)

    def test_valleys_height_keeps_only_deep_valleys(self, shallow_and_deep_valleys):
        def valley_times(**kwargs):
            density_dynamics(
                shallow_and_deep_valleys, mode="valleys", find_peaks_kwargs=kwargs
            )
            uns = shallow_and_deep_valleys.uns["pseudotime"]
            return uns["density_dynamics_d0_valleys"]["times"]

        def has(times, x):
            return np.isclose(times, x, atol=0.05).any()

        times = valley_times()  # no height: every local minimum
        assert has(times, 0.30) and has(times, 0.61)

        times = valley_times(height=0.1)  # 0.61 is shallower than 0.1 * max
        assert has(times, 0.30) and not has(times, 0.61)

    def test_valleys_heights_and_params_in_signal_units(self, shallow_and_deep_valleys):
        density_dynamics(shallow_and_deep_valleys, mode="valleys")
        res = shallow_and_deep_valleys.uns["pseudotime"]["density_dynamics_d0_valleys"]
        assert res["params"]["find_peaks_kwargs"]["height"] is None
        i = np.argmin(np.abs(res["times"] - 0.61))
        assert res["heights"][i] > 0  # unnegated density at the shallow valley

        density_dynamics(
            shallow_and_deep_valleys, mode="valleys", find_peaks_kwargs={"height": 0.1}
        )
        uns = shallow_and_deep_valleys.uns["pseudotime"]
        res = uns["density_dynamics_d0_valleys"]
        density_dynamics(shallow_and_deep_valleys)
        max_density = uns["density_dynamics_d0_peaks"]["heights"].max()
        threshold = res["params"]["find_peaks_kwargs"]["height"]
        assert threshold == pytest.approx(0.1 * max_density)
        assert (res["heights"] <= threshold).all()

    def test_does_not_mutate_find_peaks_kwargs(self, bimodal):
        kwargs = {"height": 0.5}

        density_dynamics(bimodal, find_peaks_kwargs=kwargs)

        assert kwargs == {"height": 0.5}

    def test_plotting_marks_peaks(self, bimodal):
        density_dynamics(
            bimodal,
            plot_density=True,
            plot_density_fit=True,
            plot_histogram=True,
            find_peaks_kwargs={"height": 0.5},
        )


class TestRealTime:
    def test_output_range_and_uns(self, bimodal):
        real_time(bimodal, tmax=100, units="hours")

        rt = bimodal.obs["real_time"].values
        assert np.isfinite(rt).all()
        assert rt.min() >= 0
        assert rt.max() <= 100 * 1.01
        uns = bimodal.uns["real_time"]
        assert uns["tmax"] == 100
        assert uns["t_range"] == [0.0, 100]
        assert uns["t_units"] == "hours"
        assert uns["periodic"] is False
        assert uns["params"]["pseudotime_key"] == "pseudotime"
        assert uns["params"]["units"] == "hours"
        assert set(uns["density_bspline_tck"]) == {"t", "c", "k"}

    def test_custom_key_and_tmax(self, bimodal):
        real_time(bimodal, key_added="rt", tmax=10)

        assert "rt" in bimodal.obs and "rt" in bimodal.uns
        assert "real_time" not in bimodal.obs
        assert bimodal.obs["rt"].max() <= 10 * 1.01

    def test_is_monotone_in_pseudotime(self, bimodal):
        real_time(bimodal)

        df = bimodal.obs.sort_values("pseudotime")
        diffs = np.diff(df["real_time"].values)
        assert (diffs >= -1e-3).all()

    def test_equalises_density(self, bimodal):
        # after the transformation, cells should be spread ~uniformly in real time
        real_time(bimodal, tmax=100)

        rt = bimodal.obs["real_time"].values
        counts, _ = np.histogram(rt, bins=5, range=(0, 100))
        frac = counts / counts.sum()
        np.testing.assert_allclose(frac, 0.2, atol=0.05)

    def test_cells_outside_range_get_nan(self, bimodal_times):
        times = np.concatenate([bimodal_times, [-0.2, 1.3]])
        adata = _make_adata(times, {"t_range": [0, 1], "periodic": False})

        real_time(adata)

        rt = adata.obs["real_time"]
        assert rt.iloc[-2:].isna().all()
        assert rt.iloc[:-2].notna().all()

    def test_does_not_modify_pseudotime(self, bimodal):
        before = bimodal.obs["pseudotime"].copy()

        real_time(bimodal)

        pd.testing.assert_series_equal(bimodal.obs["pseudotime"], before)

    def test_real_time_density_stored_under_pseudotime_key(self, bimodal):
        real_time(bimodal)

        assert "density" in bimodal.uns["pseudotime"]

    def test_explicit_pseudotime_t_range_is_used(self, bimodal_times):
        adata = _make_adata(bimodal_times)

        real_time(adata, pseudotime_t_range=(0.0, 1.0))

        assert list(adata.uns["real_time"]["params"]["pseudotime_t_range"]) == [0, 1]

    def test_explicit_periodic_is_forwarded(self, bimodal_times):
        adata = _make_adata(bimodal_times)

        real_time(adata, pseudotime_t_range=(0.0, 1.0), periodic=True)

        assert adata.uns["real_time"]["periodic"] is True

    def test_explicit_args_override_stored_metadata(self, bimodal):
        real_time(bimodal, pseudotime_t_range=(0.1, 0.9), periodic=True)

        assert list(bimodal.uns["real_time"]["params"]["pseudotime_t_range"]) == [
            0.1,
            0.9,
        ]
        assert bimodal.uns["real_time"]["periodic"] is True
        assert bimodal.uns["pseudotime"]["t_range"] == [0, 1]
        assert bimodal.uns["pseudotime"]["periodic"] is False

    def test_old_results_without_density_params_fall_back_to_top_level(self, bimodal):
        # stored pre-fix results have no t_range/periodic in density params
        density(bimodal)
        params = bimodal.uns["pseudotime"]["density"]["params"]
        del params["t_range"], params["periodic"]

        real_time(bimodal)

        assert list(bimodal.uns["real_time"]["params"]["pseudotime_t_range"]) == [0, 1]
        assert bimodal.uns["real_time"]["periodic"] is False

    def test_plotting_options_do_not_fail(self, bimodal):
        real_time(bimodal, plot_density_fit=True)


class TestGetRealtimesAndPseudotimes:
    def test_get_realtimes_matches_obs(self, bimodal):
        real_time(bimodal, tmax=100)

        pt = bimodal.obs["pseudotime"].values
        rt = get_realtimes(pt, bimodal)

        np.testing.assert_allclose(rt, bimodal.obs["real_time"].values)

    def test_round_trip(self, bimodal):
        real_time(bimodal, tmax=100)

        x = np.array([0.15, 0.25, 0.4, 0.55, 0.7, 0.8])
        back = get_pseudotimes(get_realtimes(x, bimodal), bimodal)

        np.testing.assert_allclose(back, x, atol=5e-3)

    def test_get_pseudotimes_is_increasing(self, bimodal):
        real_time(bimodal, tmax=100)

        pts = get_pseudotimes(np.linspace(5, 95, 10), bimodal)

        assert (np.diff(pts) > 0).all()
        assert ((pts >= 0) & (pts <= 1)).all()

    def test_missing_model_raises(self, bimodal):
        with pytest.raises(KeyError):
            get_realtimes(np.array([0.5]), bimodal)


class TestAreaUnderCurve:
    def test_uniform_density_gives_identity_scaled_by_tmax(self):
        # density = 1 on [0, 1] -> cumulative area equals pseudotime itself
        from scipy.interpolate import make_interp_spline

        x = np.linspace(0, 1, 50)
        bspl = make_interp_spline(x, np.ones_like(x), k=3)
        t, c, k = bspl.tck
        tck = dict(t=t.tolist(), c=c.tolist(), k=k)

        pts = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        out = _area_under_curve(pts, tmax=50.0, tck_dict=tck)

        np.testing.assert_allclose(out, 50.0 * pts, atol=1e-6)

    def test_preserves_input_order(self):
        from scipy.interpolate import make_interp_spline

        x = np.linspace(0, 1, 50)
        bspl = make_interp_spline(x, 2 * x, k=3)  # density 2x integrates to 1
        t, c, k = bspl.tck
        tck = dict(t=t.tolist(), c=c.tolist(), k=k)

        pts = np.array([0.9, 0.1, 0.5, 0.3])
        out = _area_under_curve(pts, tmax=1.0, tck_dict=tck)

        np.testing.assert_allclose(out, pts**2, atol=1e-4)
        assert out.shape == pts.shape
