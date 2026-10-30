import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.integrate import trapezoid  # noqa: E402
from scipy.interpolate import BSpline  # noqa: E402

from sclab.tools.cellflow.utils.density_nd import (  # noqa: E402
    DensityResult,
    density_nd,
    density_result_1d,
    fit_density_1d,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestDensityNd:
    def test_returns_density_result(self, rng):
        data = rng.normal(0.5, 0.05, size=(300, 1))

        rslt = density_nd(data, bounds=((0.0, 1.0),))

        assert isinstance(rslt, DensityResult)
        assert rslt.periodic is False
        assert rslt.bounds == ((0.0, 1.0),)
        assert rslt.grid.shape == (rslt.grid_size[0], 1)
        assert rslt.density.shape == (rslt.grid_size[0],)
        assert (rslt.density >= 0).all()
        assert rslt.scale == 1

    def test_1d_input_is_reshaped(self, rng):
        rslt = density_nd(rng.normal(0.5, 0.05, 300), bounds=((0.0, 1.0),))

        assert rslt.grid.shape[1] == 1

    def test_default_bounds_pad_data_range_by_ten_percent(self, rng):
        data = rng.uniform(2.0, 4.0, size=(200, 1))

        rslt = density_nd(data)

        lo, hi = rslt.bounds[0]
        assert lo == pytest.approx(data.min() - 0.1 * np.ptp(data))
        assert hi == pytest.approx(data.max() + 0.1 * np.ptp(data))

    def test_default_grid_size_and_bounds_in_2d(self, rng):
        data = np.column_stack([rng.uniform(0, 1, 300), rng.uniform(0, 0.5, 300)])

        rslt = density_nd(data, max_grid_size=21, bounds=((0, 1), (0, 0.5)))

        # grid_size is proportional to the span along each dimension
        assert rslt.grid_size == (21, 10)
        assert rslt.grid.shape == (21 * 10, 2)
        assert rslt.density.shape == (210,)

    def test_explicit_grid_size(self, rng):
        data = rng.uniform(0, 1, size=(100, 2))

        rslt = density_nd(data, grid_size=(7, 9), bounds=((0, 1), (0, 1)))

        assert rslt.grid.shape == (63, 2)
        assert rslt.grid[:, 0].min() == 0 and rslt.grid[:, 0].max() == 1

    def test_normalize_1d_integrates_to_one(self, rng):
        data = rng.normal(0.5, 0.08, size=(500, 1))

        rslt = density_nd(data, bounds=((0.0, 1.0),), normalize=True)

        area = trapezoid(rslt.density, rslt.grid[:, 0])
        assert area == pytest.approx(1.0)
        assert rslt.scale != 1

    def test_unnormalized_density_matches_sklearn_kde(self, rng):
        data = rng.normal(0.5, 0.08, size=(200, 1))

        rslt = density_nd(data, bounds=((0.0, 1.0),), bandwidth=0.05)

        expected = np.exp(rslt.kde.score_samples(rslt.grid))
        np.testing.assert_allclose(rslt.density, expected)
        assert rslt.kde.bandwidth == 0.05

    def test_normalize_2d_riemann_sum_is_one(self, rng):
        data = rng.normal(0.5, 0.1, size=(500, 2))

        rslt = density_nd(
            data,
            bounds=((0.0, 1.0), (0.0, 1.0)),
            grid_size=(33, 33),
            normalize=True,
            bandwidth=0.05,
        )

        d = rslt.density.reshape(33, 33)[:-1, :-1]
        assert d.sum() * (1 / 32) ** 2 == pytest.approx(1.0)

    def test_mode_located_at_data_center(self, rng):
        data = rng.normal(0.3, 0.03, size=(500, 1))

        rslt = density_nd(data, bounds=((0.0, 1.0),))

        assert rslt.grid[np.argmax(rslt.density), 0] == pytest.approx(0.3, abs=0.02)

    def test_periodic_wraps_mass_across_boundary(self, rng):
        # data concentrated near 0/1: periodic KDE sees one bump, plain KDE two halves
        data = (rng.normal(0.0, 0.03, size=(500, 1))) % 1.0
        kwargs = dict(
            bounds=((0.0, 1.0),), bandwidth=0.03, grid_size=(101,), normalize=True
        )

        plain = density_nd(data, **kwargs)
        periodic = density_nd(data, periodic=True, **kwargs)

        # periodic density is continuous across the boundary...
        assert periodic.density[0] == pytest.approx(periodic.density[-1], rel=1e-6)
        # ...and has about twice the boundary density of the plain one
        assert periodic.density[0] > 1.5 * plain.density[0]
        assert periodic.periodic is True

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: periodic=True fits the KDE on 3 stacked copies of the data, so "
        "without normalize=True the returned density integrates to 1/3",
    )
    def test_periodic_unnormalized_density_integrates_to_one(self, rng):
        data = rng.uniform(0, 1, size=(500, 1))

        rslt = density_nd(
            data,
            bounds=((0.0, 1.0),),
            bandwidth=0.03,
            grid_size=(201,),
            periodic=True,
        )

        assert trapezoid(rslt.density, rslt.grid[:, 0]) == pytest.approx(1.0, rel=0.05)

    def test_periodic_density_is_translation_equivariant(self, rng):
        data = rng.uniform(0, 1, size=(200, 1)) ** 2
        shift = 0.25
        kwargs = dict(
            bounds=((0.0, 1.0),), bandwidth=0.03, grid_size=(101,), periodic=True
        )

        a = density_nd(data, **kwargs)
        b = density_nd((data + shift) % 1.0, **kwargs)

        # grid spacing is 0.01 so a shift of 0.25 is exactly 25 grid points
        np.testing.assert_allclose(
            np.roll(a.density[:-1], 25), b.density[:-1], rtol=1e-6
        )

    def test_periodic_requires_bounds(self, rng):
        with pytest.raises(AssertionError, match="bounds must be specified"):
            density_nd(rng.uniform(size=(50, 1)), periodic=True)

    def test_bounds_must_match_dimensionality(self, rng):
        with pytest.raises(AssertionError, match="bounds for each dimension"):
            density_nd(rng.uniform(size=(50, 2)), bounds=((0, 1),))

    def test_periodic_2d_runs(self, rng):
        data = rng.uniform(size=(80, 2))

        rslt = density_nd(
            data, bounds=((0, 1), (0, 1)), periodic=True, grid_size=(9, 9)
        )

        assert rslt.density.shape == (81,)
        assert np.isfinite(rslt.density).all()


class TestFitDensity1d:
    def test_returns_normalized_density_and_spline(self, rng):
        times = rng.normal(0.5, 0.1, 800).clip(0, 1)

        rslt, bspl = fit_density_1d(times, (0.0, 1.0), periodic=False)

        assert isinstance(bspl, BSpline)
        assert bspl.k == 3
        assert trapezoid(rslt.density, rslt.grid[:, 0]) == pytest.approx(1.0)
        x = rslt.grid[:, 0]
        np.testing.assert_allclose(
            bspl(x), rslt.density, atol=0.05 * rslt.density.max()
        )

    def test_points_outside_range_are_discarded(self, rng):
        inside = rng.normal(0.5, 0.05, 500)
        outside = np.array([-5.0, 7.0, 100.0])

        r1, _ = fit_density_1d(inside, (0.0, 1.0), periodic=False)
        r2, _ = fit_density_1d(
            np.concatenate([inside, outside]), (0.0, 1.0), periodic=False
        )

        np.testing.assert_allclose(r1.density, r2.density)

    def test_periodic_spline_is_continuous_at_boundary(self, rng):
        times = rng.normal(0.0, 0.05, 800) % 1.0

        _, bspl = fit_density_1d(times, (0.0, 1.0), periodic=True)

        assert bspl(0.0) == pytest.approx(bspl(1.0), rel=1e-2)
        assert bspl(0.0) > 1.0  # much of the mass sits on the seam

    def test_default_bandwidth_is_range_over_64(self, rng):
        times = rng.uniform(0, 4, 500)

        rslt, _ = fit_density_1d(times, (0.0, 4.0), periodic=False)

        assert rslt.kde.bandwidth == pytest.approx(4.0 / 64)

    def test_grid_covers_t_range(self, rng):
        times = rng.uniform(2, 3, 300)

        rslt, _ = fit_density_1d(times, (2.0, 3.0), periodic=False, max_grid_size=65)

        assert rslt.grid[0, 0] == 2.0 and rslt.grid[-1, 0] == 3.0
        assert rslt.grid_size == (65,)


class TestDensityResult1d:
    @pytest.fixture
    def rslt_and_data(self, rng):
        data = rng.normal(0.5, 0.1, 500).clip(0, 1)
        rslt, _ = fit_density_1d(data, (0.0, 1.0), periodic=False)
        return rslt, data

    def test_requires_at_least_one_plot_option(self, rslt_and_data):
        rslt, _ = rslt_and_data

        with pytest.raises(ValueError, match="At least one"):
            density_result_1d(
                rslt,
                plot_density=False,
                plot_density_fit=False,
                plot_density_fit_derivative=False,
                plot_histogram=False,
            )

    def test_returns_axes_with_lines_when_not_shown(self, rslt_and_data):
        rslt, data = rslt_and_data

        ax = density_result_1d(
            rslt, data=data, plot_density=True, plot_density_fit=True, show=False
        )

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == 2
        assert ax.get_ylabel() == "Density"

    def test_histogram_draws_bars(self, rslt_and_data):
        rslt, data = rslt_and_data

        ax = density_result_1d(
            rslt,
            data=data,
            plot_density_fit=False,
            plot_histogram=True,
            histogram_nbins=20,
            show=False,
        )

        assert len(ax.patches) == 20

    def test_histogram_requires_data(self, rslt_and_data):
        rslt, _ = rslt_and_data

        with pytest.raises(AssertionError, match="data must be provided"):
            density_result_1d(
                rslt, plot_density_fit=False, plot_histogram=True, show=False
            )

    def test_derivative_plot_is_centered_on_zero(self, rslt_and_data):
        rslt, data = rslt_and_data

        ax = density_result_1d(
            rslt,
            data=data,
            plot_density_fit=False,
            plot_density_fit_derivative=True,
            show=False,
        )

        lo, hi = ax.get_ylim()
        assert lo == pytest.approx(-hi)

    def test_draws_on_given_axes(self, rslt_and_data):
        rslt, data = rslt_and_data
        _, ax = plt.subplots()

        out = density_result_1d(rslt, data=data, ax=ax, show=False)

        assert out is ax
        assert len(ax.lines) == 1
