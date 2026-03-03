import numpy as np
import pytest

from sclab.tools.cellflow.utils.interpolate import NDBSpline


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    n, d = 200, 5
    t = np.sort(rng.uniform(0, 1, n))
    X = rng.standard_normal((n, d))
    return t, X


@pytest.fixture
def fitted_spline(sample_data):
    t, X = sample_data
    return NDBSpline(t_range=(0, 1)).fit(t, X)


class TestFit:
    def test_returns_self(self, sample_data):
        t, X = sample_data
        spline = NDBSpline(t_range=(0, 1))
        result = spline.fit(t, X)
        assert result is spline

    def test_sets_attributes(self, fitted_spline):
        F = fitted_spline
        assert F.t is not None
        assert F.C is not None
        assert F.k == 3
        assert F.t_range == (0, 1)
        assert F.grid_size is not None and F.grid_size > 0


class TestCall:
    def test_output_shape_1d(self, rng):
        n, m = 200, 50
        t = np.sort(rng.uniform(0, 1, n))
        X = rng.standard_normal((n, 1))
        F = NDBSpline(t_range=(0, 1)).fit(t, X)
        out = F(np.linspace(0, 1, m))
        assert out.shape == (m, 1)

    def test_output_shape_nd(self, fitted_spline):
        out = fitted_spline(np.linspace(0, 1, 50))
        assert out.shape == (50, 5)

    def test_output_shape_2d_query(self, fitted_spline):
        query = np.linspace(0, 1, 12).reshape(3, 4)
        out = fitted_spline(query)
        assert out.shape == (3, 4, 5)

    def test_values_within_data_range(self, sample_data, fitted_spline):
        _, X = sample_data
        out = fitted_spline(np.linspace(0, 1, 100))
        data_range = X.max() - X.min()
        assert out.min() >= X.min() - data_range
        assert out.max() <= X.max() + data_range


class TestDerivative:
    def test_output_shape(self, fitted_spline):
        D = fitted_spline.derivative()
        out = D(np.linspace(0, 1, 50))
        assert out.shape == (50, 5)


class TestGetitem:
    def test_selects_dimensions(self, fitted_spline):
        F_sub = fitted_spline[0:2]
        out = F_sub(np.linspace(0, 1, 50))
        assert out.shape == (50, 2)

    def test_single_dimension(self, fitted_spline):
        F_sub = fitted_spline[3]
        out = F_sub(np.linspace(0, 1, 50))
        assert out.shape == (50, 1)


class TestPeriodic:
    @pytest.fixture
    def periodic_data(self, rng):
        n, d = 200, 3
        t = np.sort(rng.uniform(0, 1, n))
        X = rng.standard_normal((n, d))
        return t, X

    @pytest.fixture
    def fitted_periodic(self, periodic_data):
        t, X = periodic_data
        return NDBSpline(t_range=(0, 1), periodic=True).fit(t, X)

    def test_fit_shape(self, fitted_periodic):
        out = fitted_periodic(np.linspace(0, 1, 50))
        assert out.shape == (50, 3)

    def test_wrapping(self, fitted_periodic):
        query = np.array([0.1, 0.5, 0.9])
        tmax = 1.0
        out_base = fitted_periodic(query)
        out_wrapped = fitted_periodic(query + tmax)
        np.testing.assert_allclose(out_base, out_wrapped, atol=0.000001)
