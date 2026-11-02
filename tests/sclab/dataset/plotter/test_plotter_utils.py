import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from sclab.dataset.plotter import _utils as pu
from sclab.dataset.plotter._utils import (
    COLOR_DISCRETE_SEQUENCE,
    Rx,
    Rxyz,
    Ry,
    Rz,
    _density_nd,
    _get_color_sequence,
    _make_density_heatmap,
    make_periodic_smoothing_spline,
    rotate_multiple_steps,
    rotate_xyz,
)


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------
class TestRotations:
    @pytest.mark.parametrize("R", [Rx, Ry, Rz])
    def test_orthonormal_and_identity_at_zero(self, R):
        for deg in [0, 30, 90, 217]:
            m = R(deg)
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(m) == pytest.approx(1.0)
        np.testing.assert_allclose(R(0), np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("R", [Rx, Ry, Rz])
    def test_inverse_is_negative_angle(self, R):
        np.testing.assert_allclose(R(40) @ R(-40), np.eye(3), atol=1e-12)

    def test_right_operating_counter_clockwise_convention(self):
        # row vectors: v @ R. 90 degrees CCW about z maps x -> y
        np.testing.assert_allclose(
            np.array([1.0, 0, 0]) @ Rz(90), [0, 1, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            np.array([0, 1.0, 0]) @ Rx(90), [0, 0, 1], atol=1e-12
        )
        np.testing.assert_allclose(
            np.array([0, 0, 1.0]) @ Ry(90), [1, 0, 0], atol=1e-12
        )

    def test_axis_is_fixed(self):
        np.testing.assert_allclose(
            np.array([1.0, 0, 0]) @ Rx(77), [1, 0, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            np.array([0, 1.0, 0]) @ Ry(77), [0, 1, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            np.array([0, 0, 1.0]) @ Rz(77), [0, 0, 1], atol=1e-12
        )

    def test_rxyz_is_composition(self):
        a, b, c = 10, 20, 30
        np.testing.assert_allclose(Rxyz(a, b, c), Rz(c) @ Ry(b) @ Rx(a))
        np.testing.assert_allclose(Rxyz(0, 0, 0), np.eye(3), atol=1e-12)


@pytest.fixture
def X3():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(6, 3)), columns=["a", "b", "c"], index=list("uvwxyz")
    )


class TestRotateXYZ:
    def test_matches_matrix_product_and_keeps_labels(self, X3):
        out = rotate_xyz(X3, 15, 25, 35)
        assert out.columns.tolist() == ["a", "b", "c"]
        assert out.index.tolist() == X3.index.tolist()
        np.testing.assert_allclose(out.values, X3.values @ Rxyz(15, 25, 35))

    def test_preserves_norms_and_original(self, X3):
        before = X3.copy()
        out = rotate_xyz(X3, 15, 25, 35)
        np.testing.assert_allclose(
            np.linalg.norm(out.values, axis=1), np.linalg.norm(X3.values, axis=1)
        )
        pd.testing.assert_frame_equal(X3, before)


class TestRotateMultipleSteps:
    def test_single_and_chained_steps(self, X3):
        out = rotate_multiple_steps(X3, "x:30")
        np.testing.assert_allclose(out.values, X3.values @ Rx(30))
        out = rotate_multiple_steps(X3, "x:30, y=45; z 60".replace(" 60", "60"))
        np.testing.assert_allclose(out.values, X3.values @ Rx(30) @ Ry(45) @ Rz(60))
        assert out.columns.tolist() == ["a", "b", "c"]

    def test_case_and_whitespace_insensitive(self, X3):
        a = rotate_multiple_steps(X3, "X:30 , Y:-10")
        b = rotate_multiple_steps(X3, "x30,y-10")
        np.testing.assert_allclose(a.values, b.values)

    def test_stops_at_first_invalid_step(self, X3):
        out = rotate_multiple_steps(X3, "x:30,q:10,y:45")
        np.testing.assert_allclose(out.values, X3.values @ Rx(30))
        out = rotate_multiple_steps(X3, "x:30,y:abc,z:45")
        np.testing.assert_allclose(out.values, X3.values @ Rx(30))

    def test_short_steps_are_skipped_not_terminating(self, X3):
        out = rotate_multiple_steps(X3, "x:30,,y:45")
        np.testing.assert_allclose(out.values, X3.values @ Rx(30) @ Ry(45))

    def test_empty_string_is_identity(self, X3):
        out = rotate_multiple_steps(X3, "")
        np.testing.assert_allclose(out.values, X3.values)


# --------------------------------------------------------------------------
# periodic smoothing spline
# --------------------------------------------------------------------------
class TestPeriodicSpline:
    def _data(self, n=60, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(0, 2 * np.pi, n))
        y = np.sin(x) + noise * rng.normal(size=n)
        return x, y

    def test_recovers_sine(self):
        x, y = self._data(n=80, noise=0.02)
        spl = make_periodic_smoothing_spline(x, y, (0, 2 * np.pi), lam=1e-4)
        g = np.linspace(0.2, 2 * np.pi - 0.2, 50)
        np.testing.assert_allclose(spl(g), np.sin(g), atol=0.1)

    def test_continuity_at_seam(self):
        x, y = self._data(n=80, noise=0.05)
        spl = make_periodic_smoothing_spline(x, y, (0, 2 * np.pi))
        eps = 1e-6
        assert spl(2 * np.pi - eps) == pytest.approx(spl(0 + eps), abs=1e-3)
        # first derivative also continuous
        d = spl.derivative()
        assert d(2 * np.pi - eps) == pytest.approx(d(eps), abs=1e-2)
        d2 = spl.derivative(2)
        assert d2(2 * np.pi - eps) == pytest.approx(d2(eps), abs=0.5)

    def test_unsorted_input_and_out_of_range_points_are_handled(self):
        x, y = self._data(n=50, noise=0.0)
        order = np.random.default_rng(1).permutation(x.size)
        a = make_periodic_smoothing_spline(x, y, (0, 2 * np.pi), lam=1e-3)
        # shuffled + extra points outside [tmin, tmax) that must be discarded
        xs = np.concatenate([x[order], [-1.0, 7.0, 2 * np.pi]])
        ys = np.concatenate([y[order], [100.0, 100.0, 100.0]])
        b = make_periodic_smoothing_spline(xs, ys, (0, 2 * np.pi), lam=1e-3)
        g = np.linspace(0, 2 * np.pi, 20)
        np.testing.assert_allclose(a(g), b(g), atol=1e-8)

    def test_n_reps_must_be_odd(self):
        x, y = self._data()
        with pytest.raises(AssertionError):
            make_periodic_smoothing_spline(x, y, (0, 2 * np.pi), n_reps=4)

    def test_weights_downweight_outlier(self):
        x, y = self._data(n=60, noise=0.0)
        y = y.copy()
        y[30] += 5.0
        w = np.ones_like(y)
        w[30] = 1e-6
        plain = make_periodic_smoothing_spline(x, y, (0, 2 * np.pi), lam=1e-2)
        weighted = make_periodic_smoothing_spline(x, y, (0, 2 * np.pi), w=w, lam=1e-2)
        err_plain = abs(plain(x[30]) - np.sin(x[30]))
        err_weighted = abs(weighted(x[30]) - np.sin(x[30]))
        assert err_weighted < err_plain


# --------------------------------------------------------------------------
# density estimation
# --------------------------------------------------------------------------
class TestDensityND:
    def test_1d_grid_bounds(self):
        data = np.random.default_rng(0).normal(size=200)
        res = _density_nd(data, bandwidth=0.1, grid_size=(101,))
        xs = res.grid[:, 0]
        assert res.density.shape == (101,)
        span = data.max() - data.min()
        assert xs[0] == pytest.approx(data.min() - span / 10)
        assert xs[-1] == pytest.approx(data.max() + span / 10)

    def test_1d_density_peaks_near_mode(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=3.0, scale=0.3, size=300)
        res = _density_nd(data, bandwidth=0.05, grid_size=(201,))
        assert res.grid[np.argmax(res.density), 0] == pytest.approx(3.0, abs=0.3)

    def test_2d_grid_and_density_shapes(self):
        data = np.random.default_rng(0).normal(size=(50, 2))
        res = _density_nd(data, bandwidth=0.1, grid_size=(9, 9))
        assert res.grid.shape == (81, 2) and res.density.shape == (81,)

    def test_default_grid_size_from_max_grid_size(self):
        data = np.random.default_rng(0).normal(size=(30, 2))
        res = _density_nd(data, bandwidth=0.2, max_grid_size=7)
        assert res.grid_size == (7, 7)

    def test_bounds_must_match_dimension(self):
        with pytest.raises(AssertionError, match="bounds for each dimension"):
            _density_nd(np.zeros((5, 2)), bounds=((0, 1),))


class TestDensityHeatmap:
    def test_returns_contour_trace_with_expected_geometry(self):
        rng = np.random.default_rng(0)
        data = tuple(map(tuple, rng.normal(size=(40, 2)).round(3)))
        trace = _make_density_heatmap(data, 0.2, 3, 0.5, 5, "red")
        assert isinstance(trace, go.Contour)
        z = np.asarray(trace.z)
        assert z.shape == (9, 9)  # 2**3 + 1 grid
        assert len(trace.x) == 9 and len(trace.y) == 9
        assert (z >= 0).all()
        assert trace.contours.start < trace.contours.end
        assert trace.contours.size == pytest.approx(
            (trace.contours.end - trace.contours.start) / 5
        )
        assert trace.line.smoothing == 0.5
        # cached call returns the very same object
        assert _make_density_heatmap(data, 0.2, 3, 0.5, 5, "red") is trace


# --------------------------------------------------------------------------
# colour sequence
# --------------------------------------------------------------------------
class TestColorSequence:
    def test_sequence_is_nonempty_unique_rgb_strings(self):
        seq = COLOR_DISCRETE_SEQUENCE
        assert isinstance(seq, list) and len(seq) >= 10
        assert all(c.startswith("rgb(") for c in seq)
        assert len(set(seq)) == len(seq)

    def test_deterministic(self):
        assert _get_color_sequence() == COLOR_DISCRETE_SEQUENCE

    def test_lightgray_is_excluded_and_colors_are_distinguishable(self):
        def rgb(c):
            return np.array([int(v) for v in c[4:-1].split(",")])

        def redmean(c1, c2):
            r1, g1, b1 = c1
            r2, g2, b2 = c2
            rm = (r1 + r2) / 2
            return np.sqrt(
                (2 + rm / 256) * (r1 - r2) ** 2
                + 4 * (g1 - g2) ** 2
                + (2 + (255 - rm) / 256) * (b1 - b2) ** 2
            )

        cols = [rgb(c) for c in COLOR_DISCRETE_SEQUENCE]
        gray = np.array([211, 211, 211])
        assert all(redmean(c, gray) > 65 for c in cols)
        for i in range(len(cols)):
            for j in range(i):
                assert redmean(cols[i], cols[j]) > 65

    def test_module_attribute_wired(self):
        assert pu.COLOR_DISCRETE_SEQUENCE is COLOR_DISCRETE_SEQUENCE
