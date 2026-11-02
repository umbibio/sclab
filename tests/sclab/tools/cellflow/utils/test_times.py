import numpy as np
import pytest

from sclab.tools.cellflow.utils.times import guess_trange


class TestGuessTrange:
    def test_covers_the_data(self):
        rng = np.random.default_rng(0)
        times = rng.uniform(0.13, 0.87, 1000)

        tmin, tmax = guess_trange(times)

        assert tmin <= times.min()
        assert tmax >= times.max()

    def test_rounds_to_two_significant_steps(self):
        times = np.array([0.0132, 0.5, 0.8714])

        tmin, tmax = guess_trange(times)

        assert tmin == pytest.approx(0.01)
        assert tmax == pytest.approx(0.88)

    def test_exact_unit_range(self):
        times = np.linspace(0, 1, 101)

        assert guess_trange(times) == (0.0, 1.0)

    @pytest.mark.parametrize("scale", [1e-3, 1e-1, 1.0, 10.0, 1e3])
    def test_is_scale_equivariant(self, scale):
        rng = np.random.default_rng(1)
        base = rng.uniform(0.2, 0.9, 500)

        b_min, b_max = guess_trange(base)
        s_min, s_max = guess_trange(base * scale)

        assert s_min == pytest.approx(b_min * scale)
        assert s_max == pytest.approx(b_max * scale)

    def test_negative_times(self):
        rng = np.random.default_rng(2)
        times = rng.uniform(-2, -1, 500)

        tmin, tmax = guess_trange(times)

        assert tmin <= times.min() and tmax >= times.max()
        assert tmin >= -2.1 and tmax <= -0.9

    def test_zero_is_not_negative_zero(self):
        times = np.array([0.0, 0.3, 0.7])

        tmin, _ = guess_trange(times)

        assert tmin == 0.0
        assert np.signbit(tmin) == False  # noqa: E712

    def test_verbose_prints_summary(self, capsys):
        guess_trange(np.array([0.1, 0.9]), verbose=True)

        assert "tspan" in capsys.readouterr().out

    def test_range_is_tight(self):
        # estimated range should not exceed true range by more than ~1% of span
        rng = np.random.default_rng(3)
        for lo, hi in [(0, 1), (-0.5, 0.5), (3, 7), (100, 250)]:
            times = rng.uniform(lo, hi, 5000)
            tmin, tmax = guess_trange(times)
            span = hi - lo
            assert abs(tmin - lo) <= 0.02 * span
            assert abs(tmax - hi) <= 0.02 * span

    @pytest.mark.parametrize("s1", [0.001, 0.01, 0.1, 1, 10, 100, 1000])
    @pytest.mark.parametrize("s2", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("lw, hg", [(-2, -1), (-0.5, 0.5), (1, 2)])
    def test_range_recovered_across_scales(self, s1, s2, lw, hg):
        # moved from the in-package test_guess_trange sweep (fewer repeats, seeded)
        rng = np.random.default_rng(0)
        lo, hi = lw * s1 * s2, hg * s1 * s2
        span = hi - lo
        hits = []
        for _ in range(20):
            g_min, g_max = guess_trange(rng.uniform(lo, hi, 1000))
            hits.append(
                abs(g_min - lo) <= 0.01 * span and abs(g_max - hi) <= 0.01 * span
            )
        assert np.mean(hits) > 0.95
