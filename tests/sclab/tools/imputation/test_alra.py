import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from sclab.tools.imputation._alra import _alra_on_ndarray, _fix_alra_scale


def _alra_r_available() -> bool:
    try:
        import rpy2.robjects  # noqa: F401
        from rpy2.robjects.packages import importr

        importr("ALRA")
        return True
    except Exception:
        return False


def _numpy2ri_activate_works() -> bool:
    try:
        import rpy2.robjects.numpy2ri as numpy2ri

        numpy2ri.activate()
        numpy2ri.deactivate()
        return True
    except ImportError:
        return False
    except DeprecationWarning:
        return False


needs_alra = pytest.mark.skipif(
    not _alra_r_available(), reason="requires R with the ALRA package and rpy2"
)
# rpy2 >= 3.6 raises DeprecationWarning from numpy2ri.activate(), which
# _alra_on_ndarray calls unconditionally.
alra_activate_bug = pytest.mark.xfail(
    condition=_alra_r_available() and not _numpy2ri_activate_works(),
    strict=True,
    raises=DeprecationWarning,
    reason="BUG: _alra_on_ndarray calls rpy2 numpy2ri.activate(), which raises "
    "DeprecationWarning in rpy2>=3.6",
)


@pytest.fixture
def counts() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.poisson(1.5, size=(60, 8)).astype(np.float32)


class TestFixAlraScale:
    def test_affine_rescaling_recovers_original_when_thresholded_is_affine_copy(
        self, counts
    ):
        thrs = np.where(counts > 0, 3 * counts + 5, 0).astype(np.float32)

        out = _fix_alra_scale(counts, thrs, thrs)

        np.testing.assert_allclose(out, counts, atol=1e-4)

    def test_scaled_columns_match_input_nonzero_mean_and_std(self, counts):
        rng = np.random.default_rng(1)
        # thresholded matrix: same support as the data plus extra imputed entries
        thrs = np.abs(rng.normal(2, 1, counts.shape)).astype(np.float32) + 0.5
        thrs[rng.random(counts.shape) < 0.3] = 0
        target = thrs * 1.7

        out = _fix_alra_scale(counts, thrs, target)

        for j in range(counts.shape[1]):
            if not (thrs[:, j] > 0).any():
                continue
            # the affine map is fitted on column statistics of the nonzero entries;
            # before clipping/restoring, nonzero-target moments equal the input's
            vin = counts[:, j][counts[:, j] > 0]
            vthr = thrs[:, j][thrs[:, j] > 0]
            a = vin.std(ddof=1) / vthr.std(ddof=1)
            b = vin.mean() - a * vthr.mean()
            expected = np.where(thrs[:, j] > 0, np.maximum(target[:, j] * a + b, 0), 0)
            expected = np.where(
                (counts[:, j] > 0) & (expected == 0), counts[:, j], expected
            )
            np.testing.assert_allclose(out[:, j], expected, rtol=1e-4, atol=1e-5)

    def test_zeros_of_thresholded_stay_zero_unless_observed(self, counts):
        rng = np.random.default_rng(2)
        thrs = np.abs(rng.normal(1, 1, counts.shape)).astype(np.float32)
        thrs[rng.random(counts.shape) < 0.5] = 0

        out = _fix_alra_scale(counts, thrs, thrs + 1)

        zeros = thrs == 0
        np.testing.assert_array_equal(out[zeros & (counts == 0)], 0)
        # observed positive entries are never lost
        np.testing.assert_array_equal(
            out[zeros & (counts > 0)], counts[zeros & (counts > 0)]
        )

    def test_output_is_non_negative(self, counts):
        rng = np.random.default_rng(3)
        thrs = np.abs(rng.normal(1, 1, counts.shape)).astype(np.float32)
        # a target with strongly negative entries gets clipped
        target = rng.normal(0, 5, counts.shape).astype(np.float32)

        out = _fix_alra_scale(counts, thrs, target)

        assert (out >= 0).all()

    def test_observed_positive_values_are_restored_when_clipped(self):
        data = np.array([[4.0], [6.0], [0.0], [5.0]], dtype=np.float32)
        thrs = np.array([[1.0], [2.0], [0.0], [3.0]], dtype=np.float32)
        target = np.array([[-100.0], [2.0], [0.0], [3.0]], dtype=np.float32)

        out = _fix_alra_scale(data, thrs, target)

        assert out[0, 0] == 4.0  # clipped to 0 then restored to observed value
        assert out[2, 0] == 0.0

    def test_degenerate_columns_are_left_unscaled(self):
        data = np.array(
            [
                # col0: all zero, col1: single nonzero, col2: thrs constant, col3: ok
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 7.0, 6.0, 3.0],
                [0.0, 0.0, 8.0, 4.0],
            ],
            dtype=np.float32,
        )
        thrs = np.array(
            [
                [0.0, 0.0, 5.0, 2.0],
                [0.0, 0.0, 5.0, 4.0],
                [1.0, 3.0, 5.0, 6.0],
                [0.0, 0.0, 5.0, 8.0],
            ],
            dtype=np.float32,
        )
        target = thrs * 2

        out = _fix_alra_scale(data, thrs, target)

        # unscaled columns: target passed through (restricted to thrs > 0)
        np.testing.assert_allclose(out[:, 0], [0, 0, 2.0, 0])  # no input stats
        np.testing.assert_allclose(out[:, 1], [0, 0, 6.0, 0])  # target passes through
        np.testing.assert_allclose(out[:, 2], [10.0] * 4)  # thrs std == 0
        # scaled column: matches affine map fitted on nonzero moments
        a = data[:, 3].std(ddof=1) / thrs[:, 3].std(ddof=1)
        b = data[:, 3].mean() - a * thrs[:, 3].mean()
        np.testing.assert_allclose(out[:, 3], target[:, 3] * a + b, rtol=1e-5)

    def test_sparse_input_matches_dense(self, counts):
        rng = np.random.default_rng(4)
        thrs = np.abs(rng.normal(1, 1, counts.shape)).astype(np.float32)
        thrs[rng.random(counts.shape) < 0.4] = 0
        target = thrs * 2

        dense = _fix_alra_scale(counts, thrs, target)

        for fmt in (csr_matrix, csc_matrix):
            np.testing.assert_allclose(
                _fix_alra_scale(fmt(counts), thrs, target), dense
            )

    def test_inputs_are_not_modified(self, counts):
        rng = np.random.default_rng(5)
        thrs = np.abs(rng.normal(1, 1, counts.shape)).astype(np.float32)
        target = thrs * 2
        before = (counts.copy(), thrs.copy(), target.copy())

        _fix_alra_scale(counts, thrs, target)

        for arr, ref in zip((counts, thrs, target), before):
            np.testing.assert_array_equal(arr, ref)

    def test_shape_is_preserved(self, counts):
        thrs = (counts > 0).astype(np.float32) * 2

        out = _fix_alra_scale(counts, thrs, thrs)

        assert out.shape == counts.shape
        assert np.isfinite(out).all()


@needs_alra
@alra_activate_bug
class TestAlraWithR:
    @pytest.fixture
    def low_rank_counts(self):
        rng = np.random.default_rng(0)
        # rank-2 signal with strong dropout
        U = rng.gamma(2, 1, (150, 2))
        V = rng.gamma(2, 1, (2, 40))
        lam = U @ V
        full = rng.poisson(lam).astype(np.float32)
        drop = rng.random(full.shape) < 0.5
        observed = np.where(drop, 0, full).astype(np.float32)
        return observed, lam

    def test_returns_three_matrices_of_input_shape(self, low_rank_counts):
        observed, _ = low_rank_counts

        aprx, thrs, scaled = _alra_on_ndarray(observed)

        for m in (aprx, thrs, scaled):
            assert m.shape == observed.shape
            assert m.dtype == np.float32
        assert (thrs >= 0).all()

    def test_sparse_input_matches_dense(self, low_rank_counts):
        observed, _ = low_rank_counts

        a = _alra_on_ndarray(observed)
        b = _alra_on_ndarray(csr_matrix(observed))

        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-5)

    def test_imputation_recovers_signal(self, low_rank_counts):
        observed, lam = low_rank_counts

        _, thrs, _ = _alra_on_ndarray(observed)
        out = _fix_alra_scale(observed, thrs, thrs)

        dropped = observed == 0
        err_imputed = np.corrcoef(out[dropped], lam[dropped])[0, 1]
        err_raw = (
            np.corrcoef(observed[dropped], lam[dropped])[0, 1]
            if observed[dropped].std() > 0
            else 0
        )
        assert err_imputed > 0.5
        assert err_imputed > err_raw
        assert (out >= 0).all()
