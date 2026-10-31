import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_array, csr_matrix

from sclab.utils import check_raw_counts, looks_like_raw_counts

COUNTS = np.array([[0, 1, 2], [3, 0, 5]], dtype=float)


@pytest.mark.parametrize("fmt", [np.asarray, csr_matrix, csc_matrix, csr_array])
def test_raw_counts_accepted(fmt):
    X = fmt(COUNTS)
    assert looks_like_raw_counts(X)
    check_raw_counts(X)


@pytest.mark.parametrize("fmt", [np.asarray, csr_matrix, csr_array])
@pytest.mark.parametrize(
    "bad", [np.log1p(COUNTS), COUNTS - 1, np.where(COUNTS > 0, np.nan, 0)]
)
def test_non_counts_rejected(fmt, bad):
    X = fmt(bad)
    assert not looks_like_raw_counts(X)
    with pytest.raises(
        ValueError, match="layer 'X' does not look like raw counts.*use counts"
    ):
        check_raw_counts(X, name="layer 'X'", hint="use counts")


def test_empty_is_counts():
    assert looks_like_raw_counts(np.empty((0, 3)))
    assert looks_like_raw_counts(csr_matrix((2, 3)))
