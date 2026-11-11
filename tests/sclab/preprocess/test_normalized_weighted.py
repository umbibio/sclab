import pytest
from anndata import AnnData

import sclab.preprocess

from ..test_utils import simple_loop_adata


@pytest.fixture
def adata():
    return simple_loop_adata()


def test_sparse(adata: AnnData):
    sclab.preprocess.normalize_weighted(adata)


def test_dense(adata: AnnData):
    adata.X = adata.X.toarray()
    sclab.preprocess.normalize_weighted(adata)


def test_batch(adata: AnnData):
    sclab.preprocess.normalize_weighted(adata, batch_key="quadrant")


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_apply_by_rows_equals_in_place_assignment(fmt):
    import warnings

    import numpy as np
    import scipy.sparse as sp

    from sclab.preprocess._normalize_weighted import apply_by_rows

    X = sp.random(60, 40, density=0.3, format="csr", random_state=0)
    X = X.toarray() if fmt == "dense" else X.asformat(fmt)
    rows = np.arange(60)
    masks = [rows % 3 == 0, rows % 3 == 1]  # rows % 3 == 2: kept as they are

    def fn(x, factors=iter([2.0, 3.0])):  # a different factor per group
        return x * next(factors)

    expected = X.copy()
    with warnings.catch_warnings():  # SparseEfficiencyWarning
        warnings.simplefilter("ignore")
        for m, f in zip(masks, (2.0, 3.0)):
            expected[m, :] = expected[m] * f
    got = apply_by_rows(X.copy(), masks, fn)
    assert type(got) is type(X) and got.dtype == X.dtype
    dense = (lambda a: a) if fmt == "dense" else (lambda a: a.toarray())
    np.testing.assert_array_equal(dense(got), dense(expected))


@pytest.mark.parametrize("fmt", ["sparse", "dense"])
@pytest.mark.parametrize("batch_key", [None, "quadrant"])
@pytest.mark.parametrize("int_dtype", ["int64", "int32"])
def test_integer_counts_are_not_truncated(fmt, batch_key, int_dtype):
    import numpy as np

    float_dtype = np.float64 if int_dtype == "int64" else np.float32
    ref = simple_loop_adata()
    X = ref.X.astype(int_dtype)
    X = X.toarray() if fmt == "dense" else X
    as_int = ref.copy()
    as_int.X = X.copy()
    as_float = ref.copy()
    as_float.X = X.astype(float_dtype)

    sclab.preprocess.normalize_weighted(as_int, batch_key=batch_key)
    sclab.preprocess.normalize_weighted(as_float, batch_key=batch_key)

    assert as_int.X.dtype == float_dtype
    got = as_int.X.toarray() if fmt == "sparse" else as_int.X
    want = as_float.X.toarray() if fmt == "sparse" else as_float.X
    assert (got % 1 != 0).any()
    np.testing.assert_array_equal(got, want)
