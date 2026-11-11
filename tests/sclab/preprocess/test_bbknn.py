"""sclab's port of bbknn (``neighbors(..., batch_key=...)``, backend native).

Against bbknn's exact search (``computation="cKDTree"``): the same neighbors,
distances within ~1e-13 (brute force vs tree, float64), connectivities within
~6e-6 (float32 ``exp``, as for the scanpy port; umap-learn >= 0.5.12, see
``test_neighbors.py``), measured on these data.
"""

from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from sclab._citations import CITATIONS_KEY, UNS_KEY
from sclab.preprocess import neighbors
from sclab.preprocess._bbknn import _trim, bbknn

DIST_ATOL = 1e-12
CONN_ATOL = 1e-5
UMAP_VERSION = tuple(int(v) for v in version("umap-learn").split(".")[:3])


def make_batches(n_obs=600, n_dims=12, sizes=(0.5, 0.3, 0.2), seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, (5, n_dims))
    batch = rng.choice(len(sizes), n_obs, p=sizes)
    x = centers[rng.integers(0, 5, n_obs)] + rng.normal(size=(n_obs, n_dims))
    x += batch[:, None] * 0.7  # a batch shift
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsm["X_pca"] = x.astype(np.float32)
    adata.obs["batch"] = pd.Categorical([f"b{b}" for b in batch])
    return adata


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(neighbors_within_batch=3, n_pcs=50),
        dict(neighbors_within_batch=10, n_pcs=8),
        dict(neighbors_within_batch=5, n_pcs=12, trim=8),  # trims
        dict(neighbors_within_batch=5, n_pcs=12, trim=0),
    ],
)
def test_native_matches_bbknn_exact(kwargs):
    bb = pytest.importorskip("bbknn")
    a, b = make_batches(), make_batches()
    bb.bbknn(a, batch_key="batch", computation="cKDTree", **kwargs)
    bbknn(b, batch_key="batch", **kwargs)

    pa, pb = a.uns["neighbors"], b.uns["neighbors"]
    assert pa["params"]["bbknn"].pop("computation") == "cKDTree"
    assert pb["params"]["bbknn"].pop("computation") == "brute"
    assert pa == pb

    da, db = a.obsp["distances"], b.obsp["distances"]
    assert type(da) is type(db) and da.dtype == db.dtype == np.float64
    np.testing.assert_array_equal(da.indptr, db.indptr)
    np.testing.assert_array_equal(da.indices, db.indices)  # the same neighbors
    np.testing.assert_allclose(da.data, db.data, rtol=0, atol=DIST_ATOL)

    ca, cb = a.obsp["connectivities"], b.obsp["connectivities"]
    assert type(ca) is type(cb) and ca.dtype == cb.dtype == np.float32
    if UMAP_VERSION >= (0, 5, 12):  # earlier: infinite sigmas (fastmath)
        np.testing.assert_array_equal(ca.indptr, cb.indptr)
        np.testing.assert_array_equal(ca.indices, cb.indices)
        np.testing.assert_allclose(ca.data, cb.data, rtol=0, atol=CONN_ATOL)


@pytest.mark.parametrize("trim", [1, 4, 30])
def test_trim_matches_bbknn(trim):
    matrix = pytest.importorskip("bbknn.matrix")
    rng = np.random.default_rng(1)
    m = sparse.random(200, 200, density=0.1, random_state=rng, dtype=np.float32)
    m = (m + m.T).tocsr()
    m.data[::7] = m.data[0]  # ties at the thresholds
    expected = matrix.trimming(m.copy(), trim)
    got = _trim(m, trim)
    assert (got != expected).nnz == 0 and got.nnz < m.nnz


def test_neighbors_native_batch_key():
    adata = make_batches()
    neighbors(adata, batch_key="batch", n_neighbors=4, dims=(2, 10), backend="native")
    params = adata.uns["neighbors"]["params"]
    assert params["n_neighbors"] == 12 and params["n_pcs"] == 9
    assert params["use_rep"] == "X_pca_trimmed"
    assert params["bbknn"] == dict(trim=120, computation="brute", batch_key="batch")
    dist = adata.obsp["distances"]
    assert sparse.issparse(dist) and (dist.getnnz(axis=1) == 11).all()  # but self
    batch = adata.obs["batch"].cat.codes.to_numpy()
    rows = np.repeat(np.arange(adata.n_obs), 11)
    per_batch = pd.crosstab(rows, batch[dist.indices])
    own = per_batch.to_numpy()[np.arange(adata.n_obs), batch]
    assert (own == 3).all() and (per_batch.sum(axis=1) == 11).all()
    conn = adata.obsp["connectivities"]
    assert sparse.issparse(conn) and (conn != conn.T).nnz == 0
    cites = set(adata.uns[UNS_KEY][CITATIONS_KEY])
    assert {"bbknn", "umap", "umap-learn", "scikit-learn"} <= cites
    assert "scanpy" not in cites


def test_auto_prefers_bbknn():
    pytest.importorskip("bbknn")
    pytest.importorskip("scanpy")
    a, b = make_batches(), make_batches()
    neighbors(a, batch_key="batch")
    neighbors(b, batch_key="batch", backend="scanpy")
    assert a.uns["neighbors"]["params"]["bbknn"]["computation"] == "pynndescent"
    for key in ("distances", "connectivities"):
        assert (a.obsp[key] != b.obsp[key]).nnz == 0


def test_small_batch_raises():
    adata = make_batches(n_obs=60, sizes=(0.95, 0.05))
    with pytest.raises(ValueError, match="neighbors_within_batch"):
        bbknn(adata, batch_key="batch", neighbors_within_batch=10)
