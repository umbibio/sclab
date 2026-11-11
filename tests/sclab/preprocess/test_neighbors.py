"""sclab.preprocess.neighbors: the scanpy backend and sclab's native port.

Native vs scanpy, where scanpy's kNN search is exact (fewer than 4096 cells):
the distances are identical; the connectivities differ by at most ~4e-6
(float32): umap-learn's binary search for sigma stops within 1e-5 of its
target, and the 1-ulp differences between numpy's and numba's float32 ``exp``
move that stop in ~2% of the cells. The port follows umap-learn 0.5.12:
earlier versions compiled ``smooth_knn_dist`` with ``fastmath=True``, which
drops its ``hi == inf`` test and gives some cells an infinite sigma (all
their memberships 1), so the connectivities are compared only from 0.5.12.
"""

from importlib.metadata import version


import numpy as np
import pytest
from anndata import AnnData

from sclab._citations import CITATIONS_KEY, UNS_KEY
from sclab.preprocess import _neighbors, neighbors
from sclab.preprocess._utils import get_neighbors_adjacency_matrix

sc = pytest.importorskip("scanpy")

CONN_ATOL = 1e-5
UMAP_VERSION = tuple(int(v) for v in version("umap-learn").split(".")[:3])


def make_rep(n_obs=300, n_dims=12, seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, (5, n_dims))
    x = centers[rng.integers(0, 5, n_obs)] + rng.normal(size=(n_obs, n_dims))
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsm["X_pca"] = x.astype(np.float32)
    return adata


def assert_same_graph(a: AnnData, b: AnnData):
    assert a.uns["neighbors"] == b.uns["neighbors"]
    da, db = a.obsp["distances"], b.obsp["distances"]
    assert type(da) is type(db) and da.dtype == db.dtype
    assert (da != db).nnz == 0  # identical
    ca, cb = a.obsp["connectivities"], b.obsp["connectivities"]
    assert type(ca) is type(cb) and ca.dtype == cb.dtype == np.float32
    np.testing.assert_array_equal(ca.indptr, cb.indptr)
    np.testing.assert_array_equal(ca.indices, cb.indices)
    if UMAP_VERSION >= (0, 5, 12):
        np.testing.assert_allclose(ca.data, cb.data, rtol=0, atol=CONN_ATOL)


@pytest.mark.skipif(
    _neighbors._umap_rejects_float32(),
    reason="umap-learn >= 0.6: scanpy runs on a float64 copy (see the float64 test)",
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(metric="cosine"),
        dict(dims=(3, 12), n_neighbors=10),
        dict(n_neighbors=400),  # more than the cells: scanpy shrinks it
    ],
)
def test_native_matches_scanpy(kwargs):
    a, b = make_rep(), make_rep()
    neighbors(a, **kwargs, backend="scanpy")
    neighbors(b, **kwargs, backend="native")
    assert_same_graph(a, b)
    assert set(a.obsm) == set(b.obsm)
    cites = b.uns[UNS_KEY][CITATIONS_KEY]
    assert {"scanpy", "umap", "umap-learn", "scikit-learn"} <= set(cites)


def test_native_float64_rep():
    a, b = make_rep(), make_rep()
    for x in (a, b):
        x.obsm["X_pca"] = x.obsm["X_pca"].astype(np.float64)
    neighbors(a, backend="scanpy")
    neighbors(b, backend="native")
    assert_same_graph(a, b)


def test_auto_is_scanpy():
    a, b = make_rep(), make_rep()
    neighbors(a)
    neighbors(b, backend="scanpy")
    for key in ("distances", "connectivities"):
        assert (a.obsp[key] != b.obsp[key]).nnz == 0


@pytest.mark.skipif(UMAP_VERSION < (0, 5, 12), reason="umap-learn fastmath bug")
@pytest.mark.parametrize("n_neighbors", [None, 5, 30])
def test_adjacency_recomputed_without_scanpy(monkeypatch, n_neighbors):
    import sclab.preprocess._scanpy_port as port

    adata = make_rep()
    neighbors(adata, n_neighbors=20, backend="native")
    expected = get_neighbors_adjacency_matrix(
        adata, n_neighbors=n_neighbors, weighted=True
    )
    monkeypatch.setattr(port, "resolve_backend", lambda _: "native")
    got = get_neighbors_adjacency_matrix(adata, n_neighbors=n_neighbors, weighted=True)
    np.testing.assert_array_equal(got.indices, expected.indices)
    np.testing.assert_allclose(got.data, expected.data, rtol=0, atol=CONN_ATOL)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_scanpy_float32_rep(metric):
    # umap-learn >= 0.6 rejected scanpy's float32 cosine kNN distances
    adata = make_rep()
    rep = adata.obsm["X_pca"]
    neighbors(adata, metric=metric, backend="scanpy")
    assert adata.obsm["X_pca"] is rep
    assert adata.obsp["connectivities"].nnz > 0


@pytest.mark.parametrize("umap_06", [False, True])
def test_float64_workaround_only_with_umap_06(monkeypatch, umap_06):
    monkeypatch.setattr(_neighbors, "_umap_rejects_float32", lambda: umap_06)
    seen = []
    sc_neighbors = sc.pp.neighbors

    def spy(adata, **kwargs):
        seen.append(adata.obsm[kwargs["use_rep"]].dtype)
        return sc_neighbors(adata, **kwargs)

    monkeypatch.setattr(sc.pp, "neighbors", spy)
    adata, expected = make_rep(), make_rep()
    neighbors(adata, n_neighbors=15, backend="scanpy")
    assert seen == [np.float64 if umap_06 else np.float32]
    assert adata.obsm["X_pca"].dtype == np.float32
    if not umap_06:  # umap-learn < 0.6: exactly scanpy's result
        sc_neighbors(expected, n_neighbors=15, use_rep="X_pca", n_pcs=10)
        for key in ("distances", "connectivities"):
            assert (adata.obsp[key] != expected.obsp[key]).nnz == 0
