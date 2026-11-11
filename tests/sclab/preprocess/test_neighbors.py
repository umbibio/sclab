"""sclab.preprocess.neighbors."""

import numpy as np
import pytest
from anndata import AnnData

from sclab.preprocess import _neighbors, neighbors

sc = pytest.importorskip("scanpy")


def make_rep(n_obs=300, n_dims=12, seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, (5, n_dims))
    x = centers[rng.integers(0, 5, n_obs)] + rng.normal(size=(n_obs, n_dims))
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsm["X_pca"] = x.astype(np.float32)
    return adata


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_scanpy_float32_rep(metric):
    # umap-learn >= 0.6 rejected scanpy's float32 cosine kNN distances
    adata = make_rep()
    rep = adata.obsm["X_pca"]
    neighbors(adata, metric=metric)
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
    neighbors(adata, n_neighbors=15)
    assert seen == [np.float64 if umap_06 else np.float32]
    assert adata.obsm["X_pca"].dtype == np.float32
    if not umap_06:  # umap-learn < 0.6: exactly scanpy's result
        sc_neighbors(expected, n_neighbors=15, use_rep="X_pca", n_pcs=10)
        for key in ("distances", "connectivities"):
            assert (adata.obsp[key] != expected.obsp[key]).nnz == 0
