"""sclab's numpy port of scanpy.tl.umap / umap-learn's layout.

The layout cannot match umap-learn's bit for bit (its numba SGD is sequential,
with its own RNG), so it is compared with scanpy's on the same graph by
trustworthiness, kNN preservation and cluster silhouette. On 600 blobs cells the
two agree within ~0.01 (trustworthiness) and ~0.05 (silhouette); on 3k and 10k
real cells within 0.001 and 0.01.
"""

import sys

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from sclab._citations import active_citations
from sclab.tools.embedding import _umap_port as port
from sclab.tools.embedding import umap


def knn_graph(x, k=15):
    """A fuzzy kNN graph in the style of umap-learn's (exp(-(d - rho) / sigma),
    fuzzy union), without umap-learn."""
    dist, idx = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
    d = dist[:, 1:] - dist[:, 1:2]
    sigma = np.maximum(d.mean(1, keepdims=True), 1e-3)
    n = len(x)
    rows = np.repeat(np.arange(n), k - 1)
    a = sp.csr_matrix((np.exp(-d / sigma).ravel(), (rows, idx[:, 1:].ravel())), (n, n))
    return (a + a.T - a.multiply(a.T)).tocsr()


def blobs_adata(n=600, centers=6, seed=0, k=15, **kwargs):
    x, labels = make_blobs(
        n, n_features=10, centers=centers, cluster_std=2.0, random_state=seed, **kwargs
    )
    adata = AnnData(np.zeros((n, 1), np.float32), obsm={"X_pca": x})
    adata.obs["label"] = labels.astype(str)
    adata.obsp["connectivities"] = knn_graph(x, k)
    adata.uns["neighbors"] = {
        "params": {"n_neighbors": k, "method": "umap", "use_rep": "X_pca"},
        "connectivities_key": "connectivities",
    }
    return adata


def test_find_ab_params():
    # umap-learn 0.5.12's values; scanpy's default min_dist=0.5, umap's 0.1
    np.testing.assert_allclose(port.find_ab_params(1.0, 0.5), (0.58303, 1.33417), 1e-4)
    np.testing.assert_allclose(port.find_ab_params(1.0, 0.1), (1.57694, 0.89506), 1e-4)


def test_make_epochs_per_sample():
    eps = port.make_epochs_per_sample(np.array([1.0, 0.5, 0.1, 0.0]), 200)
    np.testing.assert_allclose(eps, [1.0, 2.0, 10.0, -1.0])


def test_matches_umap_learn_helpers():
    umap_ = pytest.importorskip("umap.umap_")
    spectral = pytest.importorskip("umap.spectral")
    w = np.random.default_rng(0).random(50)
    np.testing.assert_array_equal(
        port.make_epochs_per_sample(w, 500), umap_.make_epochs_per_sample(w, 500)
    )
    assert port.find_ab_params(1.5, 0.3) == umap_.find_ab_params(1.5, 0.3)
    graph = blobs_adata(centers=1).obsp["connectivities"].tocoo()
    ours = port.spectral_layout(None, graph, 2, np.random.RandomState(0))
    theirs = spectral.spectral_layout(None, graph, 2, np.random.RandomState(0))
    np.testing.assert_allclose(np.abs(ours), np.abs(theirs), atol=1e-6)  # signs


def test_deterministic_and_seeded():
    adata = blobs_adata()
    runs = []
    for seed in (0, 0, 1):
        b = adata.copy()
        umap(b, random_state=seed, backend="native")
        runs.append(b.obsm["X_2Dumap"])
    assert runs[0].dtype == np.float32 and runs[0].shape == (600, 2)
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.allclose(runs[0], runs[2])
    assert b.uns["umap"]["params"]["random_state"] == 1
    np.testing.assert_allclose(
        [b.uns["umap"]["params"]["a"], b.uns["umap"]["params"]["b"]],
        [0.58303, 1.33417],
        rtol=1e-4,
    )


def test_3d_and_zero_center():
    adata = blobs_adata(n=200)
    key = umap(adata, n_components=3, zero_center=True, backend="native")
    assert key == "X_3Dumap"
    np.testing.assert_allclose(adata.obsm[key].mean(0), 0, atol=1e-4)


def quality(adata, y):
    x = adata.obsm["X_pca"]
    k = 15
    a = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[1][:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(y).kneighbors(y)[1][:, 1:]
    knn = np.mean([len(set(p) & set(q)) / k for p, q in zip(a, b)])
    trust = trustworthiness(x, y, n_neighbors=k)
    return trust, knn, silhouette_score(y, adata.obs["label"])


def test_quality_close_to_scanpy():
    sc = pytest.importorskip("scanpy")
    pytest.importorskip("umap")
    adata = blobs_adata()
    ref = adata.copy()
    sc.tl.umap(ref, random_state=0)
    umap(adata, backend="native")
    trust, knn, sil = quality(adata, adata.obsm["X_2Dumap"])
    trust_ref, knn_ref, sil_ref = quality(ref, ref.obsm["X_umap"])
    assert trust > 0.9 and sil > 0.5  # well separated blobs
    assert trust >= trust_ref - 0.02
    assert knn >= knn_ref - 0.05
    assert sil >= sil_ref - 0.1


@pytest.mark.filterwarnings("ignore:Graph is not fully connected")  # centroids
def test_disconnected_components():
    # far apart blobs whose kNN graphs do not touch: one spectral layout each,
    # placed by umap-learn's meta layout (2 <= 2 * dim components: +-unit
    # vectors; 6 > 2 * dim: a spectral embedding of the centroids)
    for centers in (2, 6):
        adata = blobs_adata(n=300, centers=centers, center_box=(-500, 500))
        n_comps, _ = sp.csgraph.connected_components(adata.obsp["connectivities"])
        assert n_comps == centers
        umap(adata, backend="native")
        y = adata.obsm["X_2Dumap"]
        assert np.isfinite(y).all()
        assert silhouette_score(y, adata.obs["label"]) > 0.7


def test_spectral_failure_falls_back_to_random(monkeypatch):
    def fail(*args, **kwargs):
        raise sp.linalg.ArpackNoConvergence("no", [], [])

    monkeypatch.setattr(port.scipy.sparse.linalg, "eigsh", fail)
    adata = blobs_adata(n=200, centers=1)
    with pytest.warns(UserWarning, match="Falling back to random initialisation"):
        umap(adata, backend="native")
    assert np.isfinite(adata.obsm["X_2Dumap"]).all()


def test_native_without_scanpy(monkeypatch):
    for m in ("scanpy", "umap", "pynndescent", "numba"):
        monkeypatch.setitem(sys.modules, m, None)
    adata = blobs_adata(n=200)
    assert umap(adata) == "X_2Dumap"  # auto: native
    keys = [e["key"] for e in active_citations(adata, format="dict")]
    assert {"umap", "umap-learn"} <= set(keys)
    with pytest.raises(ImportError, match="scanpy"):
        umap(adata, backend="scanpy")


def test_needs_a_graph():
    with pytest.raises(ValueError, match="neighbors"):
        umap(AnnData(np.zeros((5, 2))), backend="native")
