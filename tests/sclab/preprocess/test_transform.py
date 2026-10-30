import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from sclab.preprocess import pool_neighbors
from sclab.preprocess._transform import periodic_pooling

from ._graph import graph_adata


def chain_adata(X):
    """0 - 1 - 2 chain graph (symmetric)."""
    edges = {(0, 1): (1.0, 0.5), (1, 0): (1.0, 0.5), (1, 2): (1.0, 0.5)}
    edges[(2, 1)] = (1.0, 0.5)
    return graph_adata(X, edges, k=2)


def line_adata(positions, k):
    """Fully connected graph; distances are |x_i - x_j|, X holds the positions."""
    n = len(positions)
    edges = {
        (i, j): (abs(positions[i] - positions[j]), 0.5)
        for i in range(n)
        for j in range(n)
        if i != j
    }
    return graph_adata(positions, edges, k=k)


def circ_dist(a, b, period=1.0):
    d = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


# ---------------------------------------------------------------------------
# pool_neighbors: where results are written
# ---------------------------------------------------------------------------


def test_pool_neighbors_default_values_and_key():
    adata = chain_adata([1.0, 2.0, 4.0])
    pool_neighbors(adata)

    assert "X_pooled" in adata.layers
    # each cell pools itself with its neighbors, uniform weights
    expected = np.array([[(1 + 2) / 2], [(1 + 2 + 4) / 3], [(2 + 4) / 2]])
    np.testing.assert_allclose(np.asarray(adata.layers["X_pooled"]), expected)

    # X is left alone
    np.testing.assert_array_equal(adata.X, [[1.0], [2.0], [4.0]])


def test_pool_neighbors_sparse_input_stays_sparse():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.X = csr_matrix(adata.X)
    pool_neighbors(adata)

    pooled = adata.layers["X_pooled"]
    assert issparse(pooled)
    np.testing.assert_allclose(pooled.toarray().ravel(), [1.5, 7 / 3, 3.0])
    assert issparse(adata.X)


def test_pool_neighbors_multiple_features():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    adata = chain_adata(X)
    pool_neighbors(adata)

    pooled = np.asarray(adata.layers["X_pooled"])
    assert pooled.shape == X.shape
    np.testing.assert_allclose(pooled[:, 1], 10 * pooled[:, 0])


def test_pool_neighbors_layer_key():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.layers["counts"] = np.array([[3.0], [6.0], [9.0]])
    pool_neighbors(adata, key="counts")

    assert "counts_pooled" in adata.layers
    assert "X_pooled" not in adata.layers
    np.testing.assert_allclose(
        np.asarray(adata.layers["counts_pooled"]).ravel(), [4.5, 6.0, 7.5]
    )
    np.testing.assert_array_equal(adata.layers["counts"].ravel(), [3.0, 6.0, 9.0])


def test_pool_neighbors_obsm_key():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.obsm["X_emb"] = np.array([[0.0, 3.0], [3.0, 0.0], [6.0, 6.0]])
    pool_neighbors(adata, key="X_emb")

    assert "X_emb_pooled" in adata.obsm
    np.testing.assert_allclose(
        adata.obsm["X_emb_pooled"],
        [[1.5, 1.5], [3.0, 3.0], [4.5, 3.0]],
    )


def test_pool_neighbors_obs_key():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.obs["score"] = [1.0, 2.0, 4.0]
    pool_neighbors(adata, key="score")

    assert "score_pooled" in adata.obs
    np.testing.assert_allclose(adata.obs["score_pooled"], [1.5, 7 / 3, 3.0])
    np.testing.assert_array_equal(adata.obs["score"], [1.0, 2.0, 4.0])


def test_pool_neighbors_key_added():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.layers["counts"] = np.array([[3.0], [6.0], [9.0]])
    pool_neighbors(adata, key="counts", key_added="custom")

    assert "custom" in adata.layers
    assert "counts_pooled" not in adata.layers
    np.testing.assert_allclose(
        np.asarray(adata.layers["custom"]).ravel(), [4.5, 6.0, 7.5]
    )


@pytest.mark.xfail(
    strict=True,
    reason="BUG: key_added always writes to adata.layers, even for obsm keys "
    "whose width differs from n_vars",
)
def test_pool_neighbors_key_added_with_obsm_key():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.obsm["X_emb"] = np.eye(3)[:, :2]
    pool_neighbors(adata, key="X_emb", key_added="custom")

    assert "custom" in adata.obsm


def test_pool_neighbors_copy_returns_without_writing():
    adata = chain_adata([1.0, 2.0, 4.0])
    result = pool_neighbors(adata, copy=True)

    assert result is not None
    np.testing.assert_allclose(np.asarray(result).ravel(), [1.5, 7 / 3, 3.0])
    assert len(adata.layers) == 0


def test_pool_neighbors_returns_none_by_default():
    adata = chain_adata([1.0, 2.0, 4.0])
    assert pool_neighbors(adata) is None


# ---------------------------------------------------------------------------
# pool_neighbors: weighting / directedness / n_neighbors
# ---------------------------------------------------------------------------


def test_pool_neighbors_weighted_uses_connectivities():
    # cell 0 has two neighbors with different connectivities
    edges = {(0, 1): (1.0, 0.5), (0, 2): (2.0, 0.25)}
    adata = graph_adata([0.0, 4.0, 8.0], edges, k=2)

    pool_neighbors(adata, weighted=True)
    pooled = np.asarray(adata.layers["X_pooled"]).ravel()

    # self weight is 1, neighbors weighted by connectivity, then row-normalized
    assert pooled[0] == pytest.approx((0 * 1 + 4 * 0.5 + 8 * 0.25) / 1.75)
    # cells without outgoing edges keep their own value
    np.testing.assert_allclose(pooled[1:], [4.0, 8.0])


def test_pool_neighbors_unweighted_ignores_connectivity_values():
    edges = {(0, 1): (1.0, 0.5), (0, 2): (2.0, 0.25)}
    adata = graph_adata([0.0, 4.0, 8.0], edges, k=2)

    pool_neighbors(adata, weighted=False)
    pooled = np.asarray(adata.layers["X_pooled"]).ravel()
    assert pooled[0] == pytest.approx((0 + 4 + 8) / 3)


def test_pool_neighbors_directed_vs_undirected():
    # only one directed edge: 0 -> 1
    adata = graph_adata([0.0, 6.0], {(0, 1): (1.0, 0.5)}, k=1)

    pool_neighbors(adata, directed=True, key_added="directed")
    pool_neighbors(adata, directed=False, key_added="undirected")

    np.testing.assert_allclose(np.asarray(adata.layers["directed"]).ravel(), [3.0, 6.0])
    np.testing.assert_allclose(
        np.asarray(adata.layers["undirected"]).ravel(), [3.0, 3.0]
    )


def test_pool_neighbors_n_neighbors_truncates_graph():
    # 4 points on a line, each cell connected to every other (k=3)
    adata = line_adata([0.0, 1.0, 2.0, 3.0], k=3)

    # n_neighbors counts the cell itself -> keep the 2 closest per row
    pool_neighbors(adata, n_neighbors=3, key_added="trunc")
    pooled = np.asarray(adata.layers["trunc"]).ravel()

    # cell 0: {1, 2}; cell 1: {0, 2}; cell 2: {1, 3}; cell 3: {2, 1}
    np.testing.assert_allclose(
        pooled, [(0 + 1 + 2) / 3, (1 + 0 + 2) / 3, (2 + 1 + 3) / 3, (3 + 2 + 1) / 3]
    )

    # without truncation, everyone pools everyone
    pool_neighbors(adata, key_added="full")
    np.testing.assert_allclose(np.asarray(adata.layers["full"]).ravel(), 1.5)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _filter_knn_matrix builds csr_matrix without shape=; "
    "shape is inferred from max column index",
)
def test_pool_neighbors_n_neighbors_when_last_cell_not_selected():
    # cell 3 is far from everyone, so it is nobody's nearest neighbor and the
    # truncated adjacency loses its last column
    adata = line_adata([0.0, 1.0, 2.0, 10.0], k=3)

    pool_neighbors(adata, n_neighbors=3, key_added="trunc")
    pooled = np.asarray(adata.layers["trunc"]).ravel()

    # cell 0: {1, 2}; cell 1: {0, 2}; cell 2: {1, 0}; cell 3: {2, 1}
    np.testing.assert_allclose(
        pooled,
        [(0 + 1 + 2) / 3, (1 + 0 + 2) / 3, (2 + 1 + 0) / 3, (10 + 2 + 1) / 3],
    )


# ---------------------------------------------------------------------------
# pool_neighbors: error paths
# ---------------------------------------------------------------------------


def test_pool_neighbors_unknown_key_raises():
    adata = chain_adata([1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="Unknown key"):
        pool_neighbors(adata, key="does_not_exist")


def test_pool_neighbors_periodic_requires_obs_key():
    adata = chain_adata([1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="obs"):
        pool_neighbors(adata, key_periodic=True, key_min=0, key_max=1)


def test_pool_neighbors_periodic_requires_bounds():
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.obs["phase"] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="key_min and key_max"):
        pool_neighbors(adata, key="phase", key_periodic=True)
    with pytest.raises(ValueError, match="key_min and key_max"):
        pool_neighbors(adata, key="phase", key_periodic=True, key_min=0)


def test_pool_neighbors_missing_neighbors_raises():
    adata = AnnData(np.ones((3, 2)))
    with pytest.raises(KeyError):
        pool_neighbors(adata)


# ---------------------------------------------------------------------------
# periodic pooling
# ---------------------------------------------------------------------------


def test_periodic_pooling_across_wrap_boundary():
    # 0.9 and 0.1 are 0.2 apart on a circle of period 1; their mean is 0.0,
    # not the arithmetic 0.5
    W = csr_matrix(np.full((2, 2), 0.5))
    X = np.array([0.9, 0.1])

    pooled = periodic_pooling(W, X, 0.0, 1.0)

    assert pooled.shape == (2, 1)
    assert np.all(circ_dist(pooled.ravel(), 0.0) < 1e-9)
    assert np.all((pooled >= 0.0) & (pooled < 1.0))


def test_periodic_pooling_no_wrap_matches_linear_mean():
    W = csr_matrix(np.full((2, 2), 0.5))
    X = np.array([0.2, 0.4])

    pooled = periodic_pooling(W, X, 0.0, 1.0)
    np.testing.assert_allclose(pooled.ravel(), [0.3, 0.3])


def test_periodic_pooling_identity_weights_is_identity():
    X = np.array([0.05, 0.5, 0.95])
    pooled = periodic_pooling(sp.identity(3, format="csr"), X, 0.0, 1.0)
    np.testing.assert_allclose(pooled.ravel(), X)


def test_periodic_pooling_respects_custom_range():
    # period 2*pi on [0, 2*pi): 6.0 and 0.5 are 0.783 apart across the wrap
    period = 2 * np.pi
    W = csr_matrix(np.full((2, 2), 0.5))
    X = np.array([6.0, 0.5])

    pooled = periodic_pooling(W, X, 0.0, period).ravel()

    expected = (6.0 + (0.5 + period - 6.0) / 2) % period
    np.testing.assert_allclose(pooled, [expected, expected])
    assert np.all((pooled >= 0) & (pooled < period))


def test_periodic_pooling_nonzero_xmin():
    # period 10 on [-5, 5): -4.5 and 4.5 are 1.0 apart across the wrap
    W = csr_matrix(np.full((2, 2), 0.5))
    pooled = periodic_pooling(W, np.array([-4.5, 4.5]), -5.0, 5.0).ravel()

    assert np.all(circ_dist(pooled, -5.0, period=10.0) < 1e-9)
    assert np.all((pooled >= -5.0) & (pooled < 5.0))


def test_pool_neighbors_periodic_end_to_end():
    # phases straddle the 0/1 boundary; chain 0-1-2
    adata = chain_adata([1.0, 2.0, 4.0])
    adata.obs["phase"] = [0.95, 0.05, 0.10]

    pool_neighbors(adata, key="phase", key_periodic=True, key_min=0.0, key_max=1.0)

    pooled = adata.obs["phase_pooled"].to_numpy()
    # cell 0: {0.95, 0.05} -> circular mean 0.0 (arithmetic mean would be 0.5)
    assert circ_dist(pooled[0], 0.0) < 1e-9
    # cell 1: {0.95, 0.05, 0.10} -> offsets from 0.05: -0.1, 0, 0.05 -> 0.05 - 0.05/3
    assert pooled[1] == pytest.approx(0.05 - 0.05 / 3)
    # cell 2: {0.05, 0.10} -> 0.075
    assert pooled[2] == pytest.approx(0.075)
    assert np.all((pooled >= 0) & (pooled < 1))
