"""numpy port of umap-learn's ``fuzzy_simplicial_set``, as
:func:`scanpy.pp.neighbors` calls it to compute the connectivities (method
``"umap"``), for environments without umap-learn and numba (e.g. Pyodide).

Ported from umap-learn 0.5.12 (https://github.com/lmcinnes/umap, BSD-3-Clause,
Copyright (c) 2017 Leland McInnes; see THIRD_PARTY_NOTICES.md), file
``umap/umap_.py``: ``smooth_knn_dist``, ``compute_membership_strengths`` and
``fuzzy_simplicial_set``, with only the options scanpy uses (precomputed kNN,
``local_connectivity=1``, ``set_op_mix_ratio=1``, ``bandwidth=1``). The numba
loops over cells are vectorized; the float32 arithmetic and the order of the
sums are kept, so results match umap-learn's within float32 rounding of
``exp`` (see ``tests/sclab/preprocess/test_neighbors.py``).

References: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform
Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
https://doi.org/10.48550/arXiv.1802.03426 -- McInnes, L., Healy, J., Saul, N.,
& Großberger, L. (2018). UMAP: Uniform Manifold Approximation and Projection.
Journal of Open Source Software, 3(29), 861. https://doi.org/10.21105/joss.00861
"""

import numpy as np
from scipy import sparse

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_FLOATMAX = np.finfo(np.float32).max


def _exp32(x: np.ndarray) -> np.ndarray:
    """float32 ``exp``, rounded from float64: it matches the ``expf`` numba
    calls in all but ~1 in 2000 values, numpy's float32 ``exp`` differs by 1
    ulp in a third of them."""
    return np.exp(x.astype(np.float64)).astype(np.float32)


def _sum32(x: np.ndarray) -> np.ndarray:
    """Row sums of a 2-D float32 array, accumulated left to right in float32
    (as the numba loops do; ``np.sum`` sums pairwise)."""
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], np.float32)
    return np.cumsum(x, axis=1, dtype=np.float32)[:, -1]


def smooth_knn_dist(
    distances: np.ndarray, k: float, n_iter: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """umap-learn's ``smooth_knn_dist`` with ``local_connectivity=1`` and
    ``bandwidth=1``: per cell, ``rho`` (distance to the nearest neighbor at a
    positive distance) and ``sigma`` (binary search for
    ``sum_j exp(-(d_j - rho) / sigma) = log2(k)`` over neighbors 1..k-1).

    ``distances`` is float32, shape ``(n_obs, k)``, rows sorted ascending.
    Returns ``(sigmas, rhos)``, float32.
    """
    n = distances.shape[0]
    target = np.log2(k)
    rows = np.arange(n)

    nonzero = distances > 0.0
    first = nonzero.argmax(axis=1)  # the first positive distance in each row
    rho = np.where(nonzero.any(axis=1), distances[rows, first], np.float32(0))

    d = distances[:, 1:] - rho[:, None]
    lo = np.zeros(n, np.float32)
    hi = np.full(n, NPY_FLOATMAX, np.float32)
    mid = np.ones(n, np.float32)
    active = np.arange(n)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for _ in range(n_iter):
            if not active.size:
                break
            da = d[active]
            terms = np.where(da > 0, _exp32(-(da / mid[active, None])), np.float32(1))
            psum = _sum32(terms).astype(np.float64)
            done = np.abs(psum - target) < SMOOTH_K_TOLERANCE
            active, psum = active[~done], psum[~done]
            above = psum > target
            i = active[above]  # too many neighbors: shrink sigma
            hi[i] = mid[i]
            mid[i] = (lo[i] + hi[i]) / np.float32(2)
            i = active[~above]  # too few: grow it
            lo[i] = mid[i]
            unbounded = hi[i] >= NPY_FLOATMAX
            mid[i] = np.where(
                unbounded, mid[i] * np.float32(2), (lo[i] + hi[i]) / np.float32(2)
            )

    # floors: a fraction of the row's (or, without a positive rho, the global)
    # mean distance, accumulated in float32 like numba's np.mean
    row_mean = _sum32(distances).astype(np.float64) / distances.shape[1]
    all_mean = np.float64(_sum32(distances.reshape(1, -1))[0]) / distances.size
    floor = MIN_K_DIST_SCALE * np.where(rho > 0.0, row_mean, all_mean)
    sigma = np.where(mid < floor, floor, mid).astype(np.float32)
    return sigma, rho.astype(np.float32)


def compute_membership_strengths(
    knn_indices: np.ndarray, knn_dists: np.ndarray, sigmas: np.ndarray, rhos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """umap-learn's ``compute_membership_strengths`` (not bipartite): the
    COO ``rows``, ``cols`` (int32) and ``vals`` (float32) of each cell's local
    fuzzy set, ``exp(-(d - rho) / sigma)``, 1 within ``rho``, 0 for the cell
    itself."""
    n, k = knn_indices.shape
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        d = knn_dists - rhos[:, None]
        vals = _exp32(-(d / sigmas[:, None]))
    vals = np.where((d <= 0.0) | (sigmas[:, None] == 0.0), np.float32(1), vals)
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    vals[knn_indices == rows.reshape(n, k)] = 0.0
    return rows, knn_indices.astype(np.int32).ravel(), vals.astype(np.float32).ravel()


def fuzzy_simplicial_set(
    knn_indices: np.ndarray, knn_dists: np.ndarray, n_obs: int, n_neighbors: int
) -> sparse.csr_matrix:
    """umap-learn's ``fuzzy_simplicial_set`` on a precomputed kNN graph, as
    ``scanpy.neighbors._connectivity.umap`` calls it: the fuzzy union
    ``A + A.T - A * A.T`` of the local fuzzy sets, as CSR (float32)."""
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(n_neighbors))
    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )
    result = sparse.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    # set_op_mix_ratio = 1: umap-learn's expression, for the same rounding
    result = 1.0 * (result + transpose - prod_matrix) + 0.0 * prod_matrix
    result.eliminate_zeros()
    return result.tocsr()
