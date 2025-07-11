from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scanpy import Neighbors
from scipy.sparse import coo_matrix, csr_matrix


def get_neighbors_adjacency_matrix(
    adata: AnnData,
    *,
    key: Optional[str] = "neighbors",
    n_neighbors: Optional[int] = None,
    weighted: bool = False,
    directed: bool = True,
) -> csr_matrix:
    # get the current neighbors
    neigh = Neighbors(adata, neighbors_key=key)
    params = adata.uns[key]["params"]

    if n_neighbors is None:
        n_neighbors = neigh.n_neighbors

    if n_neighbors < neigh.n_neighbors and not weighted:
        distances = _filter_knn_matrix(
            neigh.distances, n_neighbors=n_neighbors, mode="distances"
        )

    elif n_neighbors != neigh.n_neighbors:
        neigh.compute_neighbors(**{**params, "n_neighbors": n_neighbors})
        distances = neigh.distances

    else:
        distances = neigh.distances

    adjacency = distances.copy()
    adjacency.data = np.ones_like(adjacency.data)

    if not directed:
        # make the adjacency matrix symmetric
        adjacency = _symmetrize_sparse_matrix(adjacency)

    if weighted:
        # use the connectivities to assign weights
        adjacency = adjacency.multiply(neigh.connectivities)

    return adjacency


def _filter_knn_matrix(
    matrix: csr_matrix, *, n_neighbors: int, mode: Literal["distances", "weights"]
) -> csr_matrix:
    assert mode in ["distances", "weights"]
    nrows, _ = matrix.shape

    # Initialize arrays for new sparse matrix with pre-allocated size
    indptr = np.arange(0, (n_neighbors - 1) * (nrows + 1), n_neighbors - 1)
    data = np.zeros(nrows * (n_neighbors - 1), dtype=float)
    indices = np.zeros(nrows * (n_neighbors - 1), dtype=int)

    # Process each row to keep top n_neighbors-1 connections
    for i in range(nrows):
        start, end = matrix.indptr[i : i + 2]
        idxs = matrix.indices[start:end]
        vals = matrix.data[start:end]

        # Sort by values and keep top n_neighbors-1
        if mode == "weights":
            # Sort in descending order (keep largest weights)
            o = np.argsort(-vals)[: n_neighbors - 1]
        else:
            # Sort in ascending order (keep smallest distances)
            o = np.argsort(vals)[: n_neighbors - 1]

        # Maintain original order within top neighbors
        oo = np.argsort(idxs[o])
        start, end = indptr[i : i + 2]
        indices[start:end] = idxs[o][oo]
        data[start:end] = vals[o][oo]

    return csr_matrix((data, indices, indptr))


def _symmetrize_sparse_matrix(matrix: csr_matrix) -> csr_matrix:
    A = matrix.tocoo()

    # Make matrix symmetric by duplicating entries in both directions
    coords = np.array([[*A.row, *A.col], [*A.col, *A.row]])
    data = np.array([*A.data, *A.data])

    # Remove duplicate entries that might occur in symmetrization
    idxs = np.unique(coords, axis=1, return_index=True)[1]
    coords, data = coords[:, idxs], data[idxs]
    A = coo_matrix((data, coords), shape=matrix.shape)

    return A.tocsr()
