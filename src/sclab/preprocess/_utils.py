from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from scipy.sparse import coo_matrix, csr_matrix


def get_neighbors_adjacency_matrix(
    adata: AnnData,
    *,
    key: Optional[str] = "neighbors",
    n_neighbors: Optional[int] = None,
    weighted: bool = False,
    directed: bool = True,
) -> csr_matrix:
    # the stored graph, as scanpy's Neighbors(adata, neighbors_key=key) reads it
    uns = adata.uns[key]
    params = uns["params"]
    distances = adata.obsp[uns.get("distances_key", "distances")]
    connectivities = adata.obsp[uns.get("connectivities_key", "connectivities")]
    current = params["n_neighbors"]

    if n_neighbors is None:
        n_neighbors = current

    if n_neighbors < current and not weighted:
        distances = _filter_knn_matrix(
            distances, n_neighbors=n_neighbors, mode="distances"
        )

    elif n_neighbors != current:
        distances, connectivities = _recompute_neighbors(
            adata, {**params, "n_neighbors": n_neighbors}, key
        )

    adjacency = distances.copy()
    adjacency.data = np.ones_like(adjacency.data)

    if not directed:
        # make the adjacency matrix symmetric
        adjacency = _symmetrize_sparse_matrix(adjacency)

    if weighted:
        # use the connectivities to assign weights
        adjacency = adjacency.multiply(connectivities)

    return adjacency


def _recompute_neighbors(adata: AnnData, params: dict, key: str):
    """``(distances, connectivities)`` of the graph ``params`` describe:
    scanpy's ``Neighbors.compute_neighbors`` if scanpy is installed, else
    sclab's port (method ``"umap"`` only)."""
    from ._scanpy_port import _choose_rep, knn_graph, resolve_backend

    if resolve_backend("auto") == "scanpy":
        from scanpy import Neighbors

        neigh = Neighbors(adata, neighbors_key=key)
        neigh.compute_neighbors(**params)
        return neigh.distances, neigh.connectivities
    if params.get("method", "umap") != "umap":
        raise NotImplementedError("without scanpy, only method='umap' is ported")
    x = _choose_rep(adata, params["use_rep"], params.get("n_pcs"))
    return knn_graph(x, params["n_neighbors"], params.get("metric", "euclidean"))


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

    return csr_matrix((data, indices, indptr), shape=matrix.shape)


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
