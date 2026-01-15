from typing import Optional

import numpy as np
from anndata import AnnData
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix, spmatrix

from ._utils import get_neighbors_adjacency_matrix


def pool_neighbors(
    adata: AnnData,
    *,
    key: str | None = None,
    key_periodic: bool = False,
    key_min: float | None = None,
    key_max: float | None = None,
    n_neighbors: Optional[int] = None,
    neighbors_key: str = "neighbors",
    weighted: bool = False,
    directed: bool = True,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> csr_matrix | ndarray | None:
    """
    Given an adjacency matrix, pool cell features using a weighted sum of feature counts
    from neighboring cells. The weights are the normalized connectivities from the
    adjacency matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    key : str, optional
        Key in AnnData object to use for pooling. It can be a key in adata.obs,
        adata.layers, or adata.obsm. Defaults to None.
    key_periodic : bool, optional
        Whether to use periodic boundary conditions for pooling. It is only used
        if key is a key in adata.obs. Defaults to False.
    key_min : float, optional
        Minimum value for column in adata.obs to use for pooling. It is only used
        if key is a key in adata.obs. Defaults to None. Must be provided if
        `key_periodic` is True.
    key_max : float, optional
        Maximum value for column in adata.obs to use for pooling. It is only used
        if key is a key in adata.obs. Defaults to None. Must be provided if
        `key_periodic` is True.
    n_neighbors : int, optional
        Number of neighbors to consider. Defaults to None.
    neighbors_key : str, optional
        Key in AnnData object to use for neighbors. Defaults to None.
    weighted : bool, optional
        Whether to weight neighbors by their connectivities in the adjacency matrix.
        Defaults to False.
    directed : bool, optional
        Whether to use directed or undirected neighbors. Defaults to True.
    key_added : str, optional
        Key to use in AnnData object for the pooled features. Defaults to None.
    copy : bool, optional
        Whether to return a copy of the pooled features instead of modifying the
        original AnnData object. Defaults to False.

    Returns
    -------
    csr_matrix | ndarray | None
        The pooled features if copy is True, otherwise None.
    """
    if key is None:
        key = "X"

    if key == "X":
        X = adata.X
        key_type = "X"
    elif key in adata.layers:
        X = adata.layers[key]
        key_type = "X"
    elif key in adata.obsm:
        X = adata.obsm[key]
        key_type = "obsm"
    elif key in adata.obs:
        X = adata.obs[[key]].values
        key_type = "obs"
    else:
        raise ValueError(f"Unknown key {key}")

    if key_periodic:
        if key_type != "obs":
            raise ValueError("key_type must be 'obs' for periodic pooling")
        if key_min is None or key_max is None:
            raise ValueError(
                "key_min and key_max must be specified for periodic pooling"
            )

    adjacency = get_neighbors_adjacency_matrix(
        adata,
        key=neighbors_key,
        n_neighbors=n_neighbors,
        weighted=weighted,
        directed=directed,
    )

    W = adjacency.tolil()
    W.setdiag(1)

    W = W / W.sum(axis=1)

    if key_periodic:
        pooled = periodic_pooling(W, X, key_min, key_max)
    else:
        pooled = W.dot(X)

    if copy:
        return pooled

    if key_added is not None:
        adata.layers[key_added] = pooled
        return

    if key_type == "X":
        adata.layers[f"{key}_pooled"] = pooled

    elif key_type == "obsm":
        adata.obsm[f"{key}_pooled"] = pooled

    elif key_type == "obs":
        adata.obs[[f"{key}_pooled"]] = pooled


def periodic_pooling(W: spmatrix, X: ndarray, xmin: float, xmax: float) -> ndarray:
    """
    Weighted pooling for periodic values using local unwrapping.
    Assumes W is row-normalized (rows sum to 1).

    Complexity: O(nnz) where nnz = number of non-zero entries in W
    """

    X = X.ravel()

    period = xmax - xmin
    half = period / 2

    W_coo = coo_matrix(W)

    # Compute wrapped differences only for non-zero entries
    diff = X[W_coo.col] - X[W_coo.row]
    diff_wrapped = (diff + half) % period - half

    # Sparse matrix of weighted corrections
    corrections = coo_matrix(
        (W_coo.data * diff_wrapped, (W_coo.row, W_coo.col)), shape=W.shape
    )

    # pooled[i] = X[i] + sum_j W[i,j] * wrap(X[j] - X[i])
    pooled = X + np.asarray(corrections.sum(axis=1)).ravel()

    # Wrap back to [xmin, xmax)
    pooled = xmin + (pooled - xmin) % period
    return pooled.reshape((-1, 1))
