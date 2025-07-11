from typing import Optional

from anndata import AnnData
from numpy import ndarray
from scipy.sparse import csr_matrix

from ._utils import get_neighbors_adjacency_matrix


def pool_neighbors(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    n_neighbors: Optional[int] = None,
    neighbors_key: Optional[str] = None,
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
    layer : str, optional
        Layer in AnnData object to use for pooling. Defaults to None.
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
    if layer is None or layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]

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

    pooled = W.dot(X)

    if copy:
        return pooled

    if key_added is not None:
        adata.layers[key_added] = pooled
        return

    if layer is None or layer == "X":
        adata.X = pooled
    else:
        adata.layers[layer] = pooled
