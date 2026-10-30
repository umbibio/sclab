import numpy as np
import scipy.sparse as sp
from anndata import AnnData


def graph_adata(X, edges: dict[tuple[int, int], tuple[float, float]], k: int):
    """AnnData with a hand-built neighbors graph.

    ``edges`` maps (row, col) -> (distance, connectivity). ``k`` is the number of
    neighbors per row, so the stored ``n_neighbors`` param is ``k + 1`` (scanpy
    counts the cell itself).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]

    D = sp.lil_matrix((n, n))
    C = sp.lil_matrix((n, n))
    for (i, j), (d, c) in edges.items():
        D[i, j] = d
        C[i, j] = c

    adata = AnnData(X)
    adata.obsp["distances"] = D.tocsr()
    adata.obsp["connectivities"] = C.tocsr()
    adata.uns["neighbors"] = {
        "params": {"n_neighbors": k + 1, "method": "umap", "metric": "euclidean"},
        "connectivities_key": "connectivities",
        "distances_key": "distances",
    }
    return adata
