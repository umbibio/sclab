"""numpy/scikit-learn port of BBKNN (batch balanced kNN), as
:func:`scanpy.external.pp.bbknn` runs it, for environments without bbknn,
its kNN libraries (annoy, pynndescent) and numba (e.g. Pyodide).

Ported from bbknn 1.6.0 (https://github.com/Teichlab/bbknn, MIT License,
Copyright (c) 2018 The Python Packaging Authority, as its LICENSE file reads;
see THIRD_PARTY_NOTICES.md), files ``bbknn/__init__.py`` (``bbknn``: the
AnnData wrapper) and ``bbknn/matrix.py`` (``bbknn``, ``get_graph``,
``compute_connectivities_umap``, ``get_sparse_matrix_from_indices_distances_umap``
and ``trimming``). Each cell's ``neighbors_within_batch`` nearest neighbors are
found in every batch separately (exact search with scikit-learn, where bbknn
offers annoy, pynndescent, faiss, cKDTree or KDTree), merged and sorted by
distance; the connectivities are umap-learn's fuzzy simplicial set
(:mod:`._fuzzy_simplicial_set`) over the merged lists, trimmed to each cell's
``trim`` strongest. With bbknn's exact ``computation="cKDTree"`` the results
match (see ``tests/sclab/preprocess/test_bbknn.py``).

References: Polański, K., Young, M. D., Miao, Z., Meyer, K. B., Teichmann,
S. A., & Park, J.-E. (2020). BBKNN: fast batch alignment of single cell
transcriptomes. Bioinformatics, 36(3), 964-965.
https://doi.org/10.1093/bioinformatics/btz625 -- and McInnes et al. (2018)
for the connectivities (see :mod:`._fuzzy_simplicial_set`).
"""

import numpy as np
from anndata import AnnData
from scipy import sparse


def bbknn(
    adata: AnnData,
    *,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    metric: str = "euclidean",
    neighbors_within_batch: int = 3,
    n_pcs: int = 50,
    trim: int | None = None,
) -> None:
    """``scanpy.external.pp.bbknn`` with an exact kNN search (scikit-learn,
    brute force, any of its metrics) and bbknn's defaults otherwise
    (``set_op_mix_ratio=1``, ``local_connectivity=1``). Stores
    ``obsp["distances"]``, ``obsp["connectivities"]`` and ``uns["neighbors"]``
    as bbknn does; ``params["bbknn"]["computation"]`` is ``"brute"``.
    ``trim=None`` keeps 10 times the neighbors of a cell, 0 does not trim.
    """
    from sklearn.neighbors import NearestNeighbors

    from ._fuzzy_simplicial_set import fuzzy_simplicial_set

    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not present in `adata.obs`.")
    if use_rep not in adata.obsm:
        raise ValueError(f"Did not find {use_rep} in `.obsm.keys()`.")
    pca = adata.obsm[use_rep]
    n_pcs = int(min(n_pcs, pca.shape[1]))
    x = np.asarray(pca[:, :n_pcs], dtype=np.float64)  # cKDTree's precision
    batch_list = np.asarray([str(b) for b in adata.obs[batch_key].values])
    batches, counts = np.unique(batch_list, return_counts=True)
    k = neighbors_within_batch
    if counts.min() < k:
        raise ValueError(
            "Not all batches have at least `neighbors_within_batch` cells in them."
        )

    # get_graph: each batch's k nearest of every cell, a block of k columns each
    n, m = x.shape[0], k * len(batches)
    dist = np.empty((n, m))
    ind = np.empty((n, m), dtype=np.int64)
    for b, batch in enumerate(batches):
        ind_to = np.flatnonzero(batch_list == batch)
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
        d, i = nn.fit(x[ind_to]).kneighbors(x)
        dist[:, b * k : (b + 1) * k] = d
        ind[:, b * k : (b + 1) * k] = ind_to[i]
    rows = np.arange(n)[:, None]
    dist[ind == rows] = 0  # the tree search's exact 0; brute force rounds it
    order = np.argsort(dist, axis=1)
    ind, dist = ind[rows, order], dist[rows, order]

    # compute_connectivities_umap: the cell itself (and exact duplicates) dropped
    distances = sparse.coo_matrix(
        (dist.ravel(), (np.repeat(np.arange(n), m), ind.ravel())), shape=(n, n)
    )
    distances.eliminate_zeros()
    connectivities = fuzzy_simplicial_set(ind, dist, n, m)

    if trim is None:
        trim = 10 * m
    if trim > 0:
        connectivities = _trim(connectivities, trim)

    adata.uns["neighbors"] = dict(
        params=dict(
            n_neighbors=m,
            method="umap",
            metric=metric,
            n_pcs=n_pcs,
            bbknn=dict(trim=trim, computation="brute", batch_key=batch_key),
            use_rep=use_rep,
        ),
        distances_key="distances",
        connectivities_key="connectivities",
    )
    adata.obsp["distances"] = distances.tocsr()
    adata.obsp["connectivities"] = connectivities


def _trim(cnts: sparse.csr_matrix, trim: int) -> sparse.csr_matrix:
    """bbknn's ``trimming``: keep an entry if it is among the ``trim``
    strongest of both its row and its column (a threshold of 0 for a row with
    at most ``trim`` entries). bbknn's two passes, rows then the transposed
    columns, are one mask here."""
    n = cnts.shape[0]
    nnz = np.diff(cnts.indptr)
    rows = np.repeat(np.arange(n), nnz)
    desc = np.lexsort((-cnts.data, rows))  # each row's entries, strongest first
    thresh = np.zeros(n, cnts.data.dtype)
    big = nnz > trim
    thresh[big] = cnts.data[desc[cnts.indptr[:-1][big] + trim - 1]]
    keep = (cnts.data >= thresh[rows]) & (cnts.data >= thresh[cnts.indices])
    cnts = cnts.copy()
    cnts.data[~keep] = 0
    cnts.eliminate_zeros()
    return cnts
