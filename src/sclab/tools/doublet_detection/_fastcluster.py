# Ported from scDblFinder (R/Bioconductor), R/clustering.R: `fastcluster`,
# `.prepSCE` and `.getDR`, version 1.20.2 (Bioconductor 3.20),
# https://github.com/plger/scDblFinder, GPL-3.
# Germain P-L, Lun A, Garcia Meixide C, Macnair W, Robinson MD. Doublet
# identification in single-cell sequencing data using scDblFinder.
# F1000Research 2021, 10:979. https://doi.org/10.12688/f1000research.73600.2
import random

import numpy as np
from scipy import sparse


def fastcluster(
    counts,
    ndims: int = 20,
    nfeatures: int = 1352,
    k: int | None = None,
    nstart: int = 3,
    iter_max: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """Fast two-step clustering of cells, as scDblFinder's ``clusters=TRUE``.

    Log-normalises the counts (library-size factors, ``log2(x / sf + 1)``),
    runs a PCA of the ``nfeatures`` most variable genes, groups cells into
    ``k`` k-means meta-cells (only above 1000 cells), builds a kNN graph of
    the meta-cells and clusters it with Louvain. Defaults are those
    ``scDblFinder`` uses when called with ``clusters=TRUE``.

    Parameters
    ----------
    counts : array or sparse matrix
        Raw counts, cells x genes.
    ndims : int, optional
        Number of principal components. Default is 20.
    nfeatures : int, optional
        Number of most variable genes used for the PCA. Default is 1352.
    k : int, optional
        Number of k-means meta-cells. Default is ``min(2500, n_cells // 10)``.
    nstart, iter_max : int, optional
        k-means restarts and maximum iterations. Defaults are 3 and 50.
    random_state : int, optional
        Seed for the PCA, k-means and Louvain steps.

    Returns
    -------
    numpy.ndarray
        Integer cluster labels, starting at 1 as in R.

    Notes
    -----
    k-means uses scikit-learn's Lloyd algorithm (R uses Hartigan-Wong) and
    the PCA uses scikit-learn (R uses irlba), so labels match R's
    ``fastcluster`` closely but not exactly.

    References
    ----------
    .. [1] Germain P-L, Lun A, Garcia Meixide C, Macnair W, Robinson MD.
       Doublet identification in single-cell sequencing data using
       scDblFinder. F1000Research 2021, 10:979.
       https://doi.org/10.12688/f1000research.73600.2
    .. [2] scDblFinder, ``R/clustering.R`` (``fastcluster``, ``.prepSCE``,
       ``.getDR``), version 1.20.2, https://github.com/plger/scDblFinder,
       GPL-3.
    """
    import igraph
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    X = sparse.csr_matrix(counts, dtype=np.float64)
    n = X.shape[0]

    # scuttle::logNormCounts: size factors are library sizes over their mean
    lib = np.asarray(X.sum(axis=1)).ravel()
    if np.any(lib <= 0):
        raise ValueError("fastcluster: some cells have no counts")
    X = sparse.diags(lib.mean() / lib) @ X
    X.data = np.log2(X.data + 1)

    # scater::runPCA(ntop=nfeatures): top genes by variance, centred, unscaled
    mean = np.asarray(X.mean(axis=0)).ravel()
    var = np.asarray(X.multiply(X).mean(axis=0)).ravel() - mean**2
    top = np.argsort(-var, kind="stable")[: min(nfeatures, X.shape[1])]
    Xtop = X[:, top].toarray()
    ncomp = min(ndims, *Xtop.shape)
    x = PCA(n_components=ncomp, random_state=random_state).fit_transform(Xtop)

    if k is None:
        k = min(2500, n // 10)
    if n > 1000 and n > k:
        pre = KMeans(
            n_clusters=k, n_init=nstart, max_iter=iter_max, random_state=random_state
        ).fit_predict(x)
        pre = np.unique(pre, return_inverse=True)[1]
        size = np.bincount(pre)
        x = np.vstack([np.bincount(pre, weights=col) / size for col in x.T]).T
    else:
        pre = np.arange(n)

    # bluster::makeKNNGraph: undirected, unweighted kNN graph
    nk = x.shape[0]
    knn = min(max(2, int(np.floor(np.sqrt(nk))) - 1), 10, nk - 1)
    idx = NearestNeighbors(n_neighbors=knn).fit(x).kneighbors(return_distance=False)
    edges = np.column_stack([np.repeat(np.arange(nk), knn), idx.ravel()])
    graph = igraph.Graph(n=nk, edges=edges.tolist()).simplify()

    # igraph's Louvain draws from Python's `random`; seed it, then restore
    state = random.getstate()
    random.seed(random_state)
    try:
        membership = np.asarray(graph.community_multilevel().membership)
    finally:
        random.setstate(state)
    return membership[pre] + 1
