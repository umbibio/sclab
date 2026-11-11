from importlib.metadata import PackageNotFoundError, version

import numpy as np
from anndata import AnnData

from .._citations import record_citations
from .._compat import require


def neighbors(
    adata: AnnData,
    *,
    use_rep: str = "X_pca",
    dims: tuple[int, int] = (1, 10),
    n_neighbors: int = 20,
    metric: str = "euclidean",
    batch_key: str | None = None,
    random_state: int = 0,
):
    """Build the k-nearest-neighbor graph on a range of embedding dimensions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    use_rep : str, optional
        Key in ``adata.obsm`` of the embedding. Default is ``"X_pca"``.
    dims : tuple of int, optional
        First and last dimension to use, 1-based and inclusive. If the first
        is not 1, the selected dimensions are stored as
        ``adata.obsm[f"{use_rep}_trimmed"]`` and the graph is built on them.
        Default is ``(1, 10)``.
    n_neighbors : int, optional
        Number of neighbors (per batch with ``batch_key``). Default is 20.
    metric : str, optional
        Distance metric. Default is ``"euclidean"``.
    batch_key : str or None, optional
        If given, build a batch-balanced graph with
        :func:`scanpy.external.pp.bbknn` (exact neighbor search, no seed
        needed), else use :func:`scanpy.pp.neighbors`. Default is None.
    random_state : int, optional
        Seed for :func:`scanpy.pp.neighbors`. Default is 0.

    References
    ----------
    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0

    McInnes et al. (2018). UMAP: Uniform Manifold Approximation and
    Projection for Dimension Reduction. *arXiv*, 1802.03426.
    https://doi.org/10.48550/arXiv.1802.03426

    Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
    *Journal of Machine Learning Research*, 12, 2825–2830.
    https://jmlr.org/papers/v12/pedregosa11a.html

    Polański et al. (2020). BBKNN: fast batch alignment of single cell
    transcriptomes. *Bioinformatics*, 36, 964–965.
    https://doi.org/10.1093/bioinformatics/btz625
    """
    import scanpy as sc

    min_dim, max_dim = dims
    min_dim = min_dim - 1

    if min_dim > 0:
        adata.obsm[use_rep + "_trimmed"] = adata.obsm[use_rep][:, min_dim:max_dim]
        use_rep = use_rep + "_trimmed"
    n_dims = max_dim - min_dim

    if batch_key:
        require("bbknn")
        sc.external.pp.bbknn(
            adata,
            batch_key=batch_key,
            use_rep=use_rep,
            n_pcs=n_dims,
            use_annoy=False,
            metric=metric,
            neighbors_within_batch=n_neighbors,
        )
    else:
        rep = adata.obsm[use_rep]
        upcast = getattr(rep, "dtype", None) == np.float32 and _umap_rejects_float32()
        if upcast:  # the same values; restored below
            adata.obsm[use_rep] = rep.astype(np.float64)
        try:
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                use_rep=use_rep,
                n_pcs=n_dims,
                metric=metric,
                random_state=random_state,
            )
        finally:
            if upcast:
                adata.obsm[use_rep] = rep

    keys = ("bbknn",) if batch_key else ("umap", "scikit-learn")
    record_citations(adata, "scanpy", *keys)


def _umap_rejects_float32() -> bool:
    """True with umap-learn >= 0.6, whose ``fuzzy_simplicial_set`` requires
    C-contiguous kNN distances: scanpy (1.12.5, ``neighbors/_common.py``) slices
    them from the exact kNN matrix, a non-contiguous view that umap only copies
    when it is not already float32 (e.g. ``metric="cosine"`` on a float32
    representation). A float64 representation (the same values) avoids it;
    umap-learn < 0.6 is left alone so its results do not change."""
    try:
        major, minor = (int(v) for v in version("umap-learn").split(".")[:2])
    except (PackageNotFoundError, ValueError):
        return False
    return (major, minor) >= (0, 6)
