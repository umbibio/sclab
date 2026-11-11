from anndata import AnnData

from ..._citations import cites
from ...preprocess._scanpy_port import resolve_backend


@cites("scanpy", "umap", "umap-learn")
def umap(
    adata: AnnData,
    *,
    n_components: int = 2,
    zero_center: bool = False,
    random_state: int = 0,
    backend: str = "auto",
) -> str:
    """UMAP embedding stored as ``adata.obsm[f"X_{n_components}Dumap"]``.

    Runs :func:`scanpy.tl.umap` (or sclab's port of it, see ``backend``) on
    the precomputed neighbor graph (``adata.obsp["connectivities"]``,
    ``adata.uns["neighbors"]``) with scanpy's defaults and moves the result
    from ``adata.obsm["X_umap"]`` to the key above, so 2D and 3D embeddings
    can coexist.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a neighbor graph. Modified in-place.
    n_components : int, optional
        Number of dimensions. Default is 2.
    zero_center : bool, optional
        If True, subtract the mean so the embedding is centred at the origin.
        Default is False.
    random_state : int, optional
        Seed of the layout. Default is 0.
    backend : {"auto", "native", "scanpy"}, optional
        ``"scanpy"`` runs :func:`scanpy.tl.umap` (umap-learn), ``"native"``
        sclab's numpy/scipy port of it (umap-learn's
        ``simplicial_set_embedding``: the same spectral initialization, edge
        sampling schedule and gradients, but each epoch's SGD updates are
        applied as one vectorized step, so the layout differs from
        umap-learn's in detail; see :mod:`sclab.tools.embedding._umap_port`),
        ``"auto"`` (default) scanpy if it is installed.

    Returns
    -------
    str
        The ``adata.obsm`` key of the embedding.

    References
    ----------
    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0

    McInnes et al. (2018). UMAP: Uniform Manifold Approximation and
    Projection for Dimension Reduction. *arXiv*, 1802.03426.
    https://doi.org/10.48550/arXiv.1802.03426

    McInnes et al. (2018). UMAP: Uniform Manifold Approximation and
    Projection. *Journal of Open Source Software*, 3, 861.
    https://doi.org/10.21105/joss.00861
    """
    if resolve_backend(backend) == "scanpy":
        import scanpy as sc

        sc.tl.umap(adata, n_components=n_components, random_state=random_state)
    else:
        from ._umap_port import umap as umap_port

        umap_port(adata, n_components=n_components, random_state=random_state)
    X = adata.obsm.pop("X_umap")
    if zero_center:
        X = X - X.mean(axis=0, keepdims=True)

    key = f"X_{n_components}Dumap"
    adata.obsm[key] = X
    return key
