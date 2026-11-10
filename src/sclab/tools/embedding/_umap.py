from anndata import AnnData

from ..._citations import cites


@cites("scanpy", "umap", "umap-learn")
def umap(
    adata: AnnData,
    *,
    n_components: int = 2,
    zero_center: bool = False,
    random_state: int = 0,
) -> str:
    """UMAP embedding stored as ``adata.obsm[f"X_{n_components}Dumap"]``.

    Runs :func:`scanpy.tl.umap` on the precomputed neighbor graph and moves
    the result from ``adata.obsm["X_umap"]`` to the key above, so 2D and 3D
    embeddings can coexist.

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
        Seed for :func:`scanpy.tl.umap`. Default is 0.

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
    import scanpy as sc

    sc.tl.umap(adata, n_components=n_components, random_state=random_state)
    X = adata.obsm.pop("X_umap")
    if zero_center:
        X = X - X.mean(axis=0, keepdims=True)

    key = f"X_{n_components}Dumap"
    adata.obsm[key] = X
    return key
