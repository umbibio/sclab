from anndata import AnnData


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
    """
    import scanpy as sc

    min_dim, max_dim = dims
    min_dim = min_dim - 1

    if min_dim > 0:
        adata.obsm[use_rep + "_trimmed"] = adata.obsm[use_rep][:, min_dim:max_dim]
        use_rep = use_rep + "_trimmed"
    n_dims = max_dim - min_dim

    if batch_key:
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
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            n_pcs=n_dims,
            metric=metric,
            random_state=random_state,
        )
