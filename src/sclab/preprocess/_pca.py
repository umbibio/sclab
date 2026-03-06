from anndata import AnnData


def pca(
    adata: AnnData,
    layer: str | None = None,
    n_comps: int = 30,
    mask_var: str | None = None,
    batch_key: str | None = None,
    reference_batch: str | None = None,
    zero_center: bool = False,
):
    """Compute principal components and project all cells onto the PCA space.

    When ``reference_batch`` is provided, PCA is fitted on the reference
    batch only and all cells are projected onto those principal components.
    This prevents the PC axes from being dominated by batch effects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    layer : str or None, optional
        Layer to use as input to PCA. Uses ``adata.X`` if None. Default is
        None.
    n_comps : int, optional
        Number of principal components to compute. Default is 30.
    mask_var : str or None, optional
        Boolean column in ``adata.var`` used to select a gene subset for
        PCA (e.g. ``"highly_variable"``). Default is None (use all genes).
    batch_key : str or None, optional
        Column in ``adata.obs`` identifying batches. Required when
        ``reference_batch`` is set. Default is None.
    reference_batch : str or None, optional
        Batch value in ``adata.obs[batch_key]`` to use for fitting the PCA
        model. All cells are then projected onto the reference PCs. Default
        is None (fit PCA on all cells).
    zero_center : bool, optional
        If True, subtract the mean of the PC coordinates so that the
        embedding is centred at the origin. Default is False.

    Returns
    -------
    None
        Modifies ``adata`` in-place, storing results in:

        - ``adata.obsm["X_pca"]`` — PC coordinates for all cells.
        - ``adata.varm["PCs"]`` — loadings matrix.
        - ``adata.uns["pca"]`` — variance and variance-ratio arrays; also
          stores ``"reference_batch"`` when fitted on a reference batch.
    """
    import scanpy as sc

    pca_kwargs = dict(
        n_comps=n_comps,
        layer=layer,
        mask_var=mask_var,
        svd_solver="arpack",
    )

    if reference_batch:
        obs_mask = adata.obs[batch_key] == reference_batch
        adata_ref = adata[obs_mask].copy()
        if mask_var == "highly_variable":
            sc.pp.highly_variable_genes(
                adata_ref, layer=f"{layer if layer else 'X'}_log1p", flavor="seurat"
            )
            hvg_seurat = adata_ref.var["highly_variable"]
            sc.pp.highly_variable_genes(
                adata_ref,
                layer=layer,
                flavor="seurat_v3_paper",
                n_top_genes=hvg_seurat.sum(),
            )
            hvg_seurat_v3 = adata_ref.var["highly_variable"]
            adata_ref.var["highly_variable"] = hvg_seurat | hvg_seurat_v3

        sc.pp.pca(adata_ref, **pca_kwargs)
        uns_pca = adata_ref.uns["pca"]
        uns_pca["reference_batch"] = reference_batch
        PCs = adata_ref.varm["PCs"]
        adata.obsm["X_pca"] = adata.X.dot(PCs)
        adata.uns["pca"] = uns_pca
        adata.varm["PCs"] = PCs
    else:
        sc.pp.pca(adata, **pca_kwargs)
        adata.obsm["X_pca"] = adata.X.dot(adata.varm["PCs"])

    if zero_center:
        adata.obsm["X_pca"] -= adata.obsm["X_pca"].mean(axis=0, keepdims=True)
