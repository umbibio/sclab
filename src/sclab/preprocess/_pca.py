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
