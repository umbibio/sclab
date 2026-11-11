from anndata import AnnData

from .._citations import record_citations
from ._hvg import highly_variable_genes, hvg_citations
from ._scanpy_port import get_pp, resolve_backend


def pca(
    adata: AnnData,
    layer: str | None = None,
    n_comps: int = 30,
    mask_var: str | None = None,
    batch_key: str | None = None,
    reference_batch: str | None = None,
    zero_center: bool = False,
    counts_layer: str = "counts",
    random_state: int = 0,
    backend: str = "auto",
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
    counts_layer : str, optional
        Layer with raw counts. Only used when ``reference_batch`` is set and
        ``mask_var="highly_variable"``: highly variable genes are then
        recomputed on the reference cells with
        :func:`sclab.preprocess.highly_variable_genes` (its defaults: the
        union of the ``seurat`` flavor, on ``log1p`` of the counts, taken from
        the layer ``"{counts_layer}_log1p"`` if present (older sclab stored
        it), else computed on the fly for the reference cells only and not
        kept, and the ``seurat_v3_paper`` flavor, on the counts). Raises
        ``ValueError`` if the layer is missing. ``layer`` remains the input to
        PCA. Default is ``"counts"``.
    random_state : int, optional
        Seed of the ARPACK starting vector. Default is 0.
    backend : {"auto", "native", "scanpy"}, optional
        ``"scanpy"`` runs :func:`scanpy.pp.pca`, ``"native"`` sclab's port of
        it (the same :class:`sklearn.decomposition.PCA` fit, ARPACK solver;
        sparse input is centered implicitly, not densified), ``"auto"``
        (default) scanpy if it is installed. The results are the same.

    Returns
    -------
    None
        Modifies ``adata`` in-place, storing results in:

        - ``adata.obsm["X_pca"]`` — PC coordinates for all cells.
        - ``adata.varm["PCs"]`` — loadings matrix.
        - ``adata.uns["pca"]`` — variance and variance-ratio arrays; also
          stores ``"reference_batch"`` when fitted on a reference batch.

    References
    ----------
    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0

    Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
    *Journal of Machine Learning Research*, 12, 2825–2830.
    https://jmlr.org/papers/v12/pedregosa11a.html

    Lehoucq, Sorensen & Yang (1998). ARPACK Users' Guide. SIAM.
    https://doi.org/10.1137/1.9780898719628

    Satija et al. (2015). Spatial reconstruction of single-cell gene
    expression data. *Nature Biotechnology*, 33, 495–502.
    https://doi.org/10.1038/nbt.3192

    Stuart et al. (2019). Comprehensive Integration of Single-Cell Data.
    *Cell*, 177, 1888–1902.e21.
    https://doi.org/10.1016/j.cell.2019.05.031
    """
    backend = resolve_backend(backend)
    pp = get_pp(backend)

    pca_kwargs = dict(
        n_comps=n_comps,
        layer=layer,
        mask_var=mask_var,
        svd_solver="arpack",
        random_state=random_state,
    )

    X = adata.X if layer is None else adata.layers[layer]

    if reference_batch:
        obs_mask = adata.obs[batch_key] == reference_batch
        adata_ref = adata[obs_mask].copy()
        if mask_var == "highly_variable":
            if counts_layer not in adata_ref.layers:
                raise ValueError(
                    f"counts_layer {counts_layer!r} not found in adata.layers; it is "
                    "required to recompute highly variable genes on the reference batch."
                )
            highly_variable_genes(adata_ref, counts_layer=counts_layer, backend=backend)

        pp.pca(adata_ref, **pca_kwargs)
        uns_pca = adata_ref.uns["pca"]
        uns_pca["reference_batch"] = reference_batch
        PCs = adata_ref.varm["PCs"]
        adata.obsm["X_pca"] = X.dot(PCs)
        adata.uns["pca"] = uns_pca
        adata.varm["PCs"] = PCs
    else:
        pp.pca(adata, **pca_kwargs)
        adata.obsm["X_pca"] = X.dot(adata.varm["PCs"])

    if zero_center:
        adata.obsm["X_pca"] -= adata.obsm["X_pca"].mean(axis=0, keepdims=True)

    hvg = reference_batch and mask_var == "highly_variable"
    hvg_keys = hvg_citations(("seurat", "seurat_v3_paper")) if hvg else ()
    record_citations(adata, "scanpy", "scikit-learn", "arpack", *hvg_keys)
