from typing import Literal

import numpy as np
from anndata import AnnData
from tqdm.auto import tqdm


def preprocess(
    adata: AnnData,
    counts_layer: str = "counts",
    group_by: str | None = None,
    min_cells: int = 5,
    min_genes: int = 5,
    compute_hvg: bool = True,
    regress_total_counts: bool = False,
    regress_n_genes: bool = False,
    normalization_method: Literal["library", "weighted", "none"] = "library",
    target_scale: float = 1e4,
    log1p: bool = True,
    scale: bool = True,
):
    """Normalize, transform, and scale single-cell RNA-seq count data.

    Applies a configurable preprocessing pipeline: optional filtering,
    highly-variable gene selection, normalization, log1p transformation,
    optional covariate regression, and per-group scaling. The resulting
    processed matrix is stored in a new named layer whose suffix encodes
    the applied steps (e.g. ``counts_normt_log1p_scale``).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    counts_layer : str, optional
        Layer containing raw counts. Default is ``"counts"``.
    group_by : str or None, optional
        Column in ``adata.obs`` for per-group HVG selection, normalization,
        and scaling. When set, batch-aware processing is applied. Default
        is None.
    min_cells : int, optional
        Minimum number of cells a gene must be detected in to be retained.
        Default is 5.
    min_genes : int, optional
        Minimum number of genes detected per cell to be retained. Default
        is 5.
    compute_hvg : bool, optional
        If True, compute highly variable genes (union of Seurat and Seurat
        v3 selections) and store the result in ``adata.var["highly_variable"]``.
        Default is True.
    regress_total_counts : bool, optional
        If True, regress out total counts (or log1p total counts if
        ``log1p=True``) per cell. Default is False.
    regress_n_genes : bool, optional
        If True, regress out the number of detected genes per cell.
        Default is False.
    normalization_method : {"library", "weighted", "none"}, optional
        Normalization strategy. ``"library"`` applies library-size
        normalization to ``target_scale`` counts; ``"weighted"`` applies
        entropy-weighted normalization; ``"none"`` skips normalization.
        Default is ``"library"``.
    target_scale : float, optional
        Target sum for library-size normalization (counts per cell after
        normalization). Default is 1e4.
    log1p : bool, optional
        If True, apply log(x + 1) transformation after normalization.
        Default is True.
    scale : bool, optional
        If True, scale each gene to unit variance (zero-center disabled).
        Applied per group when ``group_by`` is set. Default is True.

    Returns
    -------
    None
        Modifies ``adata`` in-place. Stores the processed matrix in a new
        layer and updates ``adata.X``.
    """
    import scanpy as sc

    from ._normalize_weighted import normalize_weighted

    with tqdm(total=100, bar_format="{percentage:3.0f}%|{bar}|") as pbar:
        if counts_layer not in adata.layers:
            adata.layers[counts_layer] = adata.X.copy()

        if f"{counts_layer}_log1p" not in adata.layers:
            adata.layers[f"{counts_layer}_log1p"] = sc.pp.log1p(
                adata.layers[counts_layer].copy()
            )
        pbar.update(10)

        adata.X = adata.layers[counts_layer].copy()
        sc.pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        pbar.update(10)

        sc.pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        pbar.update(10)

        if compute_hvg:
            if group_by is not None:
                adata.var["highly_variable"] = False
                for name, idx in adata.obs.groupby(
                    group_by, observed=True
                ).groups.items():
                    hvg_seurat = sc.pp.highly_variable_genes(
                        adata[idx],
                        layer=f"{counts_layer}_log1p",
                        flavor="seurat",
                        inplace=False,
                    )["highly_variable"]

                    hvg_seurat_v3 = sc.pp.highly_variable_genes(
                        adata[idx],
                        layer=counts_layer,
                        flavor="seurat_v3_paper",
                        n_top_genes=hvg_seurat.sum(),
                        inplace=False,
                    )["highly_variable"]

                    adata.var[f"highly_variable_{name}"] = hvg_seurat | hvg_seurat_v3
                    adata.var["highly_variable"] |= adata.var[f"highly_variable_{name}"]

            else:
                sc.pp.highly_variable_genes(
                    adata, layer=f"{counts_layer}_log1p", flavor="seurat"
                )
                hvg_seurat = adata.var["highly_variable"]

                sc.pp.highly_variable_genes(
                    adata,
                    layer=counts_layer,
                    flavor="seurat_v3_paper",
                    n_top_genes=hvg_seurat.sum(),
                )
                hvg_seurat_v3 = adata.var["highly_variable"]

                adata.var["highly_variable"] = hvg_seurat | hvg_seurat_v3

        pbar.update(10)
        pbar.update(10)

        new_layer = counts_layer
        if normalization_method == "library":
            new_layer += "_normt"
            sc.pp.normalize_total(adata, target_sum=target_scale)
        elif normalization_method == "weighted":
            new_layer += "_normw"
            normalize_weighted(
                adata,
                target_scale=target_scale,
                batch_key=group_by,
            )

        pbar.update(10)
        pbar.update(10)

        if log1p:
            new_layer += "_log1p"
            adata.uns.pop("log1p", None)
            sc.pp.log1p(adata)
        pbar.update(10)

        vars_to_regress = []
        if regress_n_genes:
            vars_to_regress.append("n_genes_by_counts")

        if regress_total_counts and log1p:
            adata.obs["log1p_total_counts"] = np.log1p(adata.obs["total_counts"])
            vars_to_regress.append("log1p_total_counts")
        elif regress_total_counts:
            vars_to_regress.append("total_counts")

        if vars_to_regress:
            new_layer += "_regr"
            sc.pp.regress_out(adata, keys=vars_to_regress, n_jobs=1)
        pbar.update(10)

        if scale:
            new_layer += "_scale"
            if group_by is not None:
                group_col = adata.obs[group_by].astype("str").fillna("_unassigned")
                for _, idx in group_col.groupby(
                    group_col, observed=True
                ).groups.items():
                    mask = adata.obs.index.isin(idx)
                    scaled = sc.pp.scale(adata[mask].X, zero_center=False)
                    adata.X[mask, :] = scaled
            else:
                sc.pp.scale(adata, zero_center=False)

        adata.layers[new_layer] = adata.X.copy()

        pbar.update(10)

        adata.X = adata.X.astype(np.float32)
