import numpy as np
from anndata import AnnData


def qc(
    adata: AnnData,
    counts_layer: str = "counts",
    min_counts: int = 50,
    min_genes: int = 5,
    min_cells: int = 5,
    max_rank: int = 0,
):
    """Compute quality-control metrics and apply initial cell/gene filters.

    Temporarily sets ``adata.X`` to the counts layer to calculate QC metrics,
    then restores the original ``X``. Adds a ``barcode_rank`` column to
    ``adata.obs`` (rank by descending total counts).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    counts_layer : str, optional
        Layer containing raw counts. Created from ``adata.X`` if absent.
        Default is ``"counts"``.
    min_counts : int, optional
        Minimum total counts per cell. Cells below this threshold are
        removed before QC metrics are computed. Default is 50.
    min_genes : int, optional
        Minimum number of genes detected per cell. Default is 5.
    min_cells : int, optional
        Minimum number of cells a gene must be detected in. Default is 5.
    max_rank : int, optional
        If > 0, keep only cells with ``barcode_rank < max_rank`` (i.e. the
        top *max_rank* cells by total counts). Default is 0 (disabled).

    Returns
    -------
    None
        Modifies ``adata`` in-place. Adds QC columns to ``adata.obs`` and
        ``adata.var`` via :func:`scanpy.pp.calculate_qc_metrics`.
    """
    import scanpy as sc

    if counts_layer not in adata.layers:
        adata.layers[counts_layer] = adata.X.copy()

    adata.layers["qc_tmp_current_X"] = adata.X
    adata.X = adata.layers[counts_layer].copy()
    rowsums = np.asarray(adata.X.sum(axis=1)).squeeze()

    obs_idx = adata.obs_names[rowsums >= min_counts]
    adata._inplace_subset_obs(obs_idx)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata.obs["barcode_rank"] = adata.obs["total_counts"].rank(ascending=False)

    # Restore original X
    adata.X = adata.layers.pop("qc_tmp_current_X")

    if max_rank > 0:
        series = adata.obs["barcode_rank"]
        index = series.loc[series < max_rank].index
        adata._inplace_subset_obs(index)
