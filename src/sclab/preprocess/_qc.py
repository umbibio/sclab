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
