import numpy as np
from anndata import AnnData
from scipy.stats import rankdata


def filter_obs(
    adata: AnnData,
    *,
    layer: str | None = None,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_counts: int | None = None,
    max_cells: int | None = None,
) -> None:
    """Filter observations (cells) based on count and gene-detection thresholds.

    All filtering criteria are applied simultaneously; cells that fail any
    active criterion are removed. Only criteria that are not None are
    applied.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    layer : str or None, optional
        Layer to use for count computations. Uses ``adata.X`` if None.
        Default is None.
    min_counts : int or None, optional
        Minimum total counts per cell. Cells with fewer counts are removed.
    min_genes : int or None, optional
        Minimum number of genes detected (count > 0) per cell.
    max_counts : int or None, optional
        Maximum total counts per cell. Cells with more counts are removed.
    max_cells : int or None, optional
        Maximum number of cells to retain, keeping those with the highest
        total counts (i.e. keep the top *max_cells* cells by total counts).

    Returns
    -------
    None
        Modifies ``adata`` in-place by subsetting observations.
    """
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    remove_mask = np.zeros(X.shape[0], dtype=bool)

    if min_genes is not None:
        M = X > 0
        rowsums = np.asarray(M.sum(axis=1)).squeeze()
        remove_mask[rowsums < min_genes] = True

    if min_counts is not None or max_counts is not None or max_cells is not None:
        rowsums = np.asarray(X.sum(axis=1)).squeeze()

        if min_counts is not None:
            remove_mask[rowsums < min_counts] = True

        if max_counts is not None:
            remove_mask[rowsums > max_counts] = True

        if max_cells is not None:
            ranks = rankdata(-rowsums, method="min")
            remove_mask[ranks > max_cells] = True

    if remove_mask.any():
        obs_idx = adata.obs_names[~remove_mask]
        adata._inplace_subset_obs(obs_idx)
