import warnings

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse


def normalize_weighted(
    adata: AnnData,
    target_scale: float | None = None,
    batch_key: str | None = None,
    # q: float = 0.99,
) -> None:
    """Normalize counts using entropy-weighted library-size normalization.

    Each gene's contribution to each cell's library size is weighted by the
    information-entropy of that gene's count distribution across cells. This
    up-weights ubiquitously expressed genes in the library-size calculation,
    so that normalization is driven primarily by housekeeping genes rather
    than informative ones. When ``batch_key`` is provided, normalization is
    applied independently within each batch so that cross-batch count
    differences do not confound the weights.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. ``adata.X`` must be a sparse count matrix.
        Modified in-place.
    target_scale : float or None, optional
        Target library size after normalization. If None, this is set to
        1e4 by default. Default is None.
    batch_key : str or None, optional
        Column in ``adata.obs`` identifying batches. Normalization is
        applied independently per batch when set. Default is None.

    Returns
    -------
    None
        Updates ``adata.X`` in-place with the normalized count matrix.
    """
    X = adata.X
    if issparse(X):
        normalize_fn = _normalize_weighted_sparse
    else:
        normalize_fn = _normalize_weighted_dense

    if batch_key is not None:
        for _, idx in adata.obs.groupby(batch_key, observed=True).groups.items():
            mask = adata.obs.index.isin(idx)
            adata.X[mask, :] = normalize_fn(X[mask, :], target_scale)
    else:
        adata.X = normalize_fn(X, target_scale)


def _normalize_weighted_sparse(X: csr_matrix, target_scale: float | None) -> csr_matrix:
    """Compute entropy-weighted normalization on a sparse matrix."""

    Y: csr_matrix
    Z: csr_matrix

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero"
        )
        Y = X.multiply(1 / X.sum(axis=0))

    Y = Y.tocsr()
    Y.eliminate_zeros()
    Y.data = -Y.data * np.log(Y.data)
    entropy = Y.sum(axis=0)

    Z = X.multiply(entropy).tocsr()
    Z.eliminate_zeros()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero"
        )
        scale = Z.sum(axis=1)
        scale_safe = scale.A1.copy()
        scale_safe[scale_safe == 0] = 1.0
        Z = X.multiply(1 / scale_safe.reshape(-1, 1)).tocsr()

    if target_scale is None:
        target_scale = 1e4

    return Z * target_scale


def _normalize_weighted_dense(X: np.ndarray, target_scale: float | None) -> np.ndarray:
    """Compute entropy-weighted normalization on a dense matrix."""
    # Gene-level probability distributions (cells × genes)
    gene_totals = X.sum(axis=0, keepdims=True)
    gene_totals[gene_totals == 0] = 1.0
    Y = X / gene_totals

    # Shannon entropy per gene
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero"
        )
        logY = np.log(Y)
    logY[~np.isfinite(logY)] = 0.0
    entropy = -(Y * logY).sum(axis=0, keepdims=True)

    # Entropy-weighted library size
    scale = (X * entropy).sum(axis=1, keepdims=True)
    scale[scale == 0] = 1.0

    Z = X / scale

    if target_scale is None:
        target_scale = 1e4

    return Z * target_scale
