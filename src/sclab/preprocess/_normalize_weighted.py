import warnings

import numpy as np
from anndata import AnnData, ImplicitModificationWarning
from scipy.sparse import csr_matrix, issparse


def normalize_weighted(
    adata: AnnData,
    target_scale: float | None = None,
    batch_key: str | None = None,
) -> None:
    if batch_key is not None:
        for _, idx in adata.obs.groupby(batch_key, observed=True).groups.items():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=ImplicitModificationWarning,
                    message="Modifying `X` on a view results in data being overridden",
                )
                normalize_weighted(adata[idx], target_scale, None)

        return

    X: csr_matrix
    Y: csr_matrix
    Z: csr_matrix

    X = adata.X
    assert issparse(X)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero"
        )
        Y = X.multiply(1 / X.sum(axis=0))
    Y = Y.tocsr()
    Y.eliminate_zeros()
    Y.data = -Y.data * np.log(Y.data)
    entropy = Y.sum(axis=0)

    Z = X.multiply(entropy)
    Z = Z.tocsr()
    Z.eliminate_zeros()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero"
        )
        scale = Z.sum(axis=1)
        Z = Z.multiply(1 / scale)
    Z = Z.tocsr()

    if target_scale is None:
        target_scale = np.median(scale.A1[scale.A1 > 0])

    Z = Z * target_scale

    adata.X = Z

    return
