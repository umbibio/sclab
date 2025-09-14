"""Use harmony to integrate cells from different experiments.

Note: code adapted from scanpy to use a custom version of harmonypy

Harmony:
Korsunsky, I., Millard, N., Fan, J. et al. Fast, sensitive and accurate integration of single-cell data with Harmony.
Nat Methods 16, 1289-1296 (2019). https://doi.org/10.1038/s41592-019-0619-0

Scanpy:
Wolf, F., Angerer, P. & Theis, F. SCANPY: large-scale single-cell gene expression data analysis.
Genome Biol 19, 15 (2018). https://doi.org/10.1186/s13059-017-1382-0

Scverse:
Virshup, I., Bredikhin, D., Heumos, L. et al. The scverse project provides a computational ecosystem for single-cell omics data analysis.
Nat Biotechnol 41, 604-606 (2023). https://doi.org/10.1038/s41587-023-01733-8
"""

from collections.abc import Sequence

import numpy as np
from anndata import AnnData

from ._harmony import run_harmony


def harmony_integrate(
    adata: AnnData,
    key: str | Sequence[str],
    *,
    basis: str = "X_pca",
    adjusted_basis: str | None = None,
    reference_batch: str | list[str] | None = None,
    **kwargs,
):
    """Use harmonypy :cite:p:`Korsunsky2019` to integrate different experiments."""

    if adjusted_basis is None:
        adjusted_basis = f"{basis}_harmony"

    if isinstance(reference_batch, str):
        reference_batch = [reference_batch]

    if reference_batch is not None:
        reference_values = np.zeros(adata.n_obs, dtype=bool)
        for batch in reference_batch:
            reference_values |= adata.obs[key].values == batch
        kwargs["reference_values"] = reference_values

    X = adata.obsm[basis].astype(np.float64)

    harmony_out = run_harmony(X, adata.obs, key, **kwargs)

    adata.obsm[adjusted_basis] = harmony_out.Z_corr.T
