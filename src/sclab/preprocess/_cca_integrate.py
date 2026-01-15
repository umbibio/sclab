import logging
import os

import numpy as np
from anndata import AnnData

from ._cca import cca
from ._mcca import mcca_sample_space

logger = logging.getLogger(__name__)


N_CPUS = os.cpu_count()


def cca_integrate(
    adata: AnnData,
    key: str,
    *,
    basis: str = "X",
    adjusted_basis: str | None = None,
    reference_batch: str | list[str] | None = None,
    mask_var: str | None = None,
    n_components: int = 30,
    svd_solver: str = "randomized",
    normalize: bool = True,
    random_state: int | None = None,
):
    n_groups = adata.obs[key].nunique()
    if n_groups == 2:
        cca_integrate_pair(
            adata,
            key,
            adata.obs[key].unique()[0],
            adata.obs[key].unique()[1],
            basis=basis,
            adjusted_basis=adjusted_basis,
            mask_var=mask_var,
            n_components=n_components,
            svd_solver=svd_solver,
            normalize=normalize,
            random_state=random_state,
        )
    else:
        raise NotImplementedError


def cca_integrate_pair(
    adata: AnnData,
    key: str,
    group1: str,
    group2: str,
    *,
    basis: str | None = None,
    adjusted_basis: str | None = None,
    mask_var: str | None = None,
    n_components: int = 30,
    svd_solver: str = "randomized",
    normalize: bool = True,
    random_state: int | None = None,
):
    if basis is None:
        basis = "X"

    if adjusted_basis is None:
        adjusted_basis = basis + "_cca"

    if mask_var is not None:
        mask = adata.var[mask_var].values
    else:
        mask = np.ones(adata.n_vars, dtype=bool)

    Xs = {}
    groups = adata.obs.groupby(key, observed=True).groups
    for gr, idx in groups.items():
        Xs[gr] = _get_basis(adata[idx, mask], basis)

    Ys = {}
    Ys[group1], sigma, Ys[group2] = cca(
        Xs[group1],
        Xs[group2],
        n_components=n_components,
        svd_solver=svd_solver,
        normalize=normalize,
        random_state=random_state,
    )

    if (
        adjusted_basis not in adata.obsm
        or adata.obsm[adjusted_basis].shape[1] != n_components
    ):
        adata.obsm[adjusted_basis] = np.full((adata.n_obs, n_components), np.nan)

    if adjusted_basis not in adata.uns:
        adata.uns[adjusted_basis] = {}

    uns = adata.uns[adjusted_basis]
    uns[f"{group1}-{group2}"] = {"sigma": sigma}
    for gr, obs_names in groups.items():
        idx = adata.obs_names.get_indexer(obs_names)
        adata.obsm[adjusted_basis][idx] = Ys[gr]
        uns[gr] = Ys[gr]


def mcca_integrate(
    adata: AnnData,
    key: str,
    *,
    basis: str = "X",
    adjusted_basis: str | None = None,
    mask_var: str | None = None,
    n_components: int = 30,
    normalize: bool = True,
    n_jobs: int = N_CPUS,
) -> None:
    """
    Multi-set CCA-style integration of multiple batches in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    key : str
        Column in `adata.obs` that defines groups / batches.
    basis : str
        Which matrix to use as input:
        - "X" (adata.X)
        - a layer name in adata.layers
        - a key in adata.obsm
    adjusted_basis : str or None
        Name of the integrated embedding in `adata.obsm`.
        If None, defaults to f"{basis}_mcca".
    mask_var : str or None
        Column in `adata.var` indicating which features to use (bool).
        If None, all features are used.
    n_components : int
        Number of canonical components.
    normalize : bool
        L2-normalize rows of the output embedding.
    n_jobs : int
        Number of parallel jobs for sparse cross-cov.
    """
    if basis is None:
        basis = "X"

    if adjusted_basis is None:
        adjusted_basis = basis + "_mcca"

    # feature mask
    if mask_var is not None:
        mask = adata.var[mask_var].values
    else:
        mask = np.ones(adata.n_vars, dtype=bool)

    # group cells by key
    groups = adata.obs.groupby(key, observed=True).groups
    if len(groups) < 2:
        raise ValueError("Need at least two groups for integration.")
    logger.info(f"Running MCCA on {len(groups)} groups: {list(groups.keys())}")

    group_names = list(groups.keys())
    Xs = []
    group_sizes = []
    group_obs_indices = []

    # collect per-group matrices in a fixed order
    for gr in group_names:
        obs_names = groups[gr]
        idx = adata.obs_names.get_indexer(obs_names)
        group_obs_indices.append(idx)
        group_sizes.append(len(idx))

        Xg = _get_basis(adata[idx, mask], basis)
        Xs.append(Xg)

    # run multi-set CCA in sample space
    Z_big, eigvals = mcca_sample_space(
        Xs,
        n_components=n_components,
        normalize=normalize,
        n_jobs=n_jobs,
    )
    k = Z_big.shape[1]

    # allocate global embedding for all cells
    if adjusted_basis not in adata.obsm or adata.obsm[adjusted_basis].shape[1] != k:
        adata.obsm[adjusted_basis] = np.full((adata.n_obs, k), np.nan, dtype=float)

    # write uns metadata
    if adjusted_basis not in adata.uns:
        adata.uns[adjusted_basis] = {}
    uns = adata.uns[adjusted_basis]
    uns["method"] = "mcca_sample_space"
    uns["groups"] = group_names
    uns["eigvals"] = eigvals

    # scatter Z_big back into original cell order
    offsets = np.cumsum([0] + group_sizes[:-1])
    for i, (gr, idx) in enumerate(zip(group_names, group_obs_indices)):
        start = offsets[i]
        end = start + group_sizes[i]
        Z_block = Z_big[start:end, :]
        adata.obsm[adjusted_basis][idx] = Z_block

    logger.info(
        f"MCCA integration stored in adata.obsm['{adjusted_basis}'] "
        f"with {k} components."
    )


def _get_basis(adata: AnnData, basis: str):
    if basis == "X":
        X = adata.X

    elif basis in adata.layers:
        X = adata.layers[basis]

    elif basis in adata.obsm:
        X = adata.obsm[basis]

    else:
        raise ValueError(f"Unknown basis {basis}")

    return X
