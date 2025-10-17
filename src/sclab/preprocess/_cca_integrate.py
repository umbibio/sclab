import numpy as np
from anndata import AnnData

from ._cca import cca


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
