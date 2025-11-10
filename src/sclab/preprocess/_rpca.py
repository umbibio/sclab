import numpy as np
from anndata import AnnData
from numpy.typing import NDArray


def rpca(
    adata: AnnData,
    key: str,
    *,
    basis: str = "X",
    adjusted_basis: str | None = None,
    reference_batch: str | list[str] | None = None,
    mask_var: str | None = None,
    n_components: int = 30,
    min_variance_ratio: float = 0.0005,
    svd_solver: str = "arpack",
    normalize: bool = True,
):
    if basis is None:
        basis = "X"

    if adjusted_basis is None:
        adjusted_basis = basis + "_rpca"

    if mask_var is not None:
        mask = adata.var[mask_var].values
    else:
        mask = np.ones(adata.n_vars, dtype=bool)

    X = _get_basis(adata[:, mask], basis)
    uns = {}

    groups = adata.obs.groupby(key, observed=True).groups
    if reference_batch is None:
        reference_batch = list(groups.keys())
    elif isinstance(reference_batch, str):
        reference_batch = [reference_batch]

    for gr, idx in groups.items():
        if gr not in reference_batch:
            continue

        ref_basis_key = f"{adjusted_basis}_{gr}"
        ref_PCs_key = f"{adjusted_basis}_{gr}_PCs"

        X_reference = _get_basis(adata[idx, mask], basis)
        proj_result = pca_projection(
            X,
            X_reference,
            n_components=n_components,
            min_variance_ratio=min_variance_ratio,
            svd_solver=svd_solver,
            normalize=normalize,
        )
        res_ncomps = proj_result[0].shape[1]
        components = np.zeros((res_ncomps, adata.n_vars))
        components[:, mask] = proj_result[1]

        adata.obsm[ref_basis_key] = proj_result[0]
        adata.varm[ref_PCs_key] = components.T

        uns[gr] = {
            "n_components": res_ncomps,
            "explained_variance_ratio": proj_result[2],
            "explained_variance": proj_result[3],
        }

    adata.uns[adjusted_basis] = uns


def pca_projection(
    X: NDArray,
    X_reference: NDArray,
    n_components: int = 30,
    min_variance_ratio: float = 0.0005,
    svd_solver: str = "arpack",
    normalize: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    import scanpy as sc

    pca_kwargs = dict(
        n_comps=n_components,
        svd_solver=svd_solver,
        return_info=True,
    )

    pca_result = sc.pp.pca(X_reference, **pca_kwargs)
    _, components, explained_variance_ratio, explained_variance = pca_result

    components_mask = explained_variance_ratio > min_variance_ratio
    components = components[components_mask]
    explained_variance_ratio = explained_variance_ratio[components_mask]
    explained_variance = explained_variance[components_mask]

    X_pca = X.dot(components.T)

    if normalize:
        X_pca = X_pca / np.linalg.norm(X_pca, axis=1, keepdims=True)

    return X_pca, components, explained_variance_ratio, explained_variance


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
