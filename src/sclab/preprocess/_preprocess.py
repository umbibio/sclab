import warnings
from typing import Literal

import numpy as np
from anndata import AnnData, ImplicitModificationWarning
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
    weighted_norm_quantile: float = 0.9,
    log1p: bool = True,
    scale: bool = True,
):
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
                q=weighted_norm_quantile,
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
                for _, idx in adata.obs.groupby(group_by, observed=True).groups.items():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=ImplicitModificationWarning,
                            message="Modifying `X` on a view results in data being overridden",
                        )
                        adata[idx].X = sc.pp.scale(adata[idx].X, zero_center=False)
            else:
                sc.pp.scale(adata, zero_center=False)

        adata.layers[new_layer] = adata.X.copy()

        pbar.update(10)

        adata.X = adata.X.astype(np.float32)
