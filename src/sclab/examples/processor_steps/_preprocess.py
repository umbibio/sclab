import warnings

import numpy as np
from anndata import ImplicitModificationWarning
from ipywidgets import Checkbox, Dropdown
from tqdm.auto import tqdm

import sclab.preprocess
from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class Preprocess(ProcessorStepBase):
    parent: Processor
    name: str = "preprocess"
    description: str = "Preprocess"

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

        cat_metadata = parent.dataset._metadata.select_dtypes(
            include=["object", "category"]
        )
        cat_options = {"": None, **{c: c for c in cat_metadata.columns}}

        variable_controls = dict(
            layer=Dropdown(
                options=tuple(parent.dataset.adata.layers.keys()),
                value="counts",
                description="Layer",
            ),
            group_by=Dropdown(
                options=cat_options,
                value=None,
                description="GroupBy",
            ),
            regress_total_counts=Checkbox(description="Regr. out total counts"),
            regress_n_genes=Checkbox(description="Regr. out n genes"),
            normalization_method=Dropdown(
                options=("No normalization", "Library size", "Weighted library size"),
                value="Library size",
                description="Norm. method",
            ),
            log1p=Checkbox(value=True, description="Log1p"),
            scale=Checkbox(value=True, description="Scale"),
        )

        def filter_layers(change: dict):
            new_options: tuple[str] = change["new"]
            if any(s.endswith("log1p") for s in new_options):
                new_options = tuple(
                    filter(lambda y: not y.endswith("log1p"), new_options)
                )
                variable_controls["layer"].options = new_options

        variable_controls["layer"].observe(filter_layers, "options", "change")

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(
        self,
        layer: str | None = None,
        group_by: str | None = None,
        regress_total_counts: bool = False,
        regress_n_genes: bool = False,
        normalization_method: str = "No normalization",
        log1p: bool = True,
        scale: bool = True,
    ):
        import scanpy as sc

        self.output.clear_output(wait=True)
        with self.output:
            pbar = tqdm(total=100, bar_format="{percentage:3.0f}%|{bar}|")

        dataset = self.parent.dataset
        adata = dataset.adata
        if layer is None:
            layer = dataset.counts_layer

        if f"{layer}_log1p" not in adata.layers:
            adata.layers[f"{layer}_log1p"] = sc.pp.log1p(adata.layers[layer].copy())
        pbar.update(10)

        if layer != "X":
            adata.X = adata.layers[layer].copy()
        start_n_cells_i, start_n_genes = adata.shape

        sc.pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        sc.pp.filter_cells(adata, min_genes=5)
        sc.pp.filter_genes(adata, min_cells=5)
        pbar.update(10)

        sc.pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        pbar.update(10)

        if group_by is not None:
            adata.var["highly_variable"] = False
            for name, idx in adata.obs.groupby(group_by, observed=True).groups.items():
                hvg_seurat = sc.pp.highly_variable_genes(
                    adata[idx],
                    layer=f"{layer}_log1p",
                    flavor="seurat",
                    inplace=False,
                )["highly_variable"]

                hvg_seurat_v3 = sc.pp.highly_variable_genes(
                    adata[idx],
                    layer=layer,
                    flavor="seurat_v3_paper",
                    n_top_genes=hvg_seurat.sum(),
                    inplace=False,
                )["highly_variable"]

                adata.var[f"highly_variable_{name}"] = hvg_seurat | hvg_seurat_v3
                adata.var["highly_variable"] |= adata.var[f"highly_variable_{name}"]

        else:
            sc.pp.highly_variable_genes(adata, layer=f"{layer}_log1p", flavor="seurat")
            hvg_seurat = adata.var["highly_variable"]

            sc.pp.highly_variable_genes(
                adata,
                layer=layer,
                flavor="seurat_v3_paper",
                n_top_genes=hvg_seurat.sum(),
            )
            hvg_seurat_v3 = adata.var["highly_variable"]

            adata.var["highly_variable"] = hvg_seurat | hvg_seurat_v3

        pbar.update(10)
        pbar.update(10)

        new_layer = layer
        if normalization_method == "Library size":
            new_layer += "_normt"
            sc.pp.normalize_total(adata, target_sum=1e4)
        elif normalization_method == "Weighted library size":
            new_layer += "_normw"
            sclab.preprocess.normalize_weighted(
                adata, target_scale=1e4, batch_key=group_by
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

        self.broker.publish("dset_metadata_change", dataset.metadata)
        self.broker.publish("dset_data_dict_change", dataset.data_dict, "metadata")
        self.broker.publish("dset_anndata_layers_change", dataset.adata.layers.keys())
        self.broker.publish(
            "ctrl_value_change_request",
            data_key="metadata",
            selected_axes_1="n_genes_by_counts",
            selected_axes_2="total_counts",
        )
        self.broker.publish("dset_var_dataframe_change", dataset.adata.var)

        end_n_cells_i, end_n_genes = adata.shape
        if start_n_cells_i != end_n_cells_i:
            self.broker.publish("dset_total_rows_change", dataset.metadata)
        if start_n_genes != end_n_genes:
            self.broker.publish("dset_total_vars_change", adata.var)

        adata.X = adata.X.astype(np.float32)

        pbar.close()
