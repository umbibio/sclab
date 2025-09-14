import pandas as pd
import plotly.express as px
from ipywidgets import Button, Checkbox, Dropdown, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class PCA(ProcessorStepBase):
    parent: Processor
    name: str = "pca"
    description: str = "PCA"

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

        bool_var_df = parent.dataset.adata.var.select_dtypes(include=bool)
        mask_var_options = {
            "": None,
            **{col: col for col in bool_var_df.columns},
        }
        variable_controls = dict(
            n_comps=IntText(value=30, description="N comps."),
            mask_var=Dropdown(options=mask_var_options, description="Genes mask"),
            **parent.make_selectbatch_drowpdown(description="Reference Batch"),
            zero_center=Checkbox(value=False, description="Zero center"),
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def make_controls(self):
        self.plot_variance_ratio_button = Button(
            description="Plot Variance Ratio",
            layout={"width": "auto"},
            button_style="info",
            disabled=True,
        )
        self.plot_variance_ratio_button.on_click(self.plot_variance_ratio_callback)

        self.controls_list = [
            *self.variable_controls.values(),
            self.run_button,
            self.plot_variance_ratio_button,
            self.output,
        ]
        return super().make_controls()

    def function(
        self,
        n_comps: int = 30,
        mask_var: str | None = None,
        reference_batch: str | None = None,
        zero_center: bool = False,
    ):
        import scanpy as sc

        adata = self.parent.dataset.adata
        counts_layer = self.parent.dataset.counts_layer

        if reference_batch:
            batch_key = self.parent.batch_key

            obs_mask = adata.obs[batch_key] == reference_batch
            adata_ref = adata[obs_mask].copy()
            if mask_var == "highly_variable":
                sc.pp.highly_variable_genes(
                    adata_ref, layer=f"{counts_layer}_log1p", flavor="seurat"
                )
                hvg_seurat = adata_ref.var["highly_variable"]
                sc.pp.highly_variable_genes(
                    adata_ref,
                    layer=counts_layer,
                    flavor="seurat_v3_paper",
                    n_top_genes=hvg_seurat.sum(),
                )
                hvg_seurat_v3 = adata_ref.var["highly_variable"]
                adata_ref.var["highly_variable"] = hvg_seurat | hvg_seurat_v3
            sc.pp.pca(
                adata_ref, n_comps=n_comps, mask_var=mask_var, svd_solver="arpack"
            )
            uns_pca = adata_ref.uns["pca"]
            uns_pca["reference_batch"] = reference_batch
            PCs = adata_ref.varm["PCs"]
            adata.obsm["X_pca"] = adata.X.dot(PCs)
            adata.uns["pca"] = uns_pca
            adata.varm["PCs"] = PCs
        else:
            sc.pp.pca(adata, n_comps=n_comps, mask_var=mask_var, svd_solver="arpack")
            adata.obsm["X_pca"] = adata.X.dot(adata.varm["PCs"])

        if zero_center:
            adata.obsm["X_pca"] -= adata.obsm["X_pca"].mean(axis=0, keepdims=True)

        self.plot_variance_ratio_button.disabled = False
        self.broker.publish(
            "dset_data_dict_change", self.parent.dataset.data_dict, "X_pca"
        )

    def plot_variance_ratio_callback(self, *args, **kwargs):
        adata = self.parent.dataset.adata
        ncomps = self.variable_controls["n_comps"].value

        df = pd.DataFrame(
            {k: adata.uns["pca"][k] for k in ["variance", "variance_ratio"]},
            index=pd.Index(range(ncomps), name="component") + 1,
        )

        fig = px.scatter(df, y="variance_ratio")
        self.broker.publish("dplt_plot_figure_request", figure=fig)

    def dset_var_dataframe_change_callback(self, *args, **kwargs):
        var_df = self.parent.dataset.adata.var
        df = var_df.select_dtypes(include=["bool"])
        options = {"": None, **{c: c for c in df.columns}}

        control: Dropdown = self.variable_controls["mask_var"]
        current_value = control.value
        control.options = options
        if current_value not in control.options:
            control.value = None
        else:
            control.value = current_value
