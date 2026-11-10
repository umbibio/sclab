import pandas as pd
import plotly.express as px
from ipywidgets import Button, Checkbox, Dropdown, IntText

from sclab._compat import missing, requirement
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.preprocess import pca
from sclab.preprocess._scanpy_port import native_backend, resolve_backend
from sclab.utils import looks_like_raw_counts


class PCA(ProcessorStepBase):
    """Principal component analysis step.

    Fits PCA on all cells or on a reference batch, then projects all cells
    onto the resulting principal components. Stores the embedding in
    ``adata.obsm["X_pca"]``. Optionally zero-centers the embedding. A
    "Plot Variance Ratio" button is enabled after the step runs to inspect
    the explained variance per component.

    Expects preprocessed (normalized) data in ``adata.X``; raises
    ``ValueError`` if ``adata.X`` looks like raw counts.

    Uses scanpy if it is installed, else sclab's native port of
    :func:`scanpy.pp.pca` (``backend="auto"``; the same scikit-learn ARPACK
    fit); the history records which ran.
    """

    parent: Processor
    name: str = "pca"
    description: str = "PCA"
    background = True
    _checked_X = None  # the adata.X last checked by validate, and the result
    _X_is_raw = False

    def __init__(self, parent: Processor) -> None:
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
        self.plot_variance_ratio_button.on_click(
            self.guarded(self.plot_variance_ratio_callback)
        )

        self.controls_list = [
            *self.variable_controls.values(),
            self.run_button,
            self.plot_variance_ratio_button,
            self.status,
            self.output,
        ]
        return super().make_controls()

    def validate(self, params: dict) -> str | None:
        hvg = params.get("reference_batch") and params.get("mask_var") == (
            "highly_variable"
        )
        if hvg and missing("skmisc"):  # HVGs recomputed with seurat_v3_paper
            return f"Reference batch HVGs need {requirement('skmisc')}"
        X = self.parent.dataset.adata.X
        if self._checked_X is not X:  # checked once per matrix: it reads all values
            self._checked_X, self._X_is_raw = X, looks_like_raw_counts(X)
        if self._X_is_raw:
            return "adata.X looks like raw counts. Run the Preprocess step first"
        return None

    def tool(self, params: dict) -> str:
        return "Python · scanpy.pp.pca"

    def resolved_params(self, params: dict) -> dict:
        return {
            **super().resolved_params(params),
            "batch_key": self.parent.batch_key,
            "counts_layer": self.parent.dataset.counts_layer,
            "backend": resolve_backend("auto"),
        }

    def to_code(self, params: dict) -> str:
        return call(
            "sclab.preprocess.pca",
            ADATA,
            n_comps=params["n_comps"],
            mask_var=params["mask_var"],
            batch_key=params["batch_key"],
            reference_batch=params["reference_batch"],
            zero_center=params["zero_center"],
            counts_layer=params["counts_layer"],
            random_state=params["random_state"],
            **native_backend(params),
        )

    def function(
        self,
        n_comps: int = 30,
        mask_var: str | None = None,
        reference_batch: str | None = None,
        zero_center: bool = False,
        random_state: int = 0,
    ):
        yield from self.compute(
            pca,
            n_comps=n_comps,
            mask_var=mask_var,
            batch_key=self.parent.batch_key,
            reference_batch=reference_batch,
            zero_center=zero_center,
            counts_layer=self.parent.dataset.counts_layer,
            random_state=random_state,
        )

        self.plot_variance_ratio_button.disabled = False
        self.broker.publish(
            "dset_data_dict_change", self.parent.dataset.data_dict, "X_pca"
        )

    def plot_variance_ratio_callback(self, *args, **kwargs):
        adata = self.parent.dataset.adata
        n = len(adata.uns["pca"]["variance_ratio"])

        df = pd.DataFrame(
            {k: adata.uns["pca"][k] for k in ["variance", "variance_ratio"]},
            index=pd.Index(range(n), name="component") + 1,
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
