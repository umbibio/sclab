import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
from ipywidgets import Combobox, Dropdown

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.utils import layer_options, unscaled_layer, x_name, x_scaled

colorscales = list(
    filter(lambda s: "swatch" not in s and not s.startswith("_"), dir(pc.sequential))
)


class GeneExpression(ProcessorStepBase):
    """Interactive gene expression visualization step.

    Lets users search for a gene by name and immediately visualize its
    expression on the current plot. Expression can be shown on the 2D/3D
    scatter (colored by gene expression) or as a scatter plot against any
    numeric metadata column (e.g. pseudotime). The plot updates reactively
    whenever the gene, layer, axis, or color scale selection changes.

    The layer defaults to the log-normalized unscaled data once Preprocess
    ran (see :func:`sclab.utils.unscaled_layer`); scaled ``adata.X`` stays
    selectable.
    """

    parent: Processor
    plot_only = True  # see Processor.data_hash
    name: str = "gene_expression"
    description: str = "Gene Expression"

    run_button_description = "Plot Expression"

    @staticmethod
    def _gene_options(adata) -> tuple[str, ...]:
        gene_input_options = adata.var_names
        genes_df = adata.var
        info_cols = ["name", "symbol", "description"]
        for col in [
            c for c in genes_df.columns if any([s.lower() in c for s in info_cols])
        ]:
            new_info = genes_df[col].astype(str).where(genes_df[col].notna(), "")
            gene_input_options = gene_input_options + " - " + new_info

        return tuple(gene_input_options.to_list())

    @staticmethod
    def _default_layer(adata) -> str | None:
        """The log-normalized unscaled data, None before Preprocess."""
        known = x_name(adata) is not None or x_scaled(adata) is not None
        return unscaled_layer(adata) if known else None

    def __init__(self, parent: Processor) -> None:
        df = parent.dataset.metadata.select_dtypes("number")
        df = parent.dataset.metadata
        axis_key_options = {"": None, **{c: c for c in df.columns}}

        variable_controls = dict(
            gene_input=Combobox(
                placeholder="Type gene name",
                options=self._gene_options(parent.dataset.adata),
                description="Gene",
                ensure_option=True,
            ),
            layer=Dropdown(
                options=layer_options(parent.dataset.adata),
                value=self._default_layer(parent.dataset.adata),
                description="Layer",
            ),
            time_key=Dropdown(
                options=axis_key_options,
                value=None,
                description="Horiz. axis",
            ),
            colorscale=Dropdown(
                options=colorscales, value="Oryel", description="Col. scale"
            ),
        )

        self._default = variable_controls["layer"].value
        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

        send_plot = self.guarded(self.send_plot)
        for key in ("gene_input", "layer", "time_key", "colorscale"):
            self.variable_controls[key].observe(send_plot, names="value")

    def dset_anndata_layers_change_callback(self, *args, **kwargs):
        # after Processor's (it sets the options): follow the default unless chosen
        control = self.variable_controls["layer"]
        default = self._default_layer(self.parent.dataset.adata)
        if control.value in (None, self._default) and default != control.value:
            control.value = default
        self._default = default

    def validate(self, params: dict) -> str | None:
        if not params.get("gene_input"):
            return "Choose a gene"
        if not params.get("layer"):
            return "Choose a layer"
        return None

    def dset_total_vars_change_callback(self, *args, **kwargs):
        control = self.variable_controls["gene_input"]
        current_value = control.value
        control.options = self._gene_options(self.parent.dataset.adata)
        if current_value in control.options:
            control.value = current_value
        else:
            control.value = ""

    def to_code(self, params: dict) -> None:
        return None  # plot only: the data is unchanged

    def function(self, *pargs, **kwargs):
        self.send_plot({})

    def send_plot(self, change: dict):
        adata = self.parent.dataset.adata
        metadata = self.parent.dataset.metadata

        gene_input: str = self.variable_controls["gene_input"].value
        layer: str = self.variable_controls["layer"].value
        time_key: str = self.variable_controls["time_key"].value
        colorscale: str = self.variable_controls["colorscale"].value

        if gene_input is None or gene_input == "":
            self.update_output("")
            return

        if layer is None or layer == "":
            self.update_output("")
            return

        gene_id = gene_input.split(" ")[0]

        if layer == "X":
            X = adata[:, gene_id].X
        else:
            X = adata[:, gene_id].layers[layer]

        E = np.asarray(X.sum(axis=1)).flatten()

        self.update_output(f"Showing gene: {gene_id}")
        # self.variable_controls["gene_input"].value = ""

        df = pd.DataFrame({gene_id: E}, index=adata.obs.index)
        # keep all cells: the plotter highlights (2D) or filters (1D/3D) the selection
        metadata = metadata.join(df)

        if time_key is None:
            gene = {"gene": gene_id, "layer": layer, "colorscale": colorscale}
            self.parent.plotter.state["gene"] = gene  # what pinned views restore
            self.broker.publish(
                "dplt_plot_figure_request",
                metadata=metadata,
                colorby=gene_id,
                color_continuous_scale=colorscale,
            )
            return

        fig = px.scatter(
            metadata,
            x=time_key,
            y=gene_id,
            color=gene_id,
            color_continuous_scale=colorscale,
            hover_name=metadata.index,
            title=f"Gene: {gene_id}, Layer: {self.variable_controls['layer'].label}",
        )
        self.broker.publish("dplt_plot_figure_request", figure=fig)
