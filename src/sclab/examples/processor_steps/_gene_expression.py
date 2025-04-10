import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
from ipywidgets import Combobox, Dropdown

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase

colorscales = list(
    filter(lambda s: "swatch" not in s and not s.startswith("_"), dir(pc.sequential))
)


class GeneExpression(ProcessorStepBase):
    parent: Processor
    name: str = "gene_expression"
    description: str = "Gene Expression"

    run_button_description = "Plot Expression"

    def __init__(self, parent: Processor) -> None:
        df = parent.dataset.metadata.select_dtypes("number")
        df = parent.dataset.metadata
        axis_key_options = {"": None, **{c: c for c in df.columns}}

        gene_input_options = parent.dataset.adata.var_names
        genes_df = parent.dataset.adata.var
        info_cols = ["name", "symbol", "description"]
        for col in [
            c for c in genes_df.columns if any([s.lower() in c for s in info_cols])
        ]:
            new_info = genes_df[col].astype(str).str.replace("nan", "")
            gene_input_options = gene_input_options + " - " + new_info

        variable_controls = dict(
            gene_input=Combobox(
                placeholder="Type gene name",
                options=gene_input_options.to_list(),
                description="Gene",
                ensure_option=True,
            ),
            layer=Dropdown(
                options=tuple(parent.dataset.adata.layers.keys()),
                value=None,
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

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

        self.variable_controls["gene_input"].observe(self.send_plot, names="value")
        self.variable_controls["layer"].observe(self.send_plot, names="value")
        self.variable_controls["time_key"].observe(self.send_plot, names="value")
        self.variable_controls["colorscale"].observe(self.send_plot, names="value")

    def function(self, *pargs, **kwargs):
        self.send_plot({})

    def send_plot(self, change: dict):
        adata = self.parent.dataset.adata
        metadata = self.parent.dataset.metadata
        selected_cells = self.parent.dataset.selected_rows

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
        metadata = metadata.join(df)
        if selected_cells.size > 0:
            metadata = metadata.loc[selected_cells]

        if time_key is None:
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
            hover_name=adata.obs.index,
            title=f"Gene: {gene_id}, Layer: {layer}",
        )
        self.broker.publish("dplt_plot_figure_request", figure=fig)
