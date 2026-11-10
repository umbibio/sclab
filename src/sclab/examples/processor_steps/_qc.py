from ipywidgets import Dropdown, IntText

from sclab._compat import missing
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.preprocess import qc
from sclab.utils import layer_keys


class QC(ProcessorStepBase):
    """Quality control step for single-cell RNA-seq data.

    Runs :func:`sclab.preprocess.qc`: computes per-cell and per-gene QC
    metrics using the selected counts
    layer, applies minimum count/gene/cell filters, and optionally restricts
    analysis to the top-ranked cells. After running, the plotter is updated
    to show ``barcode_rank`` vs ``total_counts``.

    Requires ``scanpy`` to be installed.
    """

    parent: Processor
    name: str = "qc"
    description: str = "QC"

    def __init__(self, parent: Processor) -> None:
        variable_controls = dict(
            layer=Dropdown(
                options=tuple(layer_keys(parent.dataset.adata)),
                value=parent.dataset.counts_layer,
                description="Layer",
            ),
            min_counts=IntText(value=50, description="Min. Counts"),
            min_genes=IntText(value=5, description="Min. Genes"),
            min_cells=IntText(value=0, description="Min. Cells"),
            max_rank=IntText(value=0, description="Max. Rank"),
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

    def validate(self, params: dict) -> str | None:
        if reason := missing("scanpy"):
            return reason
        dataset = self.parent.dataset
        if (params.get("layer") or dataset.counts_layer) not in layer_keys(
            dataset.adata
        ):
            return "Choose a 'Layer' with raw counts"
        return None

    def tool(self, params: dict) -> str:
        return "Python · scanpy.pp: calculate_qc_metrics, filter_cells, filter_genes"

    def dset_counts_layer_change_callback(self, old: str, new: str):
        layer = self.variable_controls["layer"]
        if layer.value == old and new in layer.options:  # else chosen by the user
            layer.value = new

    def resolved_params(self, params: dict) -> dict:
        resolved = super().resolved_params(params)
        if resolved.get("layer") is None:
            resolved["layer"] = self.parent.dataset.counts_layer
        return resolved

    def to_code(self, params: dict) -> str:
        return call(
            "sclab.preprocess.qc",
            ADATA,
            counts_layer=params["layer"],
            min_counts=params["min_counts"],
            min_genes=params["min_genes"],
            min_cells=params["min_cells"],
            max_rank=params["max_rank"],
        )

    def function(
        self,
        layer: str | None = None,
        min_counts: int = 50,
        min_genes: int = 5,
        min_cells: int = 5,
        max_rank: int = 0,
    ):
        dataset = self.parent.dataset
        n_obs, n_vars = dataset.adata.shape

        qc(
            dataset.adata,
            counts_layer=layer or dataset.counts_layer,
            min_counts=min_counts,
            min_genes=min_genes,
            min_cells=min_cells,
            max_rank=max_rank,
        )

        if dataset.adata.n_obs != n_obs:
            self.broker.publish("dset_total_rows_change", dataset.metadata)
        if dataset.adata.n_vars != n_vars:
            self.broker.publish("dset_total_vars_change", dataset.adata.var)
        self.broker.publish("dset_metadata_change", self.parent.dataset.metadata)
        self.broker.publish(
            "ctrl_value_change_request",
            data_key="metadata",
            selected_axes_1="barcode_rank",
            selected_axes_2="total_counts",
            log_axes_2=True,
        )
        self.broker.publish("dset_var_dataframe_change", self.parent.dataset.adata.var)
