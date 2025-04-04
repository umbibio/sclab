from ipywidgets import Dropdown, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class QC(ProcessorStepBase):
    parent: Processor
    name: str = "qc"
    description: str = "QC"

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

        variable_controls = dict(
            layer=Dropdown(
                options=tuple(parent.dataset.adata.layers.keys()),
                value="counts",
                description="Layer",
            ),
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

    def compute_qc_metrics(
        self,
        layer: str | None = None,
        min_genes: int = 5,
        min_cells: int = 5,
    ):
        import scanpy as sc

        if layer is None:
            layer = self.parent.dataset.counts_layer

        dataset = self.parent.dataset
        adata = dataset.adata

        adata.layers["qc_tmp_current_X"] = adata.X
        adata.X = adata.layers[layer].copy()
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        adata.obs["barcode_rank"] = adata.obs["total_counts"].rank(ascending=False)

        # Restore original X
        adata.X = adata.layers.pop("qc_tmp_current_X")

    def function(
        self,
        layer: str | None = None,
        min_genes: int = 5,
        min_cells: int = 5,
        max_rank: int = 0,
    ):
        self.compute_qc_metrics(layer, min_genes, min_cells)

        if max_rank > 0:
            series = self.parent.dataset.adata.obs["barcode_rank"]
            index = series.loc[series < max_rank].index
            self.parent.dataset.filter_rows(index)

        self.broker.publish("dset_metadata_change", self.parent.dataset.metadata)
        self.broker.publish(
            "ctrl_value_change_request",
            data_key="metadata",
            selected_axes_1="barcode_rank",
            selected_axes_2="total_counts",
            log_axes_2=True,
        )
        self.broker.publish("dset_var_dataframe_change", self.parent.dataset.adata.var)
