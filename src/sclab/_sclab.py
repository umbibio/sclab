from pathlib import Path

from anndata import AnnData
from ipywidgets.widgets import GridBox, Layout, Tab

from ._io import read_adata
from .dataset import SCLabDataset
from .dataset.plotter import Plotter
from .dataset.processor import Processor
from .event import EventBroker


class SCLabDashboard(GridBox):
    broker: EventBroker

    def __init__(
        self,
        adata: AnnData | None = None,
        filepath: str | Path | None = None,
        name: str = "SCLab Dashboard",
        counts_layer: str = "counts",
        batch_key: str | None = None,
        copy: bool = True,
    ):
        if adata is None and filepath is None:
            raise ValueError("Either adata or filepath must be provided")

        if adata is None:
            adata = read_adata(filepath)

        self.broker = EventBroker()
        self.dataset = SCLabDataset(
            adata, name=name, counts_layer=counts_layer, copy=copy, broker=self.broker
        )
        self.plotter = Plotter(self.dataset)
        self.processor = Processor(
            self.dataset,
            self.plotter,
            batch_key=batch_key,
        )

        self.main_content = Tab(
            children=[
                self.plotter,
                self.dataset.obs_table,
                self.dataset.var_table,
                self.broker.logs_tab,
            ],
            titles=[
                "Main graph",
                "Observations",
                "Genes",
                "Logs",
            ],
        )

        super().__init__(
            [
                self.processor.main_accordion,
                self.main_content,
            ],
            layout=Layout(
                width="100%",
                grid_template_columns="350px auto",
                grid_template_areas=""" "processor plotter" """,
                border="0px solid black",
            ),
        )

    @property
    def ds(self):
        return self.dataset

    @property
    def pr(self):
        return self.processor

    @property
    def pl(self):
        return self.plotter
