from io import BytesIO
from pathlib import Path

from anndata import AnnData
from IPython.display import display
from ipywidgets.widgets import (
    Button,
    FileUpload,
    GridBox,
    HBox,
    Label,
    Layout,
    Output,
    Tab,
    Text,
    VBox,
)

from ._io import is_valid_url, load_adata_from_url, read_adata
from .dataset import SCLabDataset
from .dataset.plotter import Plotter
from .dataset.processor import Processor
from .event import EventBroker


class SCLabDashboard(GridBox):
    broker: EventBroker
    dataset: SCLabDataset
    plotter: Plotter
    processor: Processor
    main_content: Tab

    def __init__(
        self,
        adata_or_filepath_or_url: AnnData | str | None = None,
        name: str = "SCLab Dashboard",
        counts_layer: str = "counts",
        batch_key: str | None = None,
        copy: bool = False,
    ):
        if adata_or_filepath_or_url is None:
            adata = None

        elif isinstance(adata_or_filepath_or_url, AnnData):
            adata = adata_or_filepath_or_url

        elif is_valid_url(adata_or_filepath_or_url):
            url = adata_or_filepath_or_url
            adata = load_adata_from_url(url)

        elif isinstance(adata_or_filepath_or_url, str):
            filepath = adata_or_filepath_or_url
            adata = read_adata(filepath)

        self.name = name
        self.counts_layer = counts_layer
        self.batch_key = batch_key

        self.broker = EventBroker()

        self.dataset = None
        self.plotter = None
        self.processor = None
        self.main_content = None

        self.data_loader_layout = Layout(
            width="100%",
            height="500px",
            grid_template_columns="auto",
            grid_template_areas=""" "data_loader" """,
            border="0px solid black",
        )

        self.dashboard_layout = Layout(
            width="100%",
            height="100%",
            grid_template_columns="350px auto",
            grid_template_areas=""" "processor plotter" """,
            border="0px solid black",
        )

        self.data_loader = DataLoader(self)

        if adata is not None:
            self._load(adata, copy=copy)
        else:
            GridBox.__init__(self, [self.data_loader], layout=self.data_loader_layout)

    def _load(self, adata: AnnData, copy: bool = False):
        self.dataset = SCLabDataset(
            adata,
            name=self.name,
            counts_layer=self.counts_layer,
            copy=copy,
            broker=self.broker,
        )
        self.plotter = Plotter(self.dataset)
        self.processor = Processor(
            self.dataset,
            self.plotter,
            batch_key=self.batch_key,
        )

        self.main_content = Tab(
            children=[
                self.plotter,
                self.processor.results_panel,
                self.dataset.obs_table,
                self.dataset.var_table,
                self.broker.logs_tab,
            ],
            titles=[
                "Main graph",
                "Results",
                "Observations",
                "Genes",
                "Logs",
            ],
        )

        self.children = (
            self.processor.main_accordion,
            self.main_content,
        )
        self.layout = self.dashboard_layout

    @property
    def ds(self):
        return self.dataset

    @property
    def pr(self):
        return self.processor

    @property
    def pl(self):
        return self.plotter


class DataLoader(VBox):
    dashboard: SCLabDashboard
    adata: AnnData

    upload: FileUpload
    upload_info: Output
    upload_button: Button
    upload_row: HBox
    upload_row_label: Label

    url: Text
    load_button: Button
    url_row: HBox
    url_row_label: Label

    progress_output: Output
    continue_button: Button

    def __init__(self, dashboard: SCLabDashboard):
        self.dashboard = dashboard

        self.upload_row_label = Label("Load from file:", layout=Layout(width="120px"))
        self.upload = FileUpload(layout=Layout(width="200px"))
        self.upload_info = Output(layout=Layout(width="95%"))
        self.upload_button = Button(description="Load", layout=Layout(width="200px"))
        self.upload_row = HBox(
            [self.upload_row_label, self.upload, self.upload_info, self.upload_button],
            layout=Layout(width="100%"),
        )
        self.upload.observe(self.on_upload, "value")

        self.url_row_label = Label("Load from URL:", layout=Layout(width="120px"))
        self.url = Text(placeholder="https://...", layout=Layout(width="100%"))
        self.load_button = Button(description="Load", layout=Layout(width="200px"))
        self.url_row = HBox(
            [self.url_row_label, self.url, self.load_button],
            layout=Layout(width="100%"),
        )
        self.load_button.on_click(self.on_load_url)

        self.progress_output = Output(layout=Layout(width="95%"))
        self.continue_button = Button(
            description="Continue", layout=Layout(width="100%"), button_style="success"
        )
        self.continue_button.on_click(self.on_continue)

        VBox.__init__(
            self,
            [
                self.url_row,
                self.upload_row,
                self.progress_output,
            ],
            layout=Layout(width="100%"),
        )

    def on_upload(self, *args, **kwargs):
        from .scanpy.readwrite import read_10x_h5, read_h5ad

        files = self.upload.value
        if len(files) == 0:
            return

        file = files[0]

        self.upload_info.clear_output()
        with self.upload_info:
            for k, v in file.items():
                if k == "content":
                    continue
                print(f"{k}: {v}")

        filename = file["name"]
        contents = BytesIO(file["content"].tobytes())
        var_names = "gene_ids"

        path = Path(filename)

        match path.suffix:
            case ".h5":
                adata = read_10x_h5(contents)
            case ".h5ad":
                adata = read_h5ad(contents)
            case _:
                self.upload_info.clear_output()
                with self.upload_info:
                    print(f"`{filename}` is not valid")
                    print("Please upload a 10x h5 or h5ad file")
                return

        if var_names in adata.var:
            adata.var = adata.var.set_index(var_names)

        with self.progress_output:
            print(f"Loaded {adata.shape[0]} observations and {adata.shape[1]} genes\n")
            print(adata)
            display(self.continue_button)

        self.adata = adata

    def on_load_url(self, *args, **kwargs):
        self.progress_output.clear_output()
        with self.progress_output:
            self.adata = load_adata_from_url(self.url.value)
            display(self.continue_button)

    def on_continue(self, *args, **kwargs):
        self.dashboard._load(self.adata)
        self.adata = None
