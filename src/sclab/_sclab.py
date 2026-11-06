import inspect
from collections.abc import Callable
from pathlib import Path

from anndata import AnnData
from IPython.display import display
from ipywidgets.widgets import (
    HTML,
    Button,
    Checkbox,
    Dropdown,
    FileUpload,
    GridBox,
    HBox,
    IntText,
    Label,
    Layout,
    Output,
    Tab,
    Text,
    ToggleButtons,
    VBox,
    Widget,
)
from traitlets import TraitError

from ._io import _read_adata_bytes, is_valid_url, load_adata_from_url, read_adata
from .dataset import SCLabDataset
from .dataset.plotter import Plotter
from .dataset.plotter._backends import BACKENDS
from .dataset.processor import Processor
from .dataset.processor._history import UNS_KEY
from .event import EventBroker
from .utils import (
    LazyOutput,
    layer_keys,
    set_options_keep_value,
    text_or_category_columns,
)

# set at build: the steps' controls are made from them
BUILD_SETTINGS = ("counts_layer", "batch_key", "plot_backend")


class SCLabDashboard(GridBox):
    """Interactive single-cell RNA-seq analysis dashboard for Jupyter notebooks.

    Integrates data loading, preprocessing, clustering, visualization, and
    analysis into a single tabbed widget. Built on top of AnnData/Scanpy with
    an event-driven architecture for reactive updates between components.

    Parameters
    ----------
    adata_or_filepath_or_url : AnnData or str or None, optional
        Input data as an AnnData object, a file path (.h5, .h5ad, or 10x MTX
        folder), or a URL to a remote .h5 or .h5ad file. If None, a data
        loader widget is displayed instead.
    name : str, optional
        Display name for the dashboard. Default is "SCLab Dashboard".
    counts_layer : str or None, optional
        Name of the layer to use (and create if absent) for raw counts.
        None: the stored setting, else "counts".
    batch_key : str or None, optional
        Key in ``adata.obs`` identifying batches/samples for batch-aware
        steps such as Integration and Neighbors; "" for none. None: the
        stored setting, else "batch" if there is such a column.
    copy : bool, optional
        If True, work on a copy of the input AnnData. Default is False.
    source : str or None, optional
        Path or URL the AnnData was read from, for the load line of the
        generated script. Set automatically when a path or URL is given.
    plot_backend : {"plotly", "jscatter", "static"} or None, optional
        Backend of the 2D plots, also settable later as
        ``dashboard.plotter.backend``. "jscatter" needs jupyter-scatter.
        1D and 3D plots always use plotly. None: the stored setting, else
        "plotly".

    Notes
    -----
    The Settings tab holds the counts layer, the batch key, background steps,
    the default random seed, the plot backend and appearance, and whether the
    tables stay loaded when hidden. Each change is stored in
    ``adata.uns["sclab"]["settings"]`` (a dict, saved with the h5ad) and read
    back when a dashboard is built on that data. Precedence: arguments given
    here, then the stored settings, then the defaults. A setting changes the
    defaults of the steps' controls, not values the user chose: a control is
    updated only while it holds the previous setting's value. The batch key
    controls of PCA and Neighbors exist only if the dashboard is built with
    a batch key.

    Attributes
    ----------
    ds : SCLabDataset
        Shorthand property for the dataset component.
    pl : Plotter
        Shorthand property for the plotter component.
    pr : Processor
        Shorthand property for the processor component.

    Examples
    --------
    >>> import scanpy as sc
    >>> from sclab import SCLabDashboard
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> dashboard = SCLabDashboard(adata, name="PBMC Analysis")
    >>> display(dashboard)
    """

    broker: EventBroker
    dataset: SCLabDataset
    plotter: Plotter
    processor: Processor
    main_content: Tab

    def __init__(
        self,
        adata_or_filepath_or_url: AnnData | str | None = None,
        name: str = "SCLab Dashboard",
        counts_layer: str | None = None,
        batch_key: str | None = None,
        copy: bool = False,
        source: str | None = None,
        plot_backend: str | None = None,
    ):
        self._preexisting = _live_widget_ids()  # not ours to close, see close()
        self.source = source  # path or URL, for the generated script's load line
        if adata_or_filepath_or_url is None:
            adata = None

        elif isinstance(adata_or_filepath_or_url, AnnData):
            adata = adata_or_filepath_or_url

        elif is_valid_url(adata_or_filepath_or_url):
            url = adata_or_filepath_or_url
            adata = load_adata_from_url(url)
            self.source = url

        elif isinstance(adata_or_filepath_or_url, str):
            filepath = adata_or_filepath_or_url
            adata = read_adata(filepath)
            self.source = filepath

        self.name = name
        self.counts_layer = counts_layer
        self.batch_key = batch_key
        self.plot_backend = plot_backend

        self.broker = EventBroker()

        self.dataset = None
        self.plotter = None
        self.processor = None
        self.main_content = None
        # widgets made while building, incl. replaced ones held only by their comm
        self._built: set[str] = set()

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
            overflow="auto",  # scroll to the plot settings if the plot hits its min width
        )

        self.data_loader = DataLoader(self._load_from_loader)

        GridBox.__init__(self)
        if adata is not None:
            self._load(adata, copy=copy)
        else:
            self.children = (self.data_loader,)
            self.layout = self.data_loader_layout
        self._built |= _live_widget_ids() - self._preexisting

    def _load_from_loader(self, adata: AnnData, source: str | None):
        self.source = source
        self._load(adata)

    def close(self):
        """Close the dashboard and every widget it holds (frees the frontend comms)."""
        _close_tree(self, self._preexisting)
        registry = _widget_registry()
        for model_id in self._built:
            if (widget := registry.get(model_id)) is not None:
                Widget.close(widget)

    def __del__(self):
        # Widget.__del__ calls close(): on GC and at interpreter shutdown close only
        # this widget, as any widget would. Its children may be displayed elsewhere,
        # and at shutdown the registry lookup fails (sys.meta_path is None).
        try:
            Widget.close(self)
        except Exception:
            pass

    def _load(self, adata: AnnData, copy: bool = False):
        before = _live_widget_ids()
        # the arguments, else the stored settings (if still valid), else defaults
        stored = dict(adata.uns.get(UNS_KEY, {}).get("settings", {}))
        counts_layer, batch_key, backend = (stored.get(k) for k in BUILD_SETTINGS)
        if self.counts_layer is not None or counts_layer not in layer_keys(adata):
            counts_layer = self.counts_layer or "counts"
        stored_key = batch_key == "" or batch_key in adata.obs  # "": none
        if self.batch_key is not None or not stored_key:
            batch_key = self.batch_key
        if self.plot_backend is not None or backend not in BACKENDS:
            backend = self.plot_backend or "plotly"

        self.dataset = SCLabDataset(
            adata,
            name=self.name,
            counts_layer=str(counts_layer),
            copy=copy,
            broker=self.broker,
        )
        self.plotter = Plotter(self.dataset, backend=str(backend))
        self.processor = Processor(
            self.dataset,
            self.plotter,
            batch_key=batch_key if batch_key is None else str(batch_key),
            source=self.source,
        )
        self._init_settings(stored)

        self.main_content = Tab(
            children=[
                self.plotter,
                self.processor.results_panel,
                self.dataset.obs_table,
                self.dataset.var_table,
                self.processor.history_panel,
                self.broker.logs_tab,
                self.settings,
            ],
            titles=[
                "Main graph",
                "Results",
                "Observations",
                "Genes",
                "History",
                "Logs",
                "Settings",
            ],
        )

        # heavy tables are built and sent only while their tab is open
        tab = self.main_content
        self.dataset.obs_table_output.show_when((tab, self.dataset.obs_table))
        self.dataset.var_table_output.show_when((tab, self.dataset.var_table))
        self.processor.results_panel.show_when((tab, self.processor.results_panel))

        self.children = (
            VBox([self.broker.jobs.panel, self.processor.main_accordion]),
            self.main_content,
        )
        self.layout = self.dashboard_layout
        self._built |= _live_widget_ids() - before

    def _init_settings(self, stored: dict):
        """The Settings tab, with the ``stored`` settings applied (see the class doc)."""
        dataset, processor, plotter = self.dataset, self.processor, self.plotter
        self.counts_layer_dropdown = Dropdown(
            options=self._counts_layer_options(),
            value=dataset.counts_layer,
            description="Counts layer",
            tooltip="Raw counts: the default layer of QC, Preprocess, PCA, "
            "Doublet Detection and pseudobulk DE",
        )
        self.batch_key_dropdown = Dropdown(
            options=self._batch_key_options(),
            value=processor.batch_key or "",
            description="Batch key",
            tooltip="Batches or samples, for Integration, PCA and Neighbors",
        )
        self.background_checkbox = Checkbox(
            value=True,
            description="Background steps",
            tooltip="Off: steps run on the main thread, as step.run() does",
        )
        self.seed_input = IntText(
            value=0,
            description="Random seed",
            tooltip="random_state of the steps that have one; 0: their defaults",
        )
        self.keep_tables_checkbox = Checkbox(
            value=False,
            description="Keep tables loaded when hidden",
            tooltip="Reopening a table is instant, but a page reload downloads "
            "every loaded table again",
        )
        self.counts_layer_dropdown.observe(
            lambda c: dataset.set_counts_layer(c["new"]), "value"
        )
        self.batch_key_dropdown.observe(
            lambda c: processor.set_batch_key(c["new"]), "value"
        )
        self.background_checkbox.observe(
            lambda c: setattr(self.broker.jobs, "threaded", c["new"]), "value"
        )
        self.seed_input.observe(self._set_seed, "value")
        self.keep_tables_checkbox.observe(self._set_keep_tables, "value")
        self.broker.subscribe(
            "dset_anndata_layers_change",
            lambda *_, **__: set_options_keep_value(
                self.counts_layer_dropdown, self._counts_layer_options()
            ),
        )
        self.broker.subscribe(
            "dset_metadata_change",
            lambda *_, **__: set_options_keep_value(
                self.batch_key_dropdown, self._batch_key_options()
            ),
        )

        widgets = self._settings_widgets()
        for widget in (
            self.counts_layer_dropdown,
            self.batch_key_dropdown,
            self.seed_input,
        ):
            widget.style = dict(description_width="100px")  # as the plot settings
        for checkbox in (self.background_checkbox, self.keep_tables_checkbox):
            checkbox.indent = False

        def section(title, *children):
            return VBox(
                [HTML(f"<b>{title}</b>"), *children], layout=Layout(padding="8px")
            )

        self.settings = VBox(
            [
                section("Data", self.counts_layer_dropdown, self.batch_key_dropdown),
                section("Steps", self.background_checkbox, self.seed_input),
                plotter.settings,
                section("Tables", self.keep_tables_checkbox),
            ]
        )

        with self.broker.batch():  # one redraw
            for key, value in stored.items():
                if key in widgets and key not in BUILD_SETTINGS:
                    widget = widgets[key]
                    try:  # h5ad gives numpy scalars
                        widget.value = type(widget.value)(value)
                    except (TraitError, TypeError, ValueError):
                        pass  # e.g. a theme this plotly lacks: the default
        for widget in widgets.values():  # stored on change only: opening is no edit
            widget.observe(self._save_settings, "value")

    def _settings_widgets(self) -> dict[str, Widget]:
        """The settings, by their key in ``adata.uns["sclab"]["settings"]``."""
        plotter = self.plotter
        return {
            "counts_layer": self.counts_layer_dropdown,
            "batch_key": self.batch_key_dropdown,
            "background": self.background_checkbox,
            "random_state": self.seed_input,
            "plot_backend": plotter.backend_dropdown,
            "max_3d_points": plotter.max_3d_points_input,
            "min_size": plotter.min_size_input,
            "template": plotter.template_dropdown,
            "colorscale": plotter.colorscale_dropdown,
            "palette": plotter.palette_dropdown,
            "keep_tables": self.keep_tables_checkbox,
        }

    def _save_settings(self, _=None):
        settings = {k: w.value for k, w in self._settings_widgets().items()}
        self.dataset.adata.uns.setdefault(UNS_KEY, {})["settings"] = settings

    # both keep the current value: a deleted layer or column is not a new setting
    def _counts_layer_options(self) -> list[str]:
        layers = layer_keys(self.dataset.adata)
        current = self.dataset.counts_layer
        return layers if current in layers else [*layers, current]

    def _batch_key_options(self) -> dict:
        columns = text_or_category_columns(self.dataset.adata.obs)
        key = self.processor.batch_key  # e.g. a numeric "batch" column
        return {"(none)": "", **{c: c for c in columns}, **({key: key} if key else {})}

    def _set_seed(self, change: dict):
        """The steps' ``random_state``, unless the user set another (0: their defaults)."""
        old, new = change["old"], change["new"]
        for step in self.processor.steps.values():
            params = getattr(step, "fixed_params", None)
            if (
                params is None
                or "random_state" not in inspect.signature(step.function).parameters
            ):
                continue
            if params.get("random_state", 0) != old:
                continue  # set by the user
            if new == 0:
                params.pop("random_state", None)
            else:
                params["random_state"] = new

    def _set_keep_tables(self, change: dict):
        results = self.processor.results_panel.results_stack.children
        tables = [getattr(r, "table_output", None) for r in results]
        for output in (
            self.dataset.obs_table_output,
            self.dataset.var_table_output,
            *(t for t in tables if isinstance(t, LazyOutput)),
        ):
            output.unload = not change["new"]
            if output.unload and not output.shown:
                output.outputs = ()

    def generate_code(self, header: bool = True) -> str:
        """Python script reproducing the steps applied in the dashboard."""
        return self.processor.generate_code(header=header)

    @property
    def ds(self):
        return self.dataset

    @property
    def pr(self):
        return self.processor

    @property
    def pl(self):
        return self.plotter


def _widget_registry() -> dict:
    """Model id -> live widget; its name changed across ipywidgets versions."""
    from ipywidgets.widgets import widget

    if hasattr(widget, "_instances"):  # >= 8.1
        return widget._instances
    if hasattr(Widget, "_active_widgets"):  # 8.0
        return Widget._active_widgets
    return Widget.widgets


def _live_widget_ids() -> set[str]:
    return set(_widget_registry())


def _close_tree(root, keep: set[str]):
    """Close every widget reachable from ``root``.

    Follows the attributes of widgets and of sclab objects, and the items of
    dicts, lists, tuples and sets, so widgets held but not displayed (e.g.
    the figure, step controls, the broker's logs) are closed too. Widgets in
    ``keep`` (model ids alive before ``root`` was built, such as module-level
    widgets shared by every dashboard) are left open.
    """
    seen, stack, widgets = set(), [root], []
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(obj, Widget) or type(obj).__module__.startswith("sclab."):
            if isinstance(obj, Widget):
                widgets.append(obj)
            stack.extend(getattr(obj, "__dict__", {}).values())
    for widget in widgets:
        if widget.comm is not None and widget.model_id not in keep:
            Widget.close(widget)  # not the override: root may be a dashboard


class DataLoader(VBox):
    """Pick an AnnData to load; ``on_load(adata, source)`` gets the choice.

    ``user_ns`` is the namespace listed under "Defined datasets"; by default,
    the locals of whoever called the code that built this loader.
    """

    on_load: Callable[[AnnData, str | None], None]
    adata: AnnData

    upload: FileUpload
    upload_info: Output
    upload_row: HBox
    upload_row_label: Label

    url: Text
    load_button: Button
    url_row: HBox
    url_row_label: Label

    defined_adatas_dict: dict[str, AnnData]
    defined_adatas: ToggleButtons
    defined_adatas_row: HBox
    defined_adatas_label: Label

    progress_output: Output
    continue_button: Button

    def __init__(
        self,
        on_load: Callable[[AnnData, str | None], None],
        user_ns: dict | None = None,
    ):
        self.on_load = on_load
        self.source = None  # path or URL of self.adata, if loaded from one

        self.upload_row_label = Label("Load from file:", layout=Layout(width="120px"))
        self.upload = FileUpload(layout=Layout(width="200px"))
        self.upload_info = Output(layout=Layout(width="95%"))
        self.upload_row = HBox(
            [
                self.upload_row_label,
                self.upload,
                self.upload_info,
            ],
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

        if user_ns is None:
            user_ns = inspect.stack()[2].frame.f_locals
        user_f_locals = user_ns
        self.defined_adatas_dict = {}
        for name, variable_type in [(k, type(v)) for k, v in user_f_locals.items()]:
            if variable_type is AnnData and not name.startswith("_"):
                self.defined_adatas_dict[name] = user_f_locals[name]

        self.defined_adatas_label = Label(
            "Defined datasets:", layout=Layout(width="120px")
        )
        self.defined_adatas = ToggleButtons(
            options=list(self.defined_adatas_dict.keys()),
            value=None,
            layout=Layout(width="100%"),
        )
        self.defined_adatas_row = HBox(
            [self.defined_adatas_label, self.defined_adatas],
            layout=Layout(width="100%"),
        )
        self.defined_adatas.observe(self.on_defined_adatas_toggle, "value")

        self.progress_output = Output(layout=Layout(width="95%"))
        self.continue_button = Button(
            description="Continue", layout=Layout(width="100%"), button_style="success"
        )
        self.continue_button.on_click(self.on_continue)

        VBox.__init__(
            self,
            [
                # self.url_row,
                # self.upload_row,
                self.defined_adatas_row,
                self.progress_output,
            ],
            layout=Layout(width="100%"),
        )

    def on_defined_adatas_toggle(self, *args, **kwargs):
        adata = self.defined_adatas_dict[self.defined_adatas.value]

        self.progress_output.clear_output()
        with self.progress_output:
            print(f"Loaded {adata.shape[0]} observations and {adata.shape[1]} genes\n")
            print(adata)
            display(self.continue_button)

        self.adata = adata
        self.source = None

    def on_upload(self, *args, **kwargs):
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
        suffix = Path(filename).suffix

        if suffix not in (".h5", ".h5ad"):
            self.upload_info.clear_output()
            with self.upload_info:
                print(f"`{filename}` is not valid")
                print("Please upload a 10x .h5 or .h5ad file")
            return

        adata = _read_adata_bytes(
            file["content"].tobytes(), suffix, var_names="gene_ids"
        )

        with self.progress_output:
            print(f"Loaded {adata.shape[0]} observations and {adata.shape[1]} genes\n")
            print(adata)
            display(self.continue_button)

        self.adata = adata
        self.source = None  # uploaded bytes: no path to load from

    def on_load_url(self, *args, **kwargs):
        self.progress_output.clear_output()
        with self.progress_output:
            self.adata = load_adata_from_url(self.url.value)
            self.source = self.url.value
            display(self.continue_button)

    def on_continue(self, *args, **kwargs):
        self.on_load(self.adata, self.source)
        self.adata = None
        self.defined_adatas_dict = {}
