import inspect
import tempfile
from collections.abc import Callable
from html import escape
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
    Layout,
    Output,
    Select,
    Tab,
    Text,
    ToggleButtons,
    VBox,
    Widget,
)
from traitlets import TraitError

from ._io import (
    download,
    is_mtx_dir,
    is_valid_url,
    load_adata_from_url,
    peek_shape,
    read_adata,
    unique_path,
)
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
    the random seed (empty: each step's own default; 0 is a seed like any
    other), the plot backend and appearance, and whether the
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
        self.seed_input = Text(
            placeholder="step default",
            description="Random seed",
            tooltip="random_state of the steps that have one; empty: their defaults",
            continuous_update=False,
        )
        self.seed_message = HTML()
        self._seed = None  # the applied seed: None is empty
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
            self.background_checkbox,
            self.keep_tables_checkbox,
        ):
            widget.style = dict(description_width="100px")  # as the plot settings

        def section(title, *children):
            return VBox(
                [HTML(f"<b>{title}</b>"), *children], layout=Layout(padding="8px")
            )

        self.settings = VBox(
            [
                section("Data", self.counts_layer_dropdown, self.batch_key_dropdown),
                section(
                    "Steps",
                    self.background_checkbox,
                    self.seed_input,
                    self.seed_message,
                ),
                plotter.settings,
                section("Tables", self.keep_tables_checkbox),
            ]
        )

        with self.broker.batch():  # one redraw
            for key, value in stored.items():
                if key in widgets and key not in BUILD_SETTINGS:
                    widget = widgets[key]
                    try:  # h5ad gives numpy scalars
                        if key == "random_state":  # None or "": empty
                            value = "" if value in (None, "") else int(value)
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
        settings["random_state"] = self._seed  # None: empty
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
        """The steps' ``random_state``, unless the user set another (empty: their defaults)."""
        try:
            text = change["new"].strip()
            new = int(text) if text else None
        except ValueError:
            self.seed_message.value = "Enter an integer, or leave empty."
            return
        self.seed_message.value = ""
        old, self._seed = self._seed, new
        for step in self.processor.steps.values():
            params = getattr(step, "fixed_params", None)
            if (
                params is None
                or "random_state" not in inspect.signature(step.function).parameters
            ):
                continue
            if params.get("random_state") != old:
                continue  # set by the user
            if new is None:
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


UPLOAD_WARN_BYTES = 200 * 2**20  # larger uploads can starve the websocket's pings
_FILE_SUFFIXES = (".h5ad", ".h5")  # what read_adata reads, besides mtx folders


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _is_dataset(path: Path) -> bool:
    """Whether :func:`read_adata` reads ``path``: listed by the file browser."""
    return is_mtx_dir(path) or (path.is_file() and path.suffix in _FILE_SUFFIXES)


def _describe(path: Path) -> str:
    """``name · kind · size · shape``, the shape only if cheap to know."""
    if path.is_dir():
        size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
        parts = [f"{path.name}/", "10x mtx folder"]
    else:
        size = path.stat().st_size
        parts = [path.name, "10x h5" if path.suffix == ".h5" else "h5ad"]
    parts.append(_fmt_size(size))
    if shape := peek_shape(path):
        parts.append("×".join(map(str, shape)))
    return " · ".join(parts)


def _error_html(e: Exception) -> str:
    message = escape(str(e) or type(e).__name__)
    return f"<span style='color:#b91c1c;font-size:12px'>{message}</span>"


class DataLoader(VBox):
    """Pick a dataset: a file, a URL, an upload or an AnnData defined in the notebook.

    Tabs: "Files" browses the notebook's working directory (and its
    subfolders, never above it) for .h5ad, 10x .h5 files and 10x mtx folders;
    "URL" downloads a .h5ad or 10x .h5 file and "Upload" saves one sent from
    the browser, both into ``dest_dir`` (``downloads/`` and ``uploads/``; a
    temporary folder if None); "Defined in notebook" lists the AnnData objects
    of ``user_ns`` (by default the locals of whoever called the code that
    built this loader). It opens on "Defined in notebook" if there are any.

    A file goes to ``on_file(path, url=..., uploaded=...)`` (``url``: where
    it was downloaded from); without ``on_file`` it is read and goes, like a
    defined AnnData, to ``on_load(adata, source)``. Errors are shown in the
    panel.
    """

    on_load: Callable[[AnnData, str | None], None]
    adata: AnnData

    def __init__(
        self,
        on_load: Callable[[AnnData, str | None], None],
        user_ns: dict | None = None,
        on_file: Callable[..., None] | None = None,
        dest_dir: str | Path | None = None,
    ):
        self.on_load = on_load
        self.on_file = on_file or self._read_file
        self.dest_dir = Path(dest_dir) if dest_dir else None
        self.source = None  # path or URL of self.adata, if loaded from one
        self.root = self.folder = Path.cwd().resolve()
        self._listing = False
        self.status = HTML()

        # Files: the kernel lists them, nothing goes through the browser
        self.breadcrumbs = HTML()
        self.files = Select(rows=10, layout=Layout(width="100%"))
        self.add_button = Button(
            description="Load", button_style="success", layout=Layout(width="120px")
        )
        self.path_text = Text(
            placeholder="or type a path (relative to this folder)",
            layout=Layout(width="100%"),
        )
        self.go_button = Button(description="Go", layout=Layout(width="60px"))
        files_tab = VBox(
            [
                self.breadcrumbs,
                self.files,
                HBox([self.add_button]),
                HBox([self.path_text, self.go_button], layout=Layout(width="100%")),
            ]
        )
        self.files.observe(self._on_file_select, "value")
        self.add_button.on_click(lambda _: self._guard(self._load_selected))
        self.go_button.on_click(lambda _: self._guard(self._go, self.path_text.value))

        if self.dest_dir is not None:
            saved_in = f"Saved in {escape(str(self.dest_dir))}"
        else:
            saved_in = "No workspace_dir: saved in a temporary folder"
        note = "<span style='font-size:12px;color:#555'>{}</span>"

        self.url = Text(
            placeholder="https://.../data.h5ad", layout=Layout(width="100%")
        )
        self.load_button = Button(description="Download", layout=Layout(width="120px"))
        self.url_progress = HTML()
        self.url_row = VBox(
            [
                HBox([self.url, self.load_button], layout=Layout(width="100%")),
                HTML(note.format(f"A .h5ad or 10x .h5 file. {saved_in}/downloads.")),
                self.url_progress,
            ]
        )
        self.load_button.on_click(self.on_load_url)

        self.upload = FileUpload(accept=".h5ad,.h5", layout=Layout(width="200px"))
        warning = (
            "For small files only: uploads go through the browser connection. "
            f"Over ~{UPLOAD_WARN_BYTES // 2**20} MB, put the file next to the "
            f"notebook and use Files instead. {saved_in}/uploads."
        )
        self.upload_info = HTML(note.format(warning))
        self.upload_row = VBox([self.upload, self.upload_info])
        self.upload.observe(self.on_upload, "value")

        if user_ns is None:
            user_ns = inspect.stack()[2].frame.f_locals
        self.defined_adatas_dict = {
            name: value
            for name, value in user_ns.items()
            if type(value) is AnnData and not name.startswith("_")
        }
        self.defined_adatas = ToggleButtons(
            options=list(self.defined_adatas_dict.keys()),
            value=None,
            layout=Layout(width="100%"),
        )
        self.defined_adatas_row = HBox(
            [self.defined_adatas], layout=Layout(width="100%")
        )
        self.defined_adatas.observe(self.on_defined_adatas_toggle, "value")

        self.progress_output = Output(layout=Layout(width="95%"))
        self.continue_button = Button(
            description="Continue", layout=Layout(width="100%"), button_style="success"
        )
        self.continue_button.on_click(self.on_continue)

        self.tabs = Tab(
            children=[
                files_tab,
                self.url_row,
                self.upload_row,
                self.defined_adatas_row,
            ],
            titles=["Files", "URL", "Upload", "Defined in notebook"],
        )
        self.tabs.selected_index = 3 if self.defined_adatas_dict else 0
        self._guard(self._cd, self.root)

        VBox.__init__(
            self,
            [self.tabs, self.status, self.progress_output],
            layout=Layout(width="100%"),
        )

    def _guard(self, fn, *args, **kwargs) -> bool:
        """Run ``fn``; show its error in the panel instead of raising it."""
        self.status.value = ""
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - a widget callback: show every error
            self.status.value = _error_html(e)
            return False
        return True

    def _inside_root(self, path: Path) -> Path:
        path = path.expanduser().resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{path} is outside the notebook folder {self.root}")
        return path

    def _cd(self, folder: Path):
        folder = self._inside_root(Path(folder))
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")
        children = sorted(
            (p for p in folder.iterdir() if not p.name.startswith(".")),
            key=lambda p: p.name.lower(),
        )
        datasets = [(_describe(p), p) for p in children if _is_dataset(p)]
        folders = [(f"{p.name}/", p) for p in children if p.is_dir()]
        folders = [item for item in folders if not is_mtx_dir(item[1])]
        up = [("../", folder.parent)] if folder != self.root else []
        self.folder = folder
        crumbs = [
            self.root.name or str(self.root),
            *folder.relative_to(self.root).parts,
        ]
        empty = "" if datasets else " (no .h5ad, 10x .h5 or mtx folders here)"
        self.breadcrumbs.value = f"<b>{escape(' / '.join(crumbs))}</b>{escape(empty)}"
        self._listing = True  # setting options selects the first: not a pick
        try:
            self.files.options = up + folders + datasets
            self.files.value = None
        finally:
            self._listing = False

    def _on_file_select(self, change):
        path = change["new"]
        if self._listing or path is None:
            return
        if path.is_dir() and not is_mtx_dir(path):
            self._guard(self._cd, path)

    def _load_selected(self):
        if self.files.value is None:
            raise ValueError("Pick a file in the list first")
        self._load_file(self.files.value)

    def _go(self, text: str):
        path = self._inside_root(self.folder / text.strip())
        if _is_dataset(path):
            self._load_file(path)
        elif path.is_dir():
            self._cd(path)
        elif path.exists():
            raise ValueError(
                f"Cannot read {path.name}: pick a .h5ad, 10x .h5 or mtx folder"
            )
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")

    def _load_file(self, path: Path, url: str | None = None, uploaded=False):
        self.on_file(path, url=url, uploaded=uploaded)

    def _read_file(self, path: Path, url: str | None = None, uploaded=False):
        """``on_file`` without a Workspace: read it and pass it on."""
        self.on_load(read_adata(path), url or str(path))

    def _dest(self, sub: str) -> Path:
        if self.dest_dir is None:  # kept: the dataset is read from it later
            self.dest_dir = Path(tempfile.mkdtemp(prefix="sclab-"))
        return self.dest_dir / sub

    def on_load_url(self, *args, **kwargs):
        self._guard(self._download, self.url.value.strip())

    def _download(self, url: str):
        def progress(done: int, total: int | None):
            of = f" of {_fmt_size(total)}" if total else ""
            self.url_progress.value = f"Downloading: {_fmt_size(done)}{of}"

        self.url_progress.value = ""
        self.load_button.disabled = True
        try:
            path = download(url, self._dest("downloads"), progress=progress)
        finally:
            self.load_button.disabled = False
        self.url_progress.value = f"Saved to {escape(str(path))}"
        self._load_file(path, url=url)

    def on_upload(self, *args, **kwargs):
        if files := self.upload.value:
            try:
                self._guard(self._save_upload, files[0])
            finally:
                self.upload.value = ()  # frees the bytes; the same file can be sent again

    def _save_upload(self, file: dict):
        name = Path(file["name"]).name
        if Path(name).suffix not in _FILE_SUFFIXES:
            raise ValueError(f"Cannot read {name}: upload a .h5ad or 10x .h5 file")
        if file["size"] > UPLOAD_WARN_BYTES:
            self.upload_info.value = _error_html(
                Warning(
                    f"{name} is {_fmt_size(file['size'])}: large uploads can stall "
                    "the notebook. Next time, put it next to the notebook and use Files."
                )
            )
        folder = self._dest("uploads")
        folder.mkdir(parents=True, exist_ok=True)
        path = unique_path(folder / name)
        path.write_bytes(file["content"])
        self._load_file(path, uploaded=True)

    def on_defined_adatas_toggle(self, *args, **kwargs):
        adata = self.defined_adatas_dict[self.defined_adatas.value]

        self.progress_output.clear_output()
        with self.progress_output:
            print(f"Loaded {adata.shape[0]} observations and {adata.shape[1]} genes\n")
            print(adata)
            display(self.continue_button)

        self.adata = adata
        self.source = None

    def on_continue(self, *args, **kwargs):
        self.on_load(self.adata, self.source)
        self.adata = None
        self.defined_adatas_dict = {}
