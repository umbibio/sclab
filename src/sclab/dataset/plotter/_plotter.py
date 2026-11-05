import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import svgpathtools as spt
from ipywidgets.widgets import (
    HTML,
    Box,
    BoundedFloatText,
    BoundedIntText,
    Dropdown,
    GridBox,
    Layout,
    VBox,
)
from pandas.api.types import is_any_real_numeric_dtype, is_bool_dtype
from plotly.graph_objs.scatter import Marker as Marker2D
from plotly.graph_objs.scatter3d import Marker as Marker3D
from scipy.interpolate import make_smoothing_spline

from ...event import EventBroker, EventClient
from ...utils._widgets import Download, ResizeNotifier
from .._dataset import SCLabDataset
from ._backends import (
    BACKENDS,
    color_spec,
    density_lines,
    make_view,
    png_html,
    render_png,
    value_size_factors,
)
from ._controls import PlotterControls
from ._views import ViewGallery
from ._utils import (
    COLOR_DISCRETE_SEQUENCE,
    _make_density_heatmap,
    make_periodic_smoothing_spline,
    rotate_multiple_steps,
)

# 3D draws a seeded random subsample of this many cells unless full resolution is
# on; WebGL 3D rotation cost is linear in the point count
MAX_3D_POINTS = 200_000
# plotly.js fails to allocate a scatter3d trace beyond ~600-800k points
MAX_TRACE_POINTS = 500_000
POS = "__sclab_row__"  # df column: row position of each cell in Plotter._rows
# plotly.js draws opaque 3D markers with a depth test, translucent ones in trace order
OPACITY_3D_NOTE = "3D: opacity below 1 draws by trace order, not depth"


def _flat(props: dict, prefix: str = ""):
    """(dotted path, leaf) pairs of a trace's props; arrays and lists are leaves."""
    for key, value in props.items():
        if isinstance(value, dict):
            yield from _flat(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _patch_trace(old, new):
    """Make widget trace ``old`` equal to ``new``, sending only what differs.

    plotly skips values equal to the current ones, so unchanged arrays (e.g. x/y
    on a size change) are not sent again. ``_props`` is read to avoid deep-copying
    the arrays, as ``to_plotly_json`` would.
    """
    new_props = dict(_flat(new._props))
    for key, _ in list(_flat(old._props)):
        if key not in new_props and key != "uid":
            old[key] = None
    for key, value in new_props.items():
        if key not in ("type", "uid"):
            old[key] = value


def _float32(values):
    values = np.asarray(values)
    return values.astype(np.float32) if values.dtype.kind in "iuf" else values


class Plotter(GridBox, EventClient):
    g: go.FigureWidget
    controls: PlotterControls
    dataset: SCLabDataset
    state: dict[str, Any]
    # box/lasso of the last plot selection, see Processor._selection_params
    selection_shape: dict | None = None

    events: list[str] = [
        "dplt_dataset_change",
        "dplt_selected_points_change",
        "dplt_point_click",
        "dplt_layout_shapes_change",
        "dplt_start_drawing_request",
        "dplt_end_drawing_request",
        "dplt_soft_path_computed",
        "dplt_plot_figure_request",
        "dplt_add_trace_request",
        "dplt_add_vline_request",
        "dplt_add_hline_request",
        "dplt_add_data_as_line_trace_request",
    ]
    preemptions: dict[str, list[str]] = {
        "dplt_selected_points_change": [
            "dspr_selection_values_change",
        ],
        "dplt_plot_figure_request": [
            "ctrl_data_key_change",
            "ctrl_color_change",
            "ctrl_sizeby_change",
        ],
    }

    def __init__(
        self,
        dataset: SCLabDataset | None = None,
        broker: EventBroker | None = None,
        backend: str = "plotly",
    ):
        if broker is None:
            assert dataset is not None, "dataset must be provided if broker is None"
            broker = dataset.broker

        EventClient.__init__(self, broker)
        # one redraw per batch (publish, step run) instead of one per event
        self.make_new_figure = self.broker.coalesce(self.make_new_figure)

        self.modebar = dict(
            add=[],
        )
        self.state = {}
        # index map: trace i of self.g holds the cells self._rows[self._trace_rows[i]]
        # (None for traces that are not cells: density, lines, histograms)
        self._rows = pd.Index([])
        self._trace_rows: list[np.ndarray | None] = []
        self._structure = None  # what the current traces were built for, see _apply
        self._view = None  # the 2D backend's view, None for plotly (see _backends)
        self._fallback = None  # why the next 2D draw must use plotly
        self._last_call: dict = {}  # make_new_figure's arguments, for view redraws

        self._init_figure()
        self._init_controls()
        self._init_settings()
        self.load_dataset(dataset)
        self.views = ViewGallery(self)
        self.download = Download()

        graph_layout = Layout(
            display="block", width="100%", height="600px", border="0px solid red"
        )
        # the figure stays attached, hidden while a backend's view shows: a
        # re-attached plotly view misses the traces sent right after it and stays empty
        self._figure_box = Box([self.g], layout=Layout(display="block", width="100%"))
        self._resize_notifier = ResizeNotifier()
        self._shown = self.g
        self.graph_box = Box(
            [self._figure_box, self._resize_notifier, self.download],
            layout=graph_layout,
        )
        self.notice = HTML()  # why a plotly plot is shown instead of the backend's
        # min_width: the card strip scrolls instead of widening the plot column
        graph_column = VBox(
            [self.graph_box, self.notice, self.views], layout=Layout(min_width="0")
        )

        GridBox.__init__(
            self,
            [graph_column, self.controls],
            layout=Layout(
                width="100%",
                # a bare `auto` plot column collapses to a few px on narrow viewports
                grid_template_columns="minmax(400px, 1fr) 350px",
                grid_template_areas=""" "graph controls" """,
                border="0px solid black",
            ),
        )
        self.broker.subscribe("ctrl_data_key_change", self.make_new_figure)
        self.broker.subscribe("ctrl_color_change", self.make_new_figure)
        self.broker.subscribe("ctrl_n_dimensions_change", self.make_new_figure)
        self.broker.subscribe("ctrl_selected_axes_1_change", self.make_new_figure)
        self.broker.subscribe("ctrl_selected_axes_2_change", self.make_new_figure)
        self.broker.subscribe("ctrl_selected_axes_3_change", self.make_new_figure)
        self.broker.subscribe("ctrl_log_axes_1_change", self.make_new_figure)
        self.broker.subscribe("ctrl_log_axes_2_change", self.make_new_figure)
        self.broker.subscribe("ctrl_log_axes_3_change", self.make_new_figure)
        self.broker.subscribe("ctrl_rotate_steps_change", self.make_new_figure)
        self.broker.subscribe("ctrl_refresh_button_click", self.make_new_figure)
        self.broker.subscribe("ctrl_histogram_nbins_change", self.make_new_figure)
        self.broker.subscribe("dset_selected_rows_change", self.select_points)
        self.broker.subscribe("dset_total_rows_change", self.make_new_figure)
        self.broker.subscribe("dset_metadata_change", self.make_new_figure)
        self.backend = backend

    def _init_settings(self):
        """The dashboard's Settings tab: plot defaults."""
        self.backend_dropdown = Dropdown(
            options=BACKENDS,
            value="plotly",
            description="2D plots",
            tooltip="1D and 3D plots always use plotly",
        )
        self.max_3d_points_input = BoundedIntText(
            value=MAX_3D_POINTS,
            min=1,
            max=10**9,
            step=10_000,
            description="3D points",
            tooltip="3D plots draw a random subsample of this many cells",
        )
        self.min_size_input = BoundedFloatText(
            value=0.3,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Min. size",
            tooltip="Emphasize values: size of zero or missing values, relative to "
            "the marker size",
        )
        self.backend_dropdown.observe(lambda c: self._set_backend(c["new"]), "value")
        self.max_3d_points_input.observe(
            lambda _: self.ctrl_full_resolution_change_callback(None), "value"
        )
        self.min_size_input.observe(
            lambda _: self.ctrl_sizeby_change_callback(None), "value"
        )
        style = dict(description_width="80px")
        settings = (
            self.backend_dropdown,
            self.max_3d_points_input,
            self.min_size_input,
        )
        for widget in settings:
            widget.style = style
        self.settings = VBox(
            [
                HTML("<b>Plots</b>"),
                *settings,
                self.controls.full_resolution,
            ],
            layout=Layout(padding="8px"),
        )

    @property
    def backend(self) -> str:
        """The 2D plot backend: "plotly", "jscatter" (jupyter-scatter) or "static"."""
        return self.backend_dropdown.value

    @backend.setter
    def backend(self, name: str):
        if name not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {name!r}")
        self.backend_dropdown.value = name

    @property
    def max_3d_points(self) -> int:
        """3D plots draw at most this many cells, unless at full resolution."""
        return self.max_3d_points_input.value

    @max_3d_points.setter
    def max_3d_points(self, n: int):
        self.max_3d_points_input.value = n

    def _set_backend(self, name: str):
        self._say("")
        old, self._view = self._view, None
        if name != "plotly":
            try:
                self._view = make_view(name, self)
            except ImportError:
                self.backend_dropdown.value = "plotly"
                self._say(
                    "jupyter-scatter is not installed (pip install jupyter-scatter): "
                    "using plotly"
                )
                return
        self._show(self.g)
        if old is not None:
            old.close()
        self.make_new_figure(**self._last_call)

    def _say(self, text: str):
        style = "font-size:12px;color:#b45309"
        self.notice.value = f"<span style='{style}'>{text}</span>" if text else ""

    def _note_3d_opacity(self):
        c = self.controls
        show = c.n_dimensions.value == "3D" and c.marker_opacity.value < 1
        if show or OPACITY_3D_NOTE in self.notice.value:
            self._say(OPACITY_3D_NOTE if show else "")

    def _show(self, widget):
        """Put ``widget`` (the figure or a backend's view) in the plot area."""
        if self._shown is widget:
            return
        view = () if widget is self.g else (widget,)
        if view:  # free the hidden plotly traces
            self.g.data = []
            self._structure, self._trace_rows = None, []
        self._figure_box.layout.display = "none" if view else "block"
        self.graph_box.children = (
            self._figure_box,
            *view,
            self._resize_notifier,
            self.download,
        )
        self._shown = widget

    def _view_shown(self) -> bool:
        return self._shown is not self.g

    def _view_2d(self, ndims: str, figure_kwargs: dict):
        """The backend view to draw a 2D plot with, or None to draw with plotly."""
        if self._view is None:
            return None
        if ndims != "2D":
            self._say("")
            return None
        reason = self._fallback
        if self.state.get("drawing"):
            reason = "path drawing"
        elif set(figure_kwargs) - {"color_continuous_scale"}:
            reason = f"plot options {sorted(figure_kwargs)}"
        msg = f"Shown with plotly: {self.backend} does not support {reason}"
        self._say(msg if reason else "")
        return None if reason else self._view

    def _plotly_now(self, reason: str):
        """Redraw a backend's 2D plot with plotly, for plotly-only features."""
        if not self._view_shown():
            return
        self._fallback = reason
        try:
            type(self).make_new_figure(self, **self._last_call)  # now, not coalesced
        finally:
            self._fallback = None

    def _view_redraw(self) -> bool:
        """Redraw the backend's 2D plot (it sends what changed); False for plotly."""
        if not self._view_shown():
            return False
        self.make_new_figure(**self._last_call)
        return True

    def _frame_2d(self, df, col_x, col_y, colorby, sizeby, scale, figure_kwargs):
        """What a backend draws, see sclab.dataset.plotter._backends."""
        c = self.controls
        sizeby = c.sizeby.value if sizeby is None else sizeby
        scale = c.marker_size.value if scale is None else scale
        size = scale  # px diameter, as plotly draws marker.size with sizeref
        if sizeby is not None:
            values = df[sizeby].astype(float).clip(lower=0).fillna(0.0).to_numpy()
            if values.max() > 0:
                size = scale * np.sqrt(values / values.max())
        elif (factors := self._size_factors(df, colorby, sizeby)) is not None:
            size = scale * factors
        selected = self.dataset._selected_rows
        x, y = (df[col].to_numpy(dtype=float) for col in (col_x, col_y))
        density = None
        if c.show_density.value:
            density = density_lines(np.column_stack([x, y]), c)
        return {
            "index": df.index,
            "x": x,
            "y": y,
            "x_title": col_x,
            "y_title": col_y,
            "title": self.dataset._selected_data_key,
            "color": None
            if not colorby
            else color_spec(df[colorby], figure_kwargs.get("color_continuous_scale")),
            "color_title": colorby,
            "size": size,
            "marker_size": scale,
            "selected": None if selected is None else df.index.isin(selected),
            "opacity": c.marker_opacity.value,
            "density": density,
            "height": c.plot_height.value,
            "width": c.plot_width.value,
            "aspect_equal": c.aspect_equal.value,
            "hover": c.enable_hover_info.value and col_x != "rank",
        }

    def _size_factors(self, df, colorby, sizeby) -> np.ndarray | None:
        """Per-cell marker diameters relative to the marker size (None: uniform).

        "Emphasize values": for a numeric color column, cells with values > 0 are
        drawn larger, by rank, and zero or missing values small (the Settings
        tab's "Min. size"), see ``value_size_factors``. Sizes follow the first of:
        a selection (selected cells large, the others small), the "Size by"
        column, then these factors. 1D histograms ignore sizes.
        """
        if (
            sizeby is not None
            or not colorby
            or not self.controls.emphasize_values.value
        ):
            return None
        if colorby not in df or not is_any_real_numeric_dtype(df[colorby]):
            return None  # categorical or boolean colors
        return value_size_factors(df[colorby], self.min_size_input.value)

    def _init_figure(self):
        self.g = go.FigureWidget(
            dict(
                layout=dict(
                    xaxis_title="",
                    yaxis_title="",
                    title="",
                    template="simple_white",
                    height=600,
                    modebar=self.modebar,
                )
            )
        )

        # plotly>=6 (anywidget) never autosizes after attach; without this the first
        # render falls back to plotly.js's 700px default width when the div is unsized.
        # `responsive` makes plotly.js resize with its container.
        self.g._config = {**self.g._config, "responsive": True}

        # plotly counts frontend views in `_view_count` (nothing reads it) and its view
        # sends the decrement when removed, after the kernel closed the comm (closing
        # a dashboard): "No such comm" in the log. A None count is never sent.
        self.g._view_count = None

        def relayout_publisher(event):
            if not event["new"]:
                return

            relayout_data: dict = event["new"]["relayout_data"]

            if selections := relayout_data.get("selections", None):
                self.selection_shape = self._selection_shape(selections)
                if self.selection_shape:
                    selected_points = self._shape_points(self.selection_shape)
                else:  # may lag behind for lassos, see _shape_points
                    selected_points = self.selected_points
                if selected_points.empty:
                    selected_points = None
                self.broker.publish("dplt_selected_points_change", selected_points)

                with self.g.batch_update():
                    # Contour traces have no selectedpoints
                    self.g.update_traces(
                        selectedpoints=None,
                        selector=lambda t: not isinstance(t, go.Contour),
                    )
                    self.g.plotly_relayout({"selections": None})

                return

            if dragmode := relayout_data.get("dragmode", None):
                ndims = self.controls.n_dimensions.value
                self.state[f"{ndims}dragmode"] = dragmode
                return

            if shapes := relayout_data.get("shapes", None):
                self.broker.publish("dplt_layout_shapes_change", shapes=shapes)
                return

        self.g.observe(relayout_publisher, names="_js2py_relayout")

    def data_hash(self) -> str:
        """Identifies the data a pinned view was made on; the Processor refines it."""
        from ..processor import _history  # circular import

        return _history.history_hash(_history.loads_uns(self.dataset.adata))

    def _init_controls(self):
        self.controls = PlotterControls(self.broker)

    def load_dataset(self, dataset: SCLabDataset | None):
        if dataset is None:
            dataset = SCLabDataset(self.broker)

        if not isinstance(dataset, SCLabDataset):
            raise TypeError("dataset must be an instance of SCLabDataset")

        if dataset.broker.id != self.broker.id:
            raise ValueError("dataset broker must be the same as the provided broker")

        self.dataset = dataset

        self.broker.publish("dplt_dataset_change", dataset)

    def _selection_shape(self, selections: list[dict]) -> dict | None:
        """A single box/lasso of plotly's layout.selections on plain 2D axes.

        Plotly reports a box as ``{"type": "rect", "x0", "x1", "y0", "y1"}`` and a
        lasso as ``{"type": "path", "path": "M x,y L x,y ... Z"}``, in data units.
        Returns the schema of ``sclab.dataset.processor._selection_code`` or None
        where the plotted coordinates are not a basis's columns (log, rank or
        rotated axes, no data key) or for several selections.
        """
        ctrl = self.controls
        key = self.dataset._selected_data_key
        col_x, col_y = (d.value for d in ctrl.selected_axes.children[:2])
        if (
            len(selections) != 1
            or ctrl.n_dimensions.value != "2D"
            or not key
            or col_x == "rank"
            or ctrl.rotate_steps.value
            or any(c.value for c in ctrl.log_axes.children[:2])
        ):
            return None

        if key == "metadata":  # obs columns
            shape = {"dims": [col_x, col_y]}
        else:
            columns = self.dataset.data.columns
            dims = [int(columns.get_loc(col_x)), int(columns.get_loc(col_y))]
            shape = {"basis": key, "dims": dims}

        (selection,) = selections
        try:  # frontend payload
            if selection.get("type") == "rect":
                x = sorted(float(selection[k]) for k in ("x0", "x1"))
                y = sorted(float(selection[k]) for k in ("y0", "y1"))
                return {"type": "box", **shape, "x": x, "y": y}
            if selection.get("type") == "path":
                vertices = re.findall(r"([-+.\deE]+),([-+.\deE]+)", selection["path"])
                polygon = [[float(x), float(y)] for x, y in vertices]
                if len(polygon) >= 3:
                    return {"type": "lasso", **shape, "polygon": polygon}
        except (KeyError, TypeError, ValueError):
            pass
        return None

    def _shape_points(self, shape: dict) -> pd.Index:
        """The cells inside a box/lasso, as plotly selects them (edges aside).

        Preferred over the traces' selectedpoints: when the `selections` relayout
        arrives they are often not synced yet for lassos (the frontend's message
        buffer drops plotly_selected for the empty one that follows it).
        """
        from ..processor._selection_code import shape_mask  # circular import

        adata = self.dataset.adata
        return adata.obs_names[shape_mask(adata, shape)]

    @property
    def selected_points(self):
        positions = [
            rows[list(trace.selectedpoints)]
            for trace, rows in zip(self.g.data, self._trace_rows)
            if rows is not None and trace.selectedpoints
        ]
        if not positions:
            return pd.Index([])
        return self._rows[np.unique(np.concatenate(positions))]

    def select_points(self, points: pd.Index | None):
        if self.controls.n_dimensions.value != "2D":
            self.make_new_figure()
            return
        elif self._view_redraw():
            return
        else:
            with self.g.batch_update():  # one message, not one per trace property
                self.update_marker_sizes()
            return

    @property
    def data_for_plot(self):
        # TODO: Needs refactoring. Define clear behavior

        if not self.controls.rotate_steps.value:
            return self.dataset.data

        ndims_avail = self.dataset.data.shape[1]
        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        col_z: str = self.controls.selected_axes.children[2].value

        if ndims_avail < 3 or not col_z:
            return self.dataset.data

        X = self.dataset.data[[col_x, col_y, col_z]]
        return rotate_multiple_steps(X, self.controls.rotate_steps.value)

    def make_new_figure(
        self,
        metadata: pd.DataFrame | None = None,
        colorby: str | None = None,
        sizeby: str | None = None,
        marker_size_scale: float | None = None,
        new_value: str | None = None,
        *args,
        **figure_kwargs,
    ):
        self.g.layout.annotations = []
        # what drew the figure, for pinned views: the controls, or e.g. a gene's
        # values (events pass the metadata and its color column positionally)
        extra = colorby not in (None, self.controls.color.value) or figure_kwargs
        self.state["source"] = "gene" if extra else "controls"
        self._last_call = dict(
            metadata=metadata,
            colorby=colorby,
            sizeby=sizeby,
            marker_size_scale=marker_size_scale,
            **figure_kwargs,
        )

        ndims = self.controls.n_dimensions.value
        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        col_z: str = self.controls.selected_axes.children[2].value

        invalid_axes = col_x == "rank" and not col_y

        if self.dataset.data.empty or invalid_axes:
            self._show(self.g)
            layout = dict(
                xaxis_title="", yaxis_title="", title="No Data", template="simple_white"
            )
            self.g.update(dict(layout=layout, data=[]), overwrite=True)
            self._structure, self._trace_rows = None, []
            return

        view = self._view_2d(ndims, figure_kwargs)
        self._note_3d_opacity()
        self._show(self.g if view is None else view.widget)
        data = self.data_for_plot
        if metadata is None:
            metadata = self.dataset.metadata

        df = data.join(metadata.loc[:, ~metadata.columns.isin(data.columns)])
        rows = df.index
        df[POS] = np.arange(len(df))
        if col_x == "rank":
            df = df.sort_values(col_y, ascending=False, na_position="last")
            df["rank"] = np.arange(df.shape[0]) + 1

        log_x: bool = self.controls.log_axes.children[0].value
        log_y: bool = self.controls.log_axes.children[1].value
        log_z: bool = self.controls.log_axes.children[2].value

        if log_x and ndims in ["1D", "2D", "3D"]:
            df = df[df[col_x] > 0]
            df["log " + col_x] = df[col_x].apply(np.log10)
            col_x = "log " + col_x

        if log_y and ndims in ["2D", "3D"]:
            df = df[df[col_y] > 0]
            df["log " + col_y] = df[col_y].apply(np.log10)
            col_y = "log " + col_y

        if log_z and ndims == "3D":
            df = df[df[col_z] > 0]
            df["log " + col_z] = df[col_z].apply(np.log10)
            col_z = "log " + col_z

        log_x = log_y = log_z = False

        if ndims in ["1D", "2D", "3D"]:
            x = df[col_x]
            dx = x.max() - x.min()

        if ndims in ["2D", "3D"]:
            y = df[col_y]
            dy = y.max() - y.min()

        if ndims == "3D":
            z = df[col_z]
            dz = z.max() - z.min()

        selected = df["is_selected"]
        selection_is_active = not selected.isna().all()
        if selection_is_active and (ndims == "1D" or ndims == "3D"):
            df = df.loc[selected]

        if colorby is None:
            colorby = self.controls.color.value

        if view is not None:
            args = (df, col_x, col_y, colorby, sizeby, marker_size_scale, figure_kwargs)
            view.draw(self._frame_2d(*args))
            return

        if colorby:
            ascending = is_any_real_numeric_dtype(df[colorby])
            # 2D: high values drawn on top. 3D: the depth test decides (see
            # OPACITY_3D_NOTE), and the same order lets a recolor send only colors
            if not (ascending and ndims == "3D"):
                df = df.sort_values(colorby, ascending=ascending, na_position="first")

            series: pd.Series = df[colorby]
            # where(): pandas 3's str dtype keeps NaN instead of making it "nan"
            if isinstance(series.dtype, pd.CategoricalDtype | bool):
                df[colorby] = series.astype(str).where(series.notna(), "NA")
            elif ndims == "1D":
                # break into 10 evenly distributed bins
                bins = pd.cut(series, 10)
                df[colorby] = bins.astype(str).where(bins.notna(), "NA")

        n_cells = len(df)
        full = self.controls.full_resolution.value
        limit = self.max_3d_points
        if ndims == "3D" and not full and n_cells > limit:
            # seeded per cell, so recoloring keeps the same cells; selected first
            rank = np.random.default_rng(0).random(len(rows))[df[POS].to_numpy()]
            rank[df["is_selected"].eq(True).to_numpy(dtype=bool, na_value=False)] -= 1
            keep = np.argpartition(rank, limit)[:limit]
            df = df.iloc[np.sort(keep)]  # keeps the color sort order

        if ndims == "1D":
            fig = px.histogram(
                df,
                x=col_x,
                log_x=log_x,
                color=colorby,
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
                template="simple_white",
                nbins=self.controls.histogram_nbins.value,
                **figure_kwargs,
            )
        elif ndims == "2D":
            fig = px.scatter(
                df,
                x=col_x,
                y=col_y,
                log_x=log_x,
                log_y=log_y,
                color=colorby,
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
                custom_data=[POS],  # -> the index map, see _compact_traces
                template="simple_white",
                render_mode="webgl",
                **figure_kwargs,
            )
        elif ndims == "3D":
            fig = px.scatter_3d(
                df,
                x=col_x,
                y=col_y,
                z=col_z,
                log_x=log_x,
                log_y=log_y,
                log_z=log_z,
                color=colorby,
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
                custom_data=[POS],  # -> the index map, see _compact_traces
                template="simple_white",
                **figure_kwargs,
            )
        else:
            layout = dict(
                xaxis_title="",
                yaxis_title="",
                title="No Data",
                template="simple_white",
                height=self.controls.plot_height.value,
                modebar=self.modebar,
                dragmode=False,
            )
            self.g.update(dict(layout=layout, data=[]), overwrite=True)
            self._structure, self._trace_rows = None, []
            return

        fig.update_layout(legend_title_text="")
        fig.update_layout(coloraxis_colorbar_title_text="")

        fig.update_traces(marker_color="lightgray", selector={"name": "NA"})
        if colorby and is_bool_dtype(series):
            fig.update_traces(marker_color="lightgray", selector={"name": "False"})
            fig.update_traces(
                marker_color=COLOR_DISCRETE_SEQUENCE[0], selector={"name": "True"}
            )
            fig.data = fig.data[::-1]

        if colorby and isinstance(series.dtype, pd.CategoricalDtype):
            color_pallete_size = len(COLOR_DISCRETE_SEQUENCE)
            for i, cat in enumerate(series.cat.categories):
                fig.update_traces(
                    marker_color=COLOR_DISCRETE_SEQUENCE[i % color_pallete_size],
                    selector={"name": cat},
                )

        if ndims == "2D" or ndims == "3D":
            # before the density contour is added: it keeps its own opacity
            fig.update_traces(opacity=self.controls.marker_opacity.value)

            fig.update_layout(legend_traceorder="reversed")
            if col_x == "rank" or not self.controls.enable_hover_info.value:
                fig.update_traces(hoverinfo="skip", hovertemplate=None)

        else:
            fig.data = fig.data[::-1]
            fig.update_layout(legend_traceorder="normal")

        if ndims == "2D" and self.controls.show_density.value:
            # make density plot
            data = df[[col_x, col_y]].to_numpy(dtype=float)
            grid_resolution = self.controls.density_grid_resolution.value
            line_smoothing = self.controls.density_line_smoothing.value
            bandwidth_factor = self.controls.density_bandwidth_factor.value
            contours = self.controls.density_contours.value
            trace = _make_density_heatmap(
                data,
                bandwidth_factor,
                grid_resolution,
                line_smoothing,
                contours,
                "rgba(255, 255, 255, 0)",
            )
            # plotly>=6 turns numeric arrays into typed-array dicts once in the widget
            self.state["density_z_range"] = (trace.z.min(), trace.z.max())
            fig.add_trace(trace)

        height = self.controls.plot_height.value
        fig.update_layout(height=height)

        if self.controls.aspect_equal.value:
            if ndims == "3D":
                fig.update_layout(scene_aspectmode="data")
            elif ndims == "2D":
                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # fix ranges
        if ndims == "1D":
            fig.update_layout(xaxis_range=[x.min() - dx * 0.1, x.max() + dx * 0.1])

        elif ndims == "2D":
            fig.update_layout(xaxis_range=[x.min() - dx * 0.1, x.max() + dx * 0.1])
            fig.update_layout(yaxis_range=[y.min() - dy * 0.1, y.max() + dy * 0.1])

        elif ndims == "3D":
            fig.update_layout(
                scene_xaxis_range=[x.min() - dx * 0.1, x.max() + dx * 0.1]
            )
            fig.update_layout(
                scene_yaxis_range=[y.min() - dy * 0.1, y.max() + dy * 0.1]
            )
            fig.update_layout(
                scene_zaxis_range=[z.min() - dz * 0.1, z.max() + dz * 0.1]
            )

        # set dragmode
        if not (dragmode := self.get_dragmode()):
            if ndims == "3D":
                dragmode = "turntable"
            elif ndims == "2D":
                dragmode = "lasso"
            else:
                dragmode = False

        title = data_key = self.dataset._selected_data_key
        if len(df) < n_cells:
            title = f"{data_key} ({len(df):,} of {n_cells:,} cells shown)"
        fig.update_layout(title=title, modebar=self.modebar)
        fig.update_layout(template_layout_shapedefaults_fillcolor="rgba(0, 0, 0, 0)")
        fig.update_layout(legend={"itemsizing": "constant"})

        if colorby:
            fig.update_layout(showlegend=True)
        else:
            fig.update_layout(showlegend=False)

        if ndims == "3D":
            # the layout is replaced below: keep the user's camera on the same basis
            if self.g.layout.scene.uirevision == data_key:
                if camera := self.g.layout.scene.camera.to_plotly_json():
                    fig.update_layout(scene_camera=camera)
            fig.update_layout(scene_uirevision=data_key)

        traces = self._compact_traces(fig, rows, ndims)
        # sized before drawing: within a batch, a property must be set only once
        self.update_marker_sizes(
            metadata=metadata,
            colorby=colorby,
            sizeby=sizeby,
            marker_size_scale=marker_size_scale,
            traces=traces,
        )
        structure = (ndims, data_key, col_x, col_y, col_z) + tuple(
            (t.type, t.marker.coloraxis if "marker" in t else None) for t in traces
        )
        with self.g.batch_update():
            self._apply(traces, fig.layout, structure)
            self.set_dragmode(dragmode)

        self._unpin_size()

    def _compact_traces(self, fig: go.Figure, rows: pd.Index, ndims: str):
        """fig's traces with float32 arrays and cell ids as customdata.

        px put each cell's row position in customdata (``custom_data=[POS]``); it
        becomes the index map and the cell id shown on hover. 3D traces are split
        into chunks of at most ``MAX_TRACE_POINTS``, sharing their legend entry.
        Sets the index map (``_rows``, ``_trace_rows``) for the returned traces.
        """
        names = np.asarray(rows, dtype=object)
        traces, trace_rows = [], []
        for trace in fig.data:
            if trace.customdata is None:  # density, histogram bars
                traces.append(trace)
                trace_rows.append(None)
                continue
            pos = np.asarray(trace.customdata)[:, 0].astype(np.int64)
            arrays = {k: _float32(trace[k]) for k in "xyz" if k in trace}
            if isinstance(trace.marker.color, np.ndarray):  # continuous color
                arrays["marker.color"] = _float32(trace.marker.color)
            step = MAX_TRACE_POINTS if ndims == "3D" else max(len(pos), 1)
            for start in range(0, max(len(pos), 1), step):
                chunk = slice(start, start + step)
                part = trace if start == 0 else type(trace)(trace, showlegend=False)
                part.update({k: v[chunk] for k, v in arrays.items()})
                part.customdata = names[pos[chunk]]
                traces.append(part)
                trace_rows.append(pos[chunk])
        self._rows, self._trace_rows = rows, trace_rows
        return traces

    def _apply(self, traces: list, layout, structure: tuple):
        """Draw traces and layout, patching the current traces where possible.

        The traces are patched (only changed properties are sent, numpy arrays as
        binary buffers) when they were built for the same basis, dims, axes, trace
        types and continuous/categorical color, and no traces were added since.
        Otherwise they are all replaced. The layout is always replaced, as before.
        """
        current = tuple(t.type for t in self.g.data)
        if structure == self._structure and current == tuple(t.type for t in traces):
            for old, new in zip(self.g.data, traces):
                _patch_trace(old, new)
            self.g.update(layout=layout, overwrite=True)
        else:
            self.g.update(dict(data=traces, layout=layout), overwrite=True)
        self._structure = structure

    def _unpin_size(self):
        # plotly.js pins the width (and drops autosize) whenever a relayout carries
        # `height`, which every figure update does. `autosize` relayouts unpin it again
        # so the plot keeps following its container. Sent directly because the public
        # plotly_relayout skips values that are already True on the Python side.
        self.g._send_relayout_msg({"autosize": True})

    def set_dragmode(self, dragmode: str | None = None):
        ndims = self.controls.n_dimensions.value
        if dragmode is None:
            dragmode = self.state.get(f"{ndims}dragmode", False)

        self.g.plotly_relayout({"dragmode": dragmode})
        self.state[f"{ndims}dragmode"] = dragmode

    def get_dragmode(self) -> str:
        ndims = self.controls.n_dimensions.value
        return self.state.get(f"{ndims}dragmode", False)

    def set_shapes(self, shapes: list[dict]):
        self.g.layout.shapes = shapes
        self.g._send_relayout_msg({"shapes": shapes})
        self.broker.publish("dplt_layout_shapes_change", shapes=shapes)

    def clear_shapes(self):
        self.set_shapes([])

    def update_marker_sizes(
        self,
        marker_size_scale: float | None = None,
        sizeby: str | None = None,
        colorby: str | None = None,
        metadata: pd.DataFrame | None = None,
        traces: list | None = None,  # default: the figure's
    ):
        # a gene's plot: its color column is in the metadata it was drawn with
        last = {} if self.state.get("source") == "custom" else self._last_call
        if colorby is None:
            colorby = last.get("colorby") or self.controls.color.value

        if marker_size_scale is None:
            marker_size_scale = self.controls.marker_size.value

        if sizeby is None:
            sizeby = self.controls.sizeby.value

        if metadata is None:
            df = self.dataset.metadata
        else:
            df = metadata

        sizes = None  # one size for all: scalar marker.size
        index = df.index  # the cells of sizes
        marker_sizeref = 1.0 / marker_size_scale**2

        # is_selected may be a boolean column or a column of NaNs
        if active_selection := (not (selected := df["is_selected"]).isna().all()):
            # if it is a boolean column, a selection has been defined (possible all False)
            sizes = np.where(selected.eq(True).to_numpy(bool, na_value=False), 3.0, 0.5)

        elif sizeby is not None:
            sizeby_series = df[sizeby].astype(float)
            sizeby_series.loc[sizeby_series < 0] = 0.0
            sizeby_series = sizeby_series.fillna(0.0)
            marker_sizeref = sizeby_series.max() / marker_size_scale**2
            sizes = sizeby_series.to_numpy()

        else:
            values = df if colorby in df else last.get("metadata")
            if values is not None:
                factors = self._size_factors(values, colorby, sizeby)
                if factors is not None:  # diameters -> marker.size areas
                    sizes, index = factors**2, values.index

        if sizes is not None:
            if not index.equals(self._rows):  # e.g. a gene's metadata
                sizes = pd.Series(sizes, index=index).reindex(self._rows).to_numpy()
            sizes = sizes.astype(np.float32)

        trace: go.Scatter | go.Scattergl | go.Scatter3d
        traces = self.g.data if traces is None else traces
        for trace, rows in zip(traces, self._trace_rows):
            if rows is None:
                continue

            trace.hovertemplate = self.get_hovertemplate(
                info={colorby: trace.name},
                show_size=not active_selection,
                label=self._label(trace),
                metadata=df,
            )

            marker: Marker2D | Marker3D = trace.marker
            marker.sizemode = "area"
            marker.sizeref = marker_sizeref
            # plotly.js applies sizeref to arrays only: a scalar is the px diameter
            marker.size = marker_size_scale if sizes is None else sizes[rows]
            marker.line.width = 0.0

    @staticmethod
    def _label(trace) -> str:
        # cell ids: customdata on our traces, hovertext on custom (px) figures
        return "%{customdata}" if trace.hovertext is None else "%{hovertext}"

    def get_hovertemplate(
        self,
        info: dict = {},
        show_size: bool = True,
        label: str = "%{customdata}",
        metadata: pd.DataFrame | None = None,  # default: the dataset's (O(cells))
    ) -> str:
        ndims = self.controls.n_dimensions.value
        x_axis = self.controls.selected_axes.children[0].value
        y_axis = self.controls.selected_axes.children[1].value
        z_axis = self.controls.selected_axes.children[2].value
        marker_color: str | None = self.controls.color.value
        marker_size: str | None = self.controls.sizeby.value

        if marker_color is not None:
            if metadata is None:
                metadata = self.dataset.metadata
            series = metadata[marker_color]
            marker_color_is_cat = isinstance(series.dtype, pd.CategoricalDtype)
        else:
            marker_color_is_cat = False

        is_histogram = ndims == "1D"
        is_scatter = ndims in ["2D", "3D"]
        is_3d_scatter = ndims == "3D"
        is_na = ndims == "NA"

        hovertemplate = ""
        if is_scatter | is_na:
            hovertemplate = f"<b>{label}</b><br>"

        if is_scatter and marker_color and not marker_color_is_cat:
            info.pop(marker_color, None)
            hovertemplate += f"{marker_color} = %{{marker.color}}<br>"

        if is_scatter and marker_color and marker_color_is_cat:
            trace_name = info.pop(marker_color, None)
            if trace_name is not None:
                hovertemplate += f"{marker_color} = {trace_name}<br>"

        if is_scatter and marker_size and show_size and marker_size != marker_color:
            hovertemplate += f"{marker_size} = %{{marker.size}}<br>"

        if is_histogram:
            hovertemplate += "count = %{y}<br>"

        hovertemplate = hovertemplate[:-4]
        hovertemplate += "<extra>"

        if not is_na:
            hovertemplate += f"{x_axis} = %{{x}}<br>"
        if is_scatter:
            hovertemplate += f"{y_axis} = %{{y}}<br>"
        if is_3d_scatter:
            hovertemplate += f"{z_axis} = %{{z}}<br>"

        if info:
            hovertemplate += "<br>"

        for key, val in info.items():
            if key or val:
                hovertemplate += f"{key} = {val}<br>"
        else:
            hovertemplate = hovertemplate[:-4]

        hovertemplate += "</extra>"

        return hovertemplate

    def dplt_point_click_callback(self, row_name, device_state, **kwargs):
        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            return

        if row_name not in self._rows:
            return
        position = self._rows.get_loc(row_name)
        for trace, rows in zip(self.g.data, self._trace_rows):
            if rows is None or not (idx := np.flatnonzero(rows == position)).size:
                continue

            # marker.size may be a scalar px diameter: 1.0 in sizeref units
            size = 1.0 if np.isscalar(trace.marker.size) else trace.marker.size
            marker_size = np.broadcast_to(size, rows.shape).copy()
            (idx,) = idx
            default_size = self.controls.marker_size.value
            current_size = marker_size[idx]

            if current_size == default_size:
                marker_size[idx] = default_size * 3
            else:
                marker_size[idx] = default_size

            trace.marker.size = marker_size
            break

    def ctrl_show_density_change_callback(self, new_value):
        if self._view_redraw():
            return
        if new_value:
            try:
                trace = next(filter(lambda o: isinstance(o, go.Contour), self.g.data))
                trace.visible = True
            except StopIteration:
                self.make_new_figure()
        else:
            try:
                trace = next(filter(lambda o: isinstance(o, go.Contour), self.g.data))
                trace.visible = False
            except StopIteration:
                pass

    def ctrl_density_line_smoothing_change_callback(self, new_value):
        if self.controls.show_density.value:
            self.g.update_traces(
                selector=dict(type="contour"), line_smoothing=new_value
            )

    def ctrl_density_contours_change_callback(self, new_value):
        if self._view_redraw():
            return
        if self.controls.show_density.value:
            z_min, z_max = self.state["density_z_range"]
            start = z_min + 0.0001
            end = z_max + 0.0001
            size = (end - start) / new_value
            contours = dict(start=start, end=end, size=size)

            self.g.update_traces(selector=dict(type="contour"), contours=contours)

    def ctrl_density_grid_resolution_change_callback(self, new_value):
        if self.controls.show_density.value:
            self.make_new_figure()

    def ctrl_density_bandwidth_factor_change_callback(self, new_value):
        if self.controls.show_density.value:
            self.make_new_figure()

    def ctrl_full_resolution_change_callback(self, new_value):
        if self.controls.n_dimensions.value == "3D":
            self.make_new_figure()

    def ctrl_marker_size_change_callback(self, new_value):
        ndims = self.controls.n_dimensions.value
        if ndims == "1D" or self._view_redraw():
            return

        with self.g.batch_update():
            self.update_marker_sizes(marker_size_scale=new_value)

    def ctrl_sizeby_change_callback(self, new_value):
        ndims = self.controls.n_dimensions.value
        if ndims == "1D" or self._view_redraw():
            return

        with self.g.batch_update():
            self.update_marker_sizes(sizeby=new_value)

    def ctrl_emphasize_values_change_callback(self, new_value):
        self.ctrl_sizeby_change_callback(None)

    def ctrl_marker_opacity_change_callback(self, new_value):
        self._note_3d_opacity()
        ndims = self.controls.n_dimensions.value
        if ndims == "1D" or self._view_redraw():
            return

        with self.g.batch_update():
            self.g.update_traces(selector=dict(type="scatter"), opacity=new_value)
            self.g.update_traces(selector=dict(type="scattergl"), opacity=new_value)
            self.g.update_traces(selector=dict(type="scatter3d"), opacity=new_value)

    def ctrl_aspect_equal_change_callback(self, new_value):
        if self._view_redraw():
            return
        ndims = self.controls.n_dimensions.value
        if new_value:
            if ndims == "3D":
                self.g.update_layout(scene_aspectmode="data")
            elif ndims == "2D":
                self.g.update_xaxes(scaleanchor="y", scaleratio=1)
                self.g.update_yaxes(scaleanchor="x", scaleratio=1)
        else:
            if ndims == "3D":
                self.g.update_layout(scene_aspectmode="auto")
            elif ndims == "2D":
                self.g.update_xaxes(scaleanchor=None, scaleratio=None)
                self.g.update_yaxes(scaleanchor=None, scaleratio=None)

    def ctrl_enable_hover_info_change_callback(self, new_value):
        if self._view_redraw():
            return
        if new_value:
            colorby = self.controls.color.value
            metadata = self.dataset.metadata
            for trace in self.g.data:
                trace.hoverinfo = "all"
                trace.hovertemplate = self.get_hovertemplate(
                    info={colorby: trace.name},
                    label=self._label(trace),
                    metadata=metadata,
                )
        else:
            self.g.update_traces(hoverinfo="skip", hovertemplate=None)

    def ctrl_plot_width_change_callback(self, new_value):
        if new_value > 0:
            self.graph_box.layout.width = f"{new_value}px"
        else:
            self.graph_box.layout.width = "auto"
        self._view_redraw()

    def ctrl_plot_height_change_callback(self, new_value):
        self.graph_box.layout.height = f"{new_value}px"
        self.g.update_layout(height=new_value)
        self._unpin_size()
        self._view_redraw()

    def ctrl_save_button_click_callback(self):
        dataset_name = self.dataset.name
        data_key = self.dataset._selected_data_key
        if not data_key:
            return

        filename = f"{dataset_name}___saved_figure___data_{data_key}"

        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            variable = self.controls.selected_axes.children[0].value
            filename += f"___{variable}-histogram"
        elif ndims == "2D":
            x = self.controls.selected_axes.children[0].value
            y = self.controls.selected_axes.children[1].value
            filename += f"___{x}-vs-{y}"
        elif ndims == "3D":
            x = self.controls.selected_axes.children[0].value
            y = self.controls.selected_axes.children[1].value
            z = self.controls.selected_axes.children[2].value
            filename += f"___{x}-vs-{y}-vs-{z}"

        if self.controls.color.value:
            filename += f"___colorBy_{self.controls.color.value}"

        filename += ".html"
        if self._view_shown():  # a backend's plot: saved as a static image
            html = png_html(render_png(self._view.frame), filename)
        else:
            html = self.g.to_html()
        self.download.save(html, filename)  # in the browser, not on disk

    def dspr_clear_selection_click_callback(self):
        with self.g.batch_update():
            if self.controls.n_dimensions.value == "2D":
                self.g.update_traces(selectedpoints=None)
            self.g.plotly_relayout({"selections": None})

    def dplt_start_drawing_request_callback(self):
        if "2D" not in self.controls.n_dimensions.options:
            return

        self.state["drawing"] = True  # before any redraw: drawing needs plotly
        self.controls.dict["n_dimensions"].value = "2D"
        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        self._plotly_now("path drawing")
        self.state["previous_dragmode"] = self.get_dragmode()
        for control_name in [
            "data_key",
            "color",
            "sizeby",
            "n_dimensions",
            "selected_axes_1",
            "selected_axes_2",
            "selected_axes_3",
            "log_axes_1",
            "log_axes_2",
            "log_axes_3",
        ]:
            self.controls.dict[control_name].disabled = True
        self.set_dragmode("drawopenpath")

    def dplt_end_drawing_request_callback(self):
        if not self.state.get("drawing", False):
            return

        self.state["drawing"] = False
        self.clear_shapes()
        for control_name in [
            "data_key",
            "color",
            "sizeby",
            "n_dimensions",
            "selected_axes_1",
            "selected_axes_2",
            "selected_axes_3",
            "log_axes_1",
            "log_axes_2",
            "log_axes_3",
        ]:
            self.controls.dict[control_name].disabled = False
        self.set_dragmode(self.state["previous_dragmode"])

    def dplt_layout_shapes_change_callback(self, shapes: list[dict]):
        shapes = [s for s in shapes if s["name"] != "smooth path"]

        if not shapes:
            return

        path = spt.Path()
        for shape in shapes:
            p = spt.parse_path(shape["path"])
            if len(path) > 0:
                path.append(spt.Line(path[-1].end, p[0].start))
            path.extend(p)

        pi, pf = path[0].start, path[-1].end
        if np.abs(pf - pi) / path.length() < 0.01:
            path[-1].end = pi

        n = len(path) + 1
        if n < 3:
            return

        p = np.array([path[0].start] + [line.end for line in path])
        path_is_closed = p[-1] == p[0]

        if n >= 5:
            X = np.array([[line.start, line.end] for line in path])
            t = np.abs(np.diff(X))
            t = np.insert(np.cumsum(t / t.sum()), 0, 0)
            if path_is_closed:
                px_bspl = make_periodic_smoothing_spline(
                    t[:-1], p.real[:-1], t_range=(0, 1), lam=1 / 5e3 / n
                )
                py_bspl = make_periodic_smoothing_spline(
                    t[:-1], p.imag[:-1], t_range=(0, 1), lam=1 / 5e3 / n
                )
                t = np.linspace(0, 1, 10 * n) % 1
            else:
                px_bspl = make_smoothing_spline(t, p.real, lam=1 / 1e3 / n)
                py_bspl = make_smoothing_spline(t, p.imag, lam=1 / 1e3 / n)
                t = np.linspace(0, 1, 10 * n)
            x, y = px_bspl(t), py_bspl(t)

            points = x + 1j * y
            spath = spt.Path()
            spath.extend(
                [spt.Line(start, end) for start, end in zip(points, points[1:])]
            )

        else:
            spath = path

        s1 = {
            "editable": False,
            "visible": False,
            "name": "drawn path",
            "showlegend": False,
            "legend": "legend",
            "legendgroup": "",
            "legendgrouptitle": {
                "text": "",
                "font": {"weight": "normal", "style": "normal", "variant": "normal"},
            },
            "legendrank": 1000,
            "label": {"text": "", "texttemplate": ""},
            "xref": "x",
            "yref": "y",
            "layer": "above",
            "opacity": 1,
            "line": {"color": "#444", "width": 4, "dash": "solid"},
            "type": "path",
            "path": path.d(use_closed_attrib=path_is_closed).replace(" ", ""),
        }
        s2 = {
            "editable": False,
            "visible": True,
            "name": "smooth path",
            "showlegend": False,
            "legend": "legend",
            "legendgroup": "",
            "legendgrouptitle": {
                "text": "",
                "font": {"weight": "normal", "style": "normal", "variant": "normal"},
            },
            "legendrank": 1000,
            "label": {"text": "", "texttemplate": ""},
            "xref": "x",
            "yref": "y",
            "layer": "above",
            "opacity": 1,
            "line": {"color": "#444", "width": 4, "dash": "solid"},
            "type": "path",
            "path": spath.d(use_closed_attrib=path_is_closed).replace(" ", ""),
        }

        self.g._send_relayout_msg({"shapes": (s1, s2)})

        times = np.linspace(0, 1, 100 * n)
        if n >= 5:
            points = px_bspl(times) + 1j * py_bspl(times)
        else:
            points = np.array([path.point(t) for t in times])

        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        x = self.data_for_plot[col_x].values
        y = self.data_for_plot[col_y].values

        X = x + 1j * y
        P = points
        T = times

        self.broker.publish(
            "dplt_soft_path_computed",
            time_points=T,
            data_points=X,
            path_points=P,
            path_is_closed=path_is_closed,
        )

    def dplt_plot_figure_request_callback(
        self,
        figure: go.Figure | None = None,
        metadata: pd.DataFrame | None = None,
        colorby: str | None = None,
        sizeby: str | None = None,
        **figure_kwargs,
    ):
        if figure is None:
            self.make_new_figure(
                metadata=metadata,
                colorby=colorby,
                sizeby=sizeby,
                **figure_kwargs,
            )
            return

        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        if self._view_shown():
            self._say(
                f"Shown with plotly: {self.backend} does not support custom figures"
            )
            self._show(self.g)
        if colorby in self.controls.color.options:
            self.controls.color.value = colorby
        else:
            self.controls.color.value = None

        if sizeby in self.controls.sizeby.options:
            self.controls.sizeby.value = sizeby
        else:
            self.controls.sizeby.value = None

        figure.layout.template = self.g.layout.template

        # custom figures name their cells by hovertext (px hover_name)
        self._rows = rows = self.dataset.metadata.index
        self._trace_rows = []
        for trace in figure.data:
            ids = trace.hovertext if "hovertext" in trace else None
            pos = None if ids is None else rows.get_indexer(np.asarray(ids))
            self._trace_rows.append(None if pos is None or (pos < 0).any() else pos)
        self._structure = None
        self.state["source"] = "custom"

        with self.g.batch_update():
            self.g.update(figure.to_dict(), overwrite=True)
            self.update_marker_sizes(colorby=colorby, sizeby=sizeby)
            self.g.plotly_relayout({"dragmode": False})

    def dplt_add_trace_request_callback(self, trace: go.Scatter | go.Scattergl):
        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        self._plotly_now("extra traces")
        with self.g.batch_update():
            self.g.add_trace(trace)

    def dplt_add_vline_request_callback(self, vlines: float | list[float], **kwargs):
        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        self._plotly_now("extra lines")
        if not isinstance(vlines, list):
            vlines = [vlines]

        # not in batch_update: plotly reads back the shape's xref/yref
        for vline in vlines:
            self.g.add_vline(x=vline, **kwargs)

    def dplt_add_hline_request_callback(self, hlines: float | list[float], **kwargs):
        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        self._plotly_now("extra lines")
        if not isinstance(hlines, list):
            hlines = [hlines]

        # not in batch_update: plotly reads back the shape's xref/yref
        for hline in hlines:
            self.g.add_hline(y=hline, **kwargs)

    def dplt_add_data_as_line_trace_request_callback(
        self, data_key: str, data: pd.DataFrame, **kvargs
    ):
        if data_key != self.dataset._selected_data_key:
            return

        self.broker.flush(self.make_new_figure)  # draw on the pending figure
        self._plotly_now("extra traces")
        ndims = self.controls.n_dimensions.value
        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        col_z: str = self.controls.selected_axes.children[2].value

        data = rotate_multiple_steps(data, self.controls.rotate_steps.value)

        if ndims == "1D":
            if col_x + "_height" not in data.columns:
                return

            x = data[col_x].values
            y = data[col_x + "_height"].values
            trace = go.Scattergl(x=x, y=y, mode="lines", **kvargs)

        elif ndims == "2D":
            x = data[col_x].values
            y = data[col_y].values
            trace = go.Scattergl(x=x, y=y, mode="lines", **kvargs)

        elif ndims == "3D":
            x = data[col_x].values
            y = data[col_y].values
            z = data[col_z].values
            trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", **kvargs)

        else:
            return

        with self.g.batch_update():
            self.g.add_trace(trace)
