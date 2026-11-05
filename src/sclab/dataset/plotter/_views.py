"""Pinned plot views: save the plotter's view and get back to it later.

A view spec is a JSON dict, independent of the plotting backend::

    {
        "v": 2,
        "id": "3f2a9c1e",
        "name": "View 1",
        "created": "2026-11-03T10:15:42+01:00",
        "source": "controls",  # or "gene", "custom"
        "controls": {"data_key": "X_umap", "n_dimensions": "2D", ...},
        "ranges": {"x": [x0, x1], "y": [y0, y1]} | {"camera": {...}} | None,
        "dragmode": "lasso",
        "backend": "plotly",
        "history_hash": "<sha256 of the steps that changed the data>",
        "n_obs": 1000,
        "settings": {"min_size": 0.3},  # the Settings tab's "Min. size"
        "gene": {"gene": "CD3E", "layer": "counts", "colorscale": "Oryel"},
    }

``controls`` holds the value of every value widget in ``PlotterControls.dict``;
``ranges`` the frontend zoom/pan (1D/2D) or 3D camera, as synced into the
figure's layout; None for 2D plots drawn by the other backends (``backend``):
jupyter-scatter does not sync its camera back, static images have no zoom. A
restore keeps the current backend. The active selection, drawn paths and extra
traces are not part of a view. Views are stored in ``adata.uns["sclab"]["views"]``
as a JSON string, so they travel with the h5ad file. Version 1 specs (no
``source``, ``settings`` or ``gene``) are read as "controls" views.

``source`` is what drew the plot (``Plotter.state["source"]``): the plot
controls, a gene of the Gene Expression step (``gene``: restored by plotting the
gene again) or a custom figure (``dplt_plot_figure_request(figure=...)``). A
custom view is a snapshot: kept in memory with its figure, for its thumbnail
and PNG, never restored nor saved in ``adata.uns``.

Thumbnails are rendered in the kernel by matplotlib (:func:`render_png`, the
static backend's renderer), at most ``THUMB_MAX_CELLS`` cells, off the main
thread in a kernel. They are cached in memory, never in the h5ad, and redrawn
when the data changes (``Plotter.data_hash`` or the number of cells).
"""

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from html import escape
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import HTML, BoundedIntText, Button, HBox, Image, Layout, Text, VBox
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from plotly.colors import sample_colorscale
from traitlets import TraitError

from ...utils._widgets import CodeActions
from ._backends import (
    NA_COLOR,
    PALETTE,
    _hex,
    density_lines,
    make_frame,
    render_png,
)
from ._utils import rotate_multiple_steps

UNS_KEY = "sclab"  # as in sclab.dataset.processor._history (circular import)
VERSION = 2
THUMB_WIDTH = 320  # px; shown at CARD_WIDTH, sharp on high-density screens
CARD_WIDTH = 184
THUMB_MAX_CELLS = 50_000  # thumbnails draw a random subsample of more cells

# restore order matters: each of these resets the ones after it
FIRST = [
    "data_key",  # forces 2D and new axes options
    "n_dimensions",  # resets the log axes
    "selected_axes_1",  # each axis resets its log checkbox
    "selected_axes_2",
    "selected_axes_3",
    "log_axes_1",
    "log_axes_2",
    "log_axes_3",
]
LAST = ["show_density"]  # density settings first: their callbacks need a drawn density


def loads_uns(adata) -> list[dict]:
    """The views stored in ``adata.uns`` (malformed entries dropped), as v2."""
    raw = adata.uns.get(UNS_KEY, {}).get("views")
    try:
        views = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []
    if not isinstance(views, list):
        return []
    return [
        {**v, "v": VERSION, "source": v.get("source", "controls")}
        for v in views
        if isinstance(v, dict)
        and isinstance(v.get("controls"), dict)
        and v.get("source", "controls") in ("controls", "gene")
    ]


def dumps_uns(adata, views: list[dict]) -> None:
    saved = [v for v in views if v["source"] != "custom"]  # snapshots: memory only
    adata.uns.setdefault(UNS_KEY, {})["views"] = json.dumps(saved)


def _is_stale(plotter, spec: dict) -> bool:
    return (
        spec.get("history_hash") != plotter.data_hash()
        or spec.get("n_obs") != plotter.dataset.adata.n_obs
    )


def gene_values(adata, gene: str, layer: str) -> np.ndarray:
    """A gene's values per cell, as the Gene Expression step plots them."""
    X = adata[:, gene].X if layer == "X" else adata[:, gene].layers[layer]
    return np.asarray(X.sum(axis=1)).ravel()


def _gene_problems(adata, gene: dict) -> list[str]:
    if gene["gene"] not in adata.var_names:
        return [f"gene '{gene['gene']}' missing"]
    if gene["layer"] != "X" and gene["layer"] not in adata.layers:
        return [f"layer '{gene['layer']}' missing"]
    return []


def capture(plotter, name: str = "") -> dict:
    """The plotter's current view as a spec (see the module docstring).

    Raises ValueError for a gene plot not drawn by the Gene Expression step.
    """
    source = plotter.state.get("source", "controls")
    gene = plotter.state.get("gene") if source == "gene" else None
    if source == "gene" and (
        not gene or gene.get("gene") != plotter._last_call.get("colorby")
    ):
        raise ValueError(
            "This plot cannot be pinned: only plots of the plot settings, of the "
            "Gene Expression step and custom figures can"
        )
    controls = {
        k: w.value for k, w in plotter.controls.dict.items() if hasattr(w, "value")
    }
    layout = plotter.g.layout
    if plotter._view_shown() or source == "custom":  # no camera from jupyter-scatter
        ranges = None
    elif controls["n_dimensions"] == "3D":
        camera = layout.scene.camera.to_plotly_json()
        ranges = {"camera": camera} if camera else None
    elif controls["n_dimensions"] in ("1D", "2D"):
        ranges = {"x": layout.xaxis.range, "y": layout.yaxis.range}
    else:
        ranges = None
    spec = {
        "v": VERSION,
        "id": uuid4().hex[:8],
        "name": name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "controls": controls,
        "ranges": ranges,
        "dragmode": plotter.get_dragmode(),
        "backend": plotter.backend if plotter._view_shown() else "plotly",  # drew it
        "history_hash": plotter.data_hash(),
        "n_obs": plotter.dataset.adata.n_obs,
        "settings": {"min_size": plotter.min_size_input.value},
    }
    if gene:
        spec["gene"] = dict(gene)
    if source == "custom":
        spec["title"] = layout.title.text or ""
    from ..processor._history import to_json  # circular import

    return to_json(spec)


def problems(plotter, spec: dict) -> list[str]:
    """What keeps ``spec`` from being restored exactly, as short badges."""
    if spec["source"] == "custom":
        return []
    c, dataset = spec["controls"], plotter.dataset
    if c.get("data_key") not in dataset.data_dict:
        return [f"{c.get('data_key')} missing"]
    gene = spec.get("gene")
    keys = ("sizeby",) if gene else ("color", "sizeby")  # a gene plot ignores color
    found = [
        f"{key} '{c[key]}' missing"
        for key in keys
        if c.get(key) is not None and c[key] not in dataset.metadata.columns
    ]
    if gene:
        found += _gene_problems(dataset.adata, gene)
    if _is_stale(plotter, spec):
        found.append("data changed since pin")
    if spec.get("backend") == "jscatter" and c.get("n_dimensions") == "2D":
        found.append("zoom not saved (jscatter)")
    return found


def restore(plotter, spec: dict) -> list[str]:
    """Bring the plotter back to ``spec``; returns warnings (empty: exact).

    The controls are set in one broker batch, so the figure is redrawn once,
    then the saved zoom/camera is applied (each redraw resets it). The zoom is
    skipped if the data changed since the pin (history or cell count). A gene
    view plots its gene again, as the Gene Expression step would.
    """
    if spec["source"] == "custom":
        return ["snapshots of custom figures cannot be restored"]
    if plotter.state.get("drawing"):
        return ["finish drawing before restoring a view"]
    values = spec["controls"]
    if values.get("data_key") not in plotter.dataset.data_dict:
        return [f"{values.get('data_key')} missing"]

    controls, broker = plotter.controls.dict, plotter.broker
    names = [k for k in values if k in controls and hasattr(controls[k], "value")]
    middle = [k for k in names if k not in FIRST and k not in LAST]
    order = [k for k in FIRST + middle + LAST if k in names]
    gene = spec.get("gene")
    warnings = _gene_problems(plotter.dataset.adata, gene) if gene else []
    with broker.batch():
        for name in order:
            widget, value = controls[name], values[name]
            if widget.value == value:
                continue
            try:
                widget.value = value
            except TraitError:
                if name in ("color", "sizeby"):
                    widget.value = None
                    warnings.append(f"{name} '{value}' missing")
                else:
                    warnings.append(f"{name}={value!r} unavailable")
        min_size = spec.get("settings", {}).get("min_size")
        if min_size is not None and plotter.min_size_input.value != min_size:
            plotter.min_size_input.value = min_size
        # also clears other gene/custom plots and extra traces
        if gene and not warnings:
            metadata = plotter.dataset.metadata
            metadata[gene["gene"]] = gene_values(
                plotter.dataset.adata, gene["gene"], gene["layer"]
            )
            plotter.state["gene"] = dict(gene)
            plotter.make_new_figure(
                metadata=metadata,
                colorby=gene["gene"],
                color_continuous_scale=gene["colorscale"],
            )
        else:
            plotter.make_new_figure()
    broker.flush(plotter.make_new_figure)  # if called within an outer batch

    ranges = spec.get("ranges")
    if ranges and _is_stale(plotter, spec):
        warnings.append("data changed since pin: zoom not restored")
    elif ranges and plotter._view_shown():
        warnings.append(f"zoom not restored ({plotter.backend} plot)")
    elif ranges:
        if "camera" in ranges:
            relayout = {"scene.camera": ranges["camera"]}
        else:
            relayout = {f"{a}axis.range": r for a, r in ranges.items() if r}
        try:
            plotter.g.plotly_relayout(relayout)
        except ValueError:
            warnings.append("saved zoom is invalid")
    if spec.get("dragmode"):
        plotter.set_dragmode(spec["dragmode"])
    return warnings


def _plot_data(dataset, spec: dict) -> tuple[pd.DataFrame, list[str], str | None]:
    """The cells as the plotter draws ``spec``: the data, its axis columns
    (after rank and log axes) and the color column."""
    c, gene = spec["controls"], spec.get("gene")
    data = dataset.data_dict[c["data_key"]]
    axes = [c[f"selected_axes_{i + 1}"] for i in range(int(c["n_dimensions"][0]))]
    if c.get("rotate_steps") and data.shape[1] >= 3 and c.get("selected_axes_3"):
        data = rotate_multiple_steps(
            data[[c[f"selected_axes_{i}"] for i in (1, 2, 3)]], c["rotate_steps"]
        )
    metadata = dataset.metadata
    colorby = c.get("color")
    if gene:
        colorby = gene["gene"]
        metadata[colorby] = gene_values(dataset.adata, colorby, gene["layer"])
    df = data.join(metadata.loc[:, ~metadata.columns.isin(data.columns)])
    if axes[0] == "rank":
        df = df.sort_values(axes[1], ascending=False, na_position="last")
        df["rank"] = np.arange(len(df)) + 1
    for i, col in enumerate(axes):
        if c.get(f"log_axes_{i + 1}"):
            df = df[df[col] > 0]
            axes[i] = "log " + col
            df[axes[i]] = np.log10(df[col])
    return df, axes, colorby


def view_frame(plotter, spec: dict, max_cells: int | None = None) -> dict:
    """``spec``'s plot as a frame for :func:`render_view` (see ``_backends``).

    No selection; the zoom only if the data did not change since the pin.
    """
    df, axes, colorby = _plot_data(plotter.dataset, spec)
    if max_cells and len(df) > max_cells:
        # shortcut: a uniform subsample hides rare cells; fine at thumbnail size
        keep = np.random.default_rng(0).choice(len(df), max_cells, replace=False)
        df = df.iloc[np.sort(keep)]
    c, ndims = spec["controls"], len(axes)
    frame = make_frame(
        df,
        axes[0],
        axes[min(1, ndims - 1)],
        colorby,
        {**c, "show_density": False},  # drawn by render_view, maybe off the main thread
        colorscale=(spec.get("gene") or {}).get("colorscale"),
        min_size=spec.get("settings", {}).get("min_size", plotter.min_size_input.value),
        title=c["data_key"],
    )
    frame.update(ndims=ndims, nbins=c.get("histogram_nbins"))
    frame["show_density"] = ndims == 2 and c.get("show_density")
    if ndims == 3:
        frame.update(z=df[axes[2]].to_numpy(dtype=float), z_title=axes[2])
    ranges = None if _is_stale(plotter, spec) else spec.get("ranges")
    if ranges and "camera" in ranges:
        frame["camera"] = ranges["camera"].get("eye")
    elif ranges:
        frame["ranges"] = ranges
    return frame


def render_view(frame: dict, controls: dict, **kwargs) -> bytes:
    """``render_png(frame, **kwargs)``, with the density lines of ``controls``."""
    if frame.pop("show_density", False):
        xy = np.column_stack([frame["x"], frame["y"]])
        frame["density"] = density_lines(xy, controls)
    return render_png(frame, **kwargs)


def _array(value) -> np.ndarray:
    """A trace array; plotly>=6 widgets hold typed arrays as base64 dicts."""
    if isinstance(value, dict) and "bdata" in value:
        a = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        return a.reshape([int(n) for n in str(shape).split(",")]) if shape else a
    return np.asarray(value)


def figure_png(
    fig: go.Figure, dpi: float = 100, minimal: bool = False, format: str = "png"
) -> bytes:
    """A matplotlib image of a plotly figure's 2D scatter traces (and lines).

    For custom-figure snapshots; other trace types are left out.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    layout = fig.layout
    out = Figure(figsize=((layout.width or 900) / 100, (layout.height or 600) / 100))
    ax = out.add_subplot()
    drawn = 0
    for i, t in enumerate(fig.data):
        if t.type not in ("scatter", "scattergl") or t.x is None or t.y is None:
            continue
        x, y, mode = _array(t.x), _array(t.y), t.mode or "markers"
        default = PALETTE[i % len(PALETTE)]
        label = t.name if t.name and t.showlegend is not False else None
        if "lines" in mode:
            ax.plot(x, y, color=_hex(t.line.color or default), linewidth=1, label=label)
            label = None  # one legend entry per trace
        if "markers" in mode:
            color, cmap = t.marker.color, None
            if color is None or isinstance(color, str):
                color = _hex(color or default)
            else:
                color = _array(color).astype(float)
                scale = t.marker.colorscale
                if scale is None and t.marker.coloraxis:
                    scale = layout[t.marker.coloraxis].colorscale
                scale = [[p, c] for p, c in (scale or layout.colorscale.sequential)]
                cmap = ListedColormap([_hex(c) for c in sample_colorscale(scale, 256)])
            size = t.marker.size if isinstance(t.marker.size, int | float) else 6
            ax.scatter(
                x,
                y,
                c=color,
                cmap=cmap,
                s=(size * 0.72) ** 2,
                linewidths=0,
                alpha=t.opacity,
                label=label,
            )
        drawn += 1
    if not drawn:
        ax.text(0.5, 0.5, "no preview", ha="center", va="center", color="#888")
    for axis, set_lim in (("xaxis", ax.set_xlim), ("yaxis", ax.set_ylim)):
        if layout[axis].range and layout[axis].type not in ("log", "category"):
            set_lim(*layout[axis].range)
    ax.spines[["top", "right"]].set_visible(False)
    if minimal:
        ax.set_xticks([])
        ax.set_yticks([])
        out.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    else:
        ax.set_xlabel(layout.xaxis.title.text or "")
        ax.set_ylabel(layout.yaxis.title.text or "")
        ax.set_title(layout.title.text or "")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        out.tight_layout()
    buffer = io.BytesIO()
    out.savefig(buffer, format=format, dpi=dpi)
    return buffer.getvalue()


def placeholder(text: str, format: str = "png") -> bytes:
    """A thumbnail-sized image showing ``text``."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(THUMB_WIDTH / 100, THUMB_WIDTH / 150), dpi=100)
    fig.text(0.5, 0.5, text, ha="center", va="center", color="#888", fontsize=14)
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format)
    return buffer.getvalue()


def thumb_format() -> str:
    """WebP (about half the bytes of PNG) if Pillow can write it."""
    from PIL import features

    return "webp" if features.check("webp") else "png"


def _short(value):
    """Floats with 6 significant digits, for readable code."""
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [_short(v) for v in value]
    if isinstance(value, dict):
        return {k: _short(v) for k, v in value.items()}
    return value


def view_code(dataset, spec: dict) -> str:
    """Python code drawing ``spec``'s plot with plotly express, from ``adata``.

    A readable approximation: no "Emphasize values" sizes, rotation or selection;
    the density lines are plotly's ``density_contour``.
    """
    from ..processor._codegen import Name, _imports, call, literal

    c, gene = spec["controls"], spec.get("gene")
    key, ndims = c["data_key"], int(c["n_dimensions"][0])
    axes = [c[f"selected_axes_{i + 1}"] for i in range(ndims)]
    data_axes = [a for a in axes if a != "rank"]
    colorby = gene["gene"] if gene else c.get("color")
    sizeby = c.get("sizeby") if ndims > 1 else None
    if key == "metadata":
        lines = [f"df = adata.obs[{literal(data_axes)}].copy()"]
    elif isinstance(dataset.adata.obsm[key], pd.DataFrame):
        lines = [f"df = adata.obsm[{literal(key)}][{literal(data_axes)}].copy()"]
    else:
        columns = list(dataset.data_dict[key].columns)
        rows = Name(
            f"adata.obsm[{literal(key)}][:, {[columns.index(a) for a in data_axes]}]"
        )
        lines = [
            "df = "
            + call(
                "pd.DataFrame", rows, index=Name("adata.obs_names"), columns=data_axes
            )
        ]
    if gene:
        X = "X" if gene["layer"] == "X" else f"layers[{literal(gene['layer'])}]"
        values = f"adata[:, {literal(colorby)}].{X}.sum(axis=1)"
        lines.append(f"df[{literal(colorby)}] = np.asarray({values}).ravel()")
    elif colorby and colorby not in data_axes:
        lines.append(f"df[{literal(colorby)}] = adata.obs[{literal(colorby)}]")
    if sizeby and sizeby not in data_axes and sizeby != colorby:
        sizes = f"adata.obs[{literal(sizeby)}].clip(lower=0).fillna(0)"
        lines.append(f"df[{literal(sizeby)}] = {sizes}")
    if axes[0] == "rank":
        lines.append(f"df = df.sort_values({literal(axes[1])}, ascending=False)")
        lines.append('df["rank"] = np.arange(len(df)) + 1')
    if c.get("rotate_steps"):
        lines.append(f"# not included: the rotation {c['rotate_steps']!r}")

    kwargs = dict(zip("xyz", axes))
    series = None if gene or not colorby else dataset.metadata[colorby]
    if series is not None and is_bool_dtype(series):
        kwargs["color"] = colorby
        kwargs["color_discrete_map"] = {True: PALETTE[0], False: NA_COLOR}
    elif series is not None and isinstance(series.dtype, pd.CategoricalDtype):
        kwargs["color"] = colorby
        cats = series.cat.categories.tolist()
        kwargs["color_discrete_map"] = {
            cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(cats)
        }
    elif colorby and ndims > 1 and (gene or is_numeric_dtype(series)):
        kwargs["color"] = colorby
        if gene:
            kwargs["color_continuous_scale"] = gene["colorscale"]
    log = {
        f"log_{a}": True
        for i, a in enumerate("xyz"[:ndims])
        if c.get(f"log_axes_{i + 1}")
    }
    kwargs.update(log)
    if sizeby:
        kwargs.update(size=sizeby, size_max=c["marker_size"])
    if ndims == 1:
        kwargs["nbins"] = c["histogram_nbins"]
    elif c["marker_opacity"] < 1:
        kwargs["opacity"] = c["marker_opacity"]
    if ndims == 2:
        kwargs["render_mode"] = "webgl"
    kwargs.update(title=key, template="simple_white", height=c["plot_height"])
    if c.get("plot_width"):
        kwargs["width"] = c["plot_width"]
    func = {1: "px.histogram", 2: "px.scatter", 3: "px.scatter_3d"}[ndims]
    lines.append("fig = " + call(func, Name("df"), **kwargs))
    if ndims > 1 and not sizeby:
        lines.append(f"fig.update_traces(marker_size={c['marker_size']!r})")
    if ndims == 2 and c.get("show_density"):
        density = call("px.density_contour", Name("df"), x=axes[0], y=axes[1], **log)
        lines.append(f"density = {density}")
        lines.append('fig.add_traces(density.update_traces(line_color="#555555").data)')
    if ndims == 2 and c.get("aspect_equal"):
        lines.append('fig.update_yaxes(scaleanchor="x", scaleratio=1)')
    elif ndims == 3 and c.get("aspect_equal"):
        lines.append('fig.update_layout(scene_aspectmode="data")')
    ranges = spec.get("ranges") or {}
    if "camera" in ranges:
        lines.append(call("fig.update_layout", scene_camera=_short(ranges["camera"])))
    elif any(ranges.values()):
        zoom = {f"{a}axis_range": _short(r) for a, r in ranges.items() if r}
        lines.append(call("fig.update_layout", **zoom))
    lines.append("fig")

    code = "\n".join(lines)
    header = [
        f"# {spec['name']}: the pinned view, drawn with plotly express",
        "# adata: the dashboard's AnnData (dashboard.dataset.adata)",
        *_imports(code),
        "import plotly.express as px",
    ]
    return "\n".join(header) + "\n\n" + code + "\n"


class ViewGallery(VBox):
    """Strip of pinned-view cards: thumbnail, restore, rename, reorder, delete,
    code and PNG download."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.views: list[dict] = loads_uns(plotter.dataset.adata)
        self._cards: dict[str, dict] = {}  # view id -> its card's widgets
        self._figures: dict[str, go.Figure] = {}  # custom views' snapshots
        self._thumbs: dict[str, object] = {}  # view id -> what its thumbnail shows
        self._pool: ThreadPoolExecutor | None = None
        self._code_id: str | None = None  # the view whose code is shown
        self.pin_button = Button(
            description="Pin view",
            icon="thumb-tack",
            tooltip="Save the current plot view (not the selection)",
            layout=Layout(width="auto", margin="2px"),
        )
        self.pin_button.on_click(lambda _: self.pin())
        self.dpi = BoundedIntText(
            value=200,
            min=50,
            max=1200,
            step=50,
            description="PNG dpi",
            tooltip="Resolution of the PNG downloads (100: as on screen)",
            layout=Layout(width="125px", margin="2px"),
            style={"description_width": "initial"},
        )
        self.header = VBox([self.pin_button, self.dpi], layout=Layout(flex="0 0 auto"))
        self.cards = HBox(layout=Layout(width="100%", overflow="auto"))
        self.message = HTML()
        self.code = CodeActions(layout=Layout(display="none"))
        self.code_view = HTML(layout=Layout(display="none"))
        VBox.__init__(
            self,
            [self.cards, self.message, self.code, self.code_view],
            layout=Layout(width="100%", min_width="0"),
        )
        # after a step's batch: its history entry is recorded by then
        self.render = plotter.broker.coalesce(self._render)
        for event in [
            "dset_data_dict_change",
            "dset_metadata_change",
            "dset_total_rows_change",
        ]:
            plotter.broker.subscribe(event, self.render)
        self._render()

    def _spec(self, view_id: str) -> dict:
        return next(v for v in self.views if v["id"] == view_id)

    def pin(self, name: str | None = None) -> dict | None:
        custom = self.plotter.state.get("source") == "custom"
        name = name or f"{'Snapshot' if custom else 'View'} {len(self.views) + 1}"
        try:
            spec = capture(self.plotter, name)
        except ValueError as e:
            self._say(str(e))
            return None
        if custom:
            self._figures[spec["id"]] = go.Figure(self.plotter.g)
        self.views.append(spec)
        self._sync()
        self._say("")
        return spec

    def restore(self, view_id: str) -> list[str]:
        warnings = restore(self.plotter, self._spec(view_id))
        self._say("; ".join(warnings))
        return warnings

    def rename(self, view_id: str, name: str):
        self._spec(view_id)["name"] = name
        self._sync()

    def delete(self, view_id: str):
        self.views = [v for v in self.views if v["id"] != view_id]
        self._figures.pop(view_id, None)
        if self._code_id == view_id:
            self.show_code(view_id)  # hides it
        self._sync()

    def move(self, view_id: str, offset: int):
        """Move a view's card ``offset`` places (negative: to the left)."""
        i = self.views.index(self._spec(view_id))
        j = min(max(i + offset, 0), len(self.views) - 1)
        self.views.insert(j, self.views.pop(i))
        self._sync()

    def code_for(self, view_id: str) -> str:
        return view_code(self.plotter.dataset, self._spec(view_id))

    def show_code(self, view_id: str):
        """Show the view's code with Copy/Download buttons; again: hide it."""
        hide = self._code_id == view_id
        self._code_id = None if hide else view_id
        for widget in (self.code, self.code_view):
            widget.layout.display = "none" if hide else ""
        if hide:
            return
        try:
            code = self.code_for(view_id)
        except Exception as e:  # e.g. its data is gone
            code = f"# no code: {type(e).__name__}: {e}"
        name = re.sub(r"[^\w.-]+", "_", self._spec(view_id)["name"])
        self.code.text, self.code.filename = code, f"{name}.py"
        self.code_view.value = (
            "<pre style='font-family:monospace;font-size:12px;user-select:text;"
            f"white-space:pre;overflow:auto;margin:0'>{escape(code)}</pre>"
        )

    def _image_work(self, view_id: str, dpi: float, max_cells=None, **kwargs):
        """The view's image as a zero-argument callable.

        The data is read now (the main thread's); the callable only draws, so it
        can run in a worker thread. ``kwargs``: of :func:`render_png`.
        """
        spec = self._spec(view_id)
        if spec["source"] == "custom":
            return partial(figure_png, self._figures[view_id], dpi, **kwargs)
        frame = view_frame(self.plotter, spec, max_cells)
        return partial(render_view, frame, spec["controls"], dpi=dpi, **kwargs)

    def png(self, view_id: str, dpi: float | None = None) -> bytes:
        """The view as a full-size PNG (``dpi`` 100: as on screen)."""
        return self._image_work(view_id, dpi or self.dpi.value)()

    def download_png(self, view_id: str):
        """Download the view as a PNG of the "PNG dpi" resolution, drawn off the
        main thread in a kernel."""
        name = f"{self.plotter.dataset.name}_{self._spec(view_id)['name']}"
        filename = re.sub(r"[^\w.-]+", "_", name) + ".png"

        def save(result):
            try:
                png = result()
            except Exception as e:
                self._say(f"PNG failed: {type(e).__name__}: {e}")
                return
            self.plotter.download.save(png, filename, "image/png")
            self._say("")

        try:
            work = self._image_work(view_id, self.dpi.value)
        except Exception as e:  # e.g. its data is gone
            self._say(f"PNG failed: {type(e).__name__}: {e}")
            return
        self._say(f"Drawing {filename}...")
        self._offload(work, save)

    def _offload(self, work, done):
        """``done(result)``, with ``result()`` returning ``work()`` or raising.

        In a kernel, ``work`` runs in a worker thread and ``done`` on the kernel's
        IOLoop (widgets are not thread-safe); otherwise both inline.
        """
        from ...event._jobs import _kernel_loop  # circular import

        loop = _kernel_loop()
        if loop is None:
            done(work)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="sclab-views")
        future = self._pool.submit(work)
        # add_callback is thread-safe
        future.add_done_callback(lambda f: loop.add_callback(done, f.result))

    def _sync(self):
        dumps_uns(self.plotter.dataset.adata, self.views)
        self._render()
        self.plotter.broker.publish("dset_state_change")

    def _say(self, text: str):
        style = "font-size:12px;color:#b45309"
        self.message.value = f"<span style='{style}'>{escape(text)}</span>"

    def _render(self, *_, **__):
        # cards are kept per view: only their badges and thumbnails follow the data
        for view_id in set(self._cards) - {v["id"] for v in self.views}:
            self._thumbs.pop(view_id, None)
            for widget in self._cards.pop(view_id).values():
                widget.close()
        last = len(self.views) - 1
        for i, spec in enumerate(self.views):
            if spec["id"] not in self._cards:
                self._cards[spec["id"]] = self._card(spec)
            card = self._cards[spec["id"]]
            card["left"].disabled, card["right"].disabled = i == 0, i == last
            if spec["source"] != "custom":
                card["restore"].disabled = (
                    spec["controls"].get("data_key")
                    not in self.plotter.dataset.data_dict
                )
            style = "font-size:11px;color:#b45309"
            found = escape("; ".join(problems(self.plotter, spec)))
            card["badge"].value = f"<span style='{style}'>{found}</span>"
            self._thumbnail(spec)
        cards = [self._cards[v["id"]]["card"] for v in self.views]
        self.cards.children = [self.header, *cards]

    def _thumbnail(self, spec: dict):
        """Render the view's thumbnail if it has none or the data changed."""
        view_id, custom = spec["id"], spec["source"] == "custom"
        p = self.plotter
        key = "snapshot" if custom else (p.data_hash(), p.dataset.adata.n_obs)
        if self._thumbs.get(view_id) == key:
            return
        self._thumbs[view_id] = key  # also while rendering: one job per change
        fmt = thumb_format()
        if custom:
            width = self._figures[view_id].layout.width
        else:
            width = spec["controls"].get("plot_width")
        dpi = 100 * THUMB_WIDTH / (width or 900)
        try:
            kwargs = dict(minimal=True, format=fmt)
            if not custom:
                kwargs["max_cells"] = THUMB_MAX_CELLS
            work = self._image_work(view_id, dpi, **kwargs)
        except Exception:  # e.g. its data or gene is gone
            work = partial(placeholder, "no preview", fmt)
        self._offload(work, partial(self._show_thumbnail, view_id, key, fmt))

    def _show_thumbnail(self, view_id: str, key, fmt: str, result):
        if self._thumbs.get(view_id) != key or view_id not in self._cards:
            return  # deleted, or a newer one is rendering
        try:
            image = result()
        except Exception:
            image = placeholder("no preview", fmt)
        thumb = self._cards[view_id]["thumb"]
        thumb.format, thumb.value = fmt, image

    def _card(self, spec: dict) -> dict:
        c, view_id, custom = spec["controls"], spec["id"], spec["source"] == "custom"
        name = Text(spec["name"], continuous_update=False, layout=Layout(width="94px"))
        name.observe(lambda ch: self.rename(view_id, ch["new"]), names="value")
        small = dict(layout=Layout(width="28px", padding="0", margin="1px"))
        left = Button(icon="chevron-left", tooltip="Move left", **small)
        right = Button(icon="chevron-right", tooltip="Move right", **small)
        delete = Button(icon="trash", tooltip="Delete", **small)
        left.on_click(lambda _: self.move(view_id, -1))
        right.on_click(lambda _: self.move(view_id, 1))
        delete.on_click(lambda _: self.delete(view_id))
        thumb = Image(layout=Layout(width=f"{CARD_WIDTH}px", height="auto"))
        if custom:
            summary = f"Snapshot: {spec.get('title') or 'custom figure'}"
            tooltip = (
                "Custom figure: PNG only, cannot be restored, not saved in the file"
            )
        else:
            color = spec["gene"]["gene"] if spec.get("gene") else c.get("color")
            summary = " · ".join(
                str(v) for v in (c.get("data_key"), c.get("n_dimensions"), color) if v
            )
            tooltip = f"Restore: {summary}"
        restore_button = Button(
            description=summary,
            tooltip=tooltip,
            disabled=custom,
            layout=Layout(width=f"{CARD_WIDTH}px", margin="1px 0"),
        )
        restore_button.on_click(lambda _: self.restore(view_id))
        half = Layout(width=f"{CARD_WIDTH // 2 - 2}px", margin="1px")
        code = Button(
            description="Code",
            icon="code",
            tooltip="Python code drawing this plot",
            disabled=custom,
            layout=half,
        )
        code.on_click(lambda _: self.show_code(view_id))
        png = Button(
            description="PNG", icon="download", tooltip="Download as PNG", layout=half
        )
        png.on_click(lambda _: self.download_png(view_id))
        badge = HTML(layout=Layout(width=f"{CARD_WIDTH}px"))
        row = HBox([name, left, right, delete])
        actions = HBox([code, png])
        card = VBox(
            [row, thumb, restore_button, actions, badge],
            layout=Layout(
                flex="0 0 auto", border="1px solid #ddd", margin="2px", padding="2px"
            ),
        )
        return dict(
            card=card,
            row=row,
            name=name,
            left=left,
            right=right,
            delete=delete,
            thumb=thumb,
            restore=restore_button,
            actions=actions,
            code=code,
            png=png,
            badge=badge,
        )
