"""2D plot backends other than plotly: jupyter-scatter (``jscatter``) and ``static``.

The :class:`~sclab.dataset.plotter.Plotter` stays in charge: it computes a
*frame* (what to draw, see :func:`make_frame`) and calls ``view.draw(frame)``
whenever the 2D plot changes. A frame is a dict::

    index     cell ids (pd.Index), the order of x, y and every array below
    x, y      float arrays; x_title, y_title, title: str
    color     None | {"kind": "categorical", "values": pd.Categorical,
                      "colors": {label: hex}}
                   | {"kind": "continuous", "values": float array,
                      "colors": [hex, ...], "range": (vmin, vmax)}
    color_title    the color column
    size      point diameters in px: a float or a float array (as plotly draws)
    marker_size    the Marker size control: the scale of the selection sizes
    selected  bool array of the dataset selection, or None
    opacity, height, width (0: auto), aspect_equal, hover
    density   None | list of (n, 2) arrays: the density contour lines

1D and 3D plots, path drawing, custom figures and extra traces always use plotly.
"""

import base64
import io

import numpy as np
import pandas as pd
import plotly.io as pio
from ipywidgets import Box, Image, Layout
from pandas.api.types import (
    is_any_real_numeric_dtype,
    is_bool_dtype,
    is_numeric_dtype,
)
from plotly.colors import get_colorscale, hex_to_rgb, sample_colorscale, unlabel_rgb
from scipy.stats import rankdata

from ._utils import COLOR_DISCRETE_SEQUENCE, _make_density_heatmap

BACKENDS = ("plotly", "jscatter", "static")
NA_COLOR = "lightgray"


def _hex(color: str) -> str:
    """plotly's "rgb(r, g, b)" as "#rrggbb" (matplotlib does not parse rgb())."""
    if not color.startswith("rgb"):
        return color
    return "#%02x%02x%02x" % tuple(round(float(v)) for v in unlabel_rgb(color))


PALETTE = [_hex(c) for c in COLOR_DISCRETE_SEQUENCE]


def color_spec(series: pd.Series, colorscale=None) -> dict | None:
    """How a color column is drawn: the plotly backend's colors and NA handling."""
    if is_bool_dtype(series):
        labels = series.map({True: "True", False: "False"})
        colors = {"True": PALETTE[0], "False": NA_COLOR}
    elif isinstance(series.dtype, pd.CategoricalDtype):
        labels = (
            series.astype(object).where(series.notna()).map(str, na_action="ignore")
        )
        colors = {
            str(c): PALETTE[i % len(PALETTE)]
            for i, c in enumerate(series.cat.categories)
        }
    elif is_numeric_dtype(series):
        if colorscale is None:  # px's default: the template's
            colorscale = pio.templates["simple_white"].layout.colorscale.sequential
        elif isinstance(colorscale, str):
            colorscale = get_colorscale(colorscale)
        # sample_colorscale takes "rgb()" colors only
        colorscale = [
            [pos, c if c.startswith("rgb") else "rgb(%d, %d, %d)" % hex_to_rgb(c)]
            for pos, c in colorscale
        ]
        values = series.to_numpy(dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        return {
            "kind": "continuous",
            "values": values,
            "colors": [_hex(c) for c in sample_colorscale(colorscale, 256)],
            "range": (float(vmin), float(vmax) if vmax > vmin else float(vmin) + 1),
        }
    else:
        return None

    labels = labels.fillna("NA").astype(str)
    colors["NA"] = NA_COLOR
    present = [c for c in colors if c in set(labels)]  # legend: drawn labels only
    return {
        "kind": "categorical",
        "values": pd.Categorical(labels.to_numpy(), categories=present),
        "colors": {c: colors[c] for c in present},
    }


def value_size_factors(values, min_factor: float = 0.3) -> np.ndarray:
    """Marker diameters relative to the marker size, emphasizing the cells with values.

    Missing values and values <= 0 get ``min_factor``; positive values are scaled
    by their rank (percentile among the positive values) from 0.6 to 1.5.
    """
    values = pd.Series(values).to_numpy(dtype=float, na_value=np.nan)
    factors = np.full(values.shape, float(min_factor))
    positive = values > 0  # False for NaN
    if positive.any():
        percentile = (rankdata(values[positive]) - 0.5) / positive.sum()
        factors[positive] = 0.6 + 0.9 * percentile
    return factors


def size_factors(df, colorby, sizeby, emphasize: bool, min_size: float):
    """Per-cell marker diameters relative to the marker size (None: uniform).

    See ``Plotter._size_factors``: only for a numeric color column, without "Size
    by", with "Emphasize values" on.
    """
    if sizeby is not None or not colorby or not emphasize:
        return None
    if colorby not in df or not is_any_real_numeric_dtype(df[colorby]):
        return None  # categorical or boolean colors
    return value_size_factors(df[colorby], min_size)


def density_lines(xy: np.ndarray, values: dict) -> list[np.ndarray]:
    """The density contour lines plotly draws, from the same grid.

    ``values``: the plot control values (``PlotterControls.dict`` names).
    """
    import contourpy  # a matplotlib dependency

    trace = _make_density_heatmap(
        xy,
        values["density_bandwidth_factor"],
        values["density_grid_resolution"],
        values["density_line_smoothing"],
        values["density_contours"],
    )
    x, y, z = (np.asarray(trace[k], dtype=float) for k in "xyz")
    generator = contourpy.contour_generator(x, y, z)
    levels = np.linspace(z.min(), z.max(), values["density_contours"] + 1)[1:-1]
    return [line for level in levels for line in generator.lines(level)]


def make_frame(
    df,
    col_x: str,
    col_y: str,
    colorby: str | None,
    values: dict,
    sizeby: str | None = None,
    scale: float | None = None,
    colorscale=None,
    min_size: float = 0.3,
    selected: np.ndarray | None = None,
    title: str = "",
) -> dict:
    """A 2D frame (see the module docstring) of ``df``'s columns.

    ``values``: the plot control values (``PlotterControls.dict`` names); ``sizeby``
    and ``scale`` override theirs.
    """
    sizeby = values["sizeby"] if sizeby is None else sizeby
    scale = values["marker_size"] if scale is None else scale
    size = scale  # px diameter, as plotly draws marker.size with sizeref
    if sizeby is not None:
        sizes = df[sizeby].astype(float).clip(lower=0).fillna(0.0).to_numpy()
        if sizes.max() > 0:
            size = scale * np.sqrt(sizes / sizes.max())
    elif (
        factors := size_factors(
            df, colorby, sizeby, values["emphasize_values"], min_size
        )
    ) is not None:
        size = scale * factors
    x, y = (df[col].to_numpy(dtype=float) for col in (col_x, col_y))
    density = None
    if values["show_density"]:
        density = density_lines(np.column_stack([x, y]), values)
    return {
        "index": df.index,
        "x": x,
        "y": y,
        "x_title": col_x,
        "y_title": col_y,
        "title": title,
        "color": None if not colorby else color_spec(df[colorby], colorscale),
        "color_title": colorby,
        "size": size,
        "marker_size": scale,
        "selected": selected,
        "opacity": values["marker_opacity"],
        "density": density,
        "height": values["plot_height"],
        "width": values["plot_width"],
        "aspect_equal": values["aspect_equal"],
        "hover": values["enable_hover_info"] and col_x != "rank",
    }


def render_png(
    frame: dict, dpi: float = 100, minimal: bool = False, format: str = "png"
) -> bytes:
    """The frame as a matplotlib image (no pyplot: no global figure state).

    The figure is ``frame["width"] x frame["height"]`` px at 100 dpi, as on
    screen; ``dpi`` scales the image. ``minimal`` (thumbnails): no legend, color
    bar, labels or ticks. Optional frame keys, for pinned views: ``ndims`` (1: a
    histogram of x, ``nbins`` bins; 3: x, y and ``z``, seen from ``camera``, a
    plotly scene camera) and ``ranges`` (``{"x": [lo, hi], "y": ...}``, the zoom).
    """
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure

    width = frame["width"] or 900
    fig = Figure(figsize=(width / 100, frame["height"] / 100), dpi=100)
    ndims = frame.get("ndims", 2)
    ax = fig.add_subplot(projection="3d" if ndims == 3 else None)
    coords = [frame["x"], frame["y"]] + ([frame["z"]] if ndims == 3 else [])
    x = coords[0]
    diameter = np.broadcast_to(np.asarray(frame["size"], dtype=float), x.shape)
    if frame["selected"] is not None:  # as plotly: the selection sets the size
        area = np.where(frame["selected"], 3.0, 0.5)  # Plotter.update_marker_sizes
        diameter = frame["marker_size"] * np.sqrt(area)
    common = dict(s=(diameter * 72 / 100) ** 2, alpha=frame["opacity"], linewidths=0)

    spec = frame["color"]
    if ndims == 1:
        categorical = spec is not None and spec["kind"] == "categorical"
        labels = list(spec["colors"]) if categorical else [None]
        ax.hist(
            [x[spec["values"] == c] for c in labels] if categorical else [x],
            bins=frame.get("nbins") or 10,
            stacked=True,
            color=[spec["colors"][c] for c in labels] if categorical else PALETTE[0],
            label=labels,
        )
        if categorical and not minimal:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    elif spec is None:
        ax.scatter(*coords, c=PALETTE[0], **common)
    elif spec["kind"] == "categorical":
        codes = spec["values"].codes
        order = sorted(spec["colors"], key=lambda c: c != "NA")  # NA drawn first
        for label in order:
            i = spec["values"].categories.get_loc(label)
            mask = codes == i
            ax.scatter(
                *(v[mask] for v in coords),
                c=spec["colors"][label],
                label=label,
                **{**common, "s": common["s"][mask]},
            )
        if not minimal:
            legend = ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9
            )
            # handles are named legendHandles before matplotlib 3.7
            handles = getattr(legend, "legend_handles", None) or legend.legendHandles
            for handle in handles:
                handle.set_sizes([30])
                handle.set_alpha(1)
    else:
        values = spec["values"]
        order = np.argsort(np.nan_to_num(values, nan=-np.inf), kind="stable")
        cmap = ListedColormap(spec["colors"])
        cmap.set_bad(NA_COLOR)
        sc = ax.scatter(
            *(v[order] for v in coords),
            c=values[order],
            cmap=cmap,
            norm=Normalize(*spec["range"]),
            plotnonfinite=True,
            **{**common, "s": common["s"][order]},
        )
        if not minimal:
            fig.colorbar(sc, ax=ax)

    if ndims == 2:
        for line in frame["density"] or []:
            ax.plot(line[:, 0], line[:, 1], color="#555555", linewidth=0.8)

    ranges = frame.get("ranges") or {}
    limits = [ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if ndims == 3 else [])
    for axis, set_lim, v in zip("xyz", limits, coords):
        if ranges.get(axis):
            set_lim(*ranges[axis])
        elif ndims > 1 or axis == "x":
            lo, hi = np.nanmin(v), np.nanmax(v)
            pad = (hi - lo) * 0.1 or 1.0
            set_lim(lo - pad, hi + pad)
    if ndims == 3:
        eye = {"x": 1.25, "y": 1.25, "z": 1.25, **(frame.get("camera") or {})}
        ax.view_init(  # plotly's eye: the camera position seen from the center
            elev=np.degrees(np.arctan2(eye["z"], np.hypot(eye["x"], eye["y"]))),
            azim=np.degrees(np.arctan2(eye["y"], eye["x"])),
        )
        spans = [np.ptp(np.nan_to_num(v)) or 1.0 for v in coords]
        ax.set_box_aspect(spans if frame["aspect_equal"] else (1, 1, 1))
    elif frame["aspect_equal"]:
        ax.set_aspect("equal", adjustable="datalim")
    if minimal:
        ax.set_xticks([])
        ax.set_yticks([])
        if ndims == 3:
            ax.set_zticks([])
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    else:
        ax.set_xlabel(frame["x_title"])
        ax.set_ylabel("count" if ndims == 1 else frame["y_title"])
        if ndims == 3:
            ax.set_zlabel(frame["z_title"])
        ax.set_title(frame["title"] or "")
        fig.tight_layout()
    if ndims < 3:
        ax.spines[["top", "right"]].set_visible(False)
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi)
    return buffer.getvalue()


def png_html(png: bytes, title: str) -> str:
    """A standalone HTML page showing ``png``, for the plotter's Save button."""
    data = base64.b64encode(png).decode()
    return f"<html><title>{title}</title><img src='data:image/png;base64,{data}'>"


class StaticView:
    """Non-interactive matplotlib image of the 2D plot."""

    name = "static"

    def __init__(self, plotter):
        self.plotter = plotter
        self.frame: dict | None = None
        self.widget = Image(format="png", layout=Layout(max_width="100%"))

    def draw(self, frame: dict):
        self.frame = frame
        self.widget.value = render_png(frame)

    def close(self):
        self.widget.close()


def _equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(
            a, b, equal_nan=a.dtype.kind == "f"
        )
    if isinstance(a, pd.Categorical):
        return a.equals(b) and list(a.categories) == list(b.categories)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_equal(u, v) for u, v in zip(a, b))
    return a == b


class JscatterView:
    """WebGL 2D scatter by jupyter-scatter; lassos publish the plotly lasso's event.

    Only the encodings that changed since the last frame are sent. The selection
    shown is the dataset's (jscatter's own highlight); a lasso publishes
    ``dplt_selected_points_change`` and, when the lasso polygon reproduces the
    selection, records it as the plotter's ``selection_shape`` (as plotly lassos).
    The camera is not synced back by jupyter-scatter: pinned views keep no zoom.
    Its initial view fits the y range: a plot taller than wide starts cut off at
    the sides (zoom out with the mouse wheel).
    """

    name = "jscatter"

    def __init__(self, plotter):
        import jscatter  # ImportError: the plotter falls back to plotly

        self._jscatter = jscatter
        self.plotter = plotter
        self.scatter = None
        self.frame: dict | None = None
        self._sent: dict = {}  # encoding -> last value sent to the widget
        self._echo = False  # True while the selection is set from Python
        self._names: pd.Index | None = None  # cells of the last published selection
        # display=block: a flex Box shrinks jscatter's width="auto" canvas to nothing
        self.widget = Box(layout=Layout(width="100%", display="block"))

    def draw(self, frame: dict):
        old, self.frame = self.frame, frame
        same_points = (
            self.scatter is not None
            and frame["index"].equals(old["index"])
            and _equal(frame["x"], old["x"])
            and _equal(frame["y"], old["y"])
        )
        if same_points:  # one message: separate ones race in jscatter's redraws
            with self.scatter.widget.hold_sync():
                self._apply(frame)
            return
        self._create(frame)
        self._apply(frame)  # before the widget exists: drawn once, complete
        widget = self.scatter.widget
        widget.observe(self._on_selection, names="selection")
        # the frontend sends the polygon after the selection, as its own update
        widget.observe(self._on_polygon, names="lasso_selection_polygon")
        self.widget.children = [widget]

    def _apply(self, frame: dict):
        s = self.scatter
        if self._changed("color", frame["color"]):
            spec = frame["color"]
            if spec is None:
                s.color(by=None, default=PALETTE[0])
            elif spec["kind"] == "categorical":
                s.color(by=spec["values"], map=spec["colors"])
            else:
                s.color(by=spec["values"], map=spec["colors"], norm=spec["range"])
            s.legend(spec is not None)
        if self._changed("size", frame["size"]):
            size = frame["size"]
            lo, hi = np.min(size), np.max(size)
            if np.ndim(size) == 0 or lo == hi:
                s.size(by=None, default=float(lo))
            else:  # jscatter bins the sizes: 64 steps between the smallest and largest
                s.size(by=size, map=np.linspace(lo, hi, 64).tolist(), norm=(lo, hi))
        if self._changed("opacity", frame["opacity"]):
            s.opacity(default=frame["opacity"])
        if self._changed("density", frame["density"]):
            line = self._jscatter.Line
            lines = [
                line([tuple(p) for p in v], line_color="#555555")
                for v in frame["density"] or []
            ]
            s.annotations(lines)
        if self._changed("labels", [frame["x_title"], frame["y_title"]]):
            s.axes(labels=[frame["x_title"], frame["y_title"]])
        if self._changed("height", frame["height"]):
            s.height(frame["height"])
        if self._changed("hover", frame["hover"]):
            s.tooltip(enable=frame["hover"])
        if self._changed("selected", frame["selected"]):
            self._echo = True
            try:
                selected = frame["selected"]
                s.selection(None if selected is None else frame["index"][selected])
            finally:
                self._echo = False

    def _changed(self, key: str, value) -> bool:
        if key in self._sent and _equal(self._sent[key], value):
            return False
        self._sent[key] = value
        return True

    def _create(self, frame: dict):
        if self.scatter is not None:
            self.scatter.widget.close()
        df = pd.DataFrame({"x": frame["x"], "y": frame["y"]}, index=frame["index"])
        self.scatter = self._jscatter.Scatter(
            data=df,
            x="x",
            y="y",
            data_use_index=True,
            axes=True,
            background_color="white",
            lasso_initiator=True,
            opacity_unselected=0.3,
            opacity_by=None,  # not jscatter's density-based default: the control's
        )
        self._sent = {}

    def _lasso_shape(self, names: pd.Index | None) -> dict | None:
        """The last lasso polygon as a selection shape, if it selects ``names``."""
        polygon = self.scatter.widget.lasso_selection_polygon
        if names is None or polygon is None or len(polygon) < 3:
            return None
        path = (
            "M"
            + "L".join(f"{x!r},{y!r}" for x, y in np.asarray(polygon, float).tolist())
            + "Z"
        )
        plotter = self.plotter
        shape = plotter._selection_shape([{"type": "path", "path": path}])
        # merged/subtracted lassos, or cells on its edge, are recorded as a mask
        inside = None if shape is None else plotter._shape_points(shape)
        if inside is None or inside.size != names.size or not inside.isin(names).all():
            return None
        return shape

    def _on_selection(self, change):
        if self._echo:
            return
        idx = np.asarray([] if change["new"] is None else change["new"], dtype=int)
        names = self.frame["index"][idx] if idx.size else None
        self._names = names
        self.plotter.selection_shape = self._lasso_shape(names)
        self.plotter.broker.publish("dplt_selected_points_change", names)

    def _on_polygon(self, change):
        if self.plotter.selection_shape is not None or self._names is None:
            return
        if (shape := self._lasso_shape(self._names)) is not None:
            self.plotter.selection_shape = shape
            self.plotter.broker.publish("dplt_selected_points_change", self._names)

    def close(self):
        self.widget.close()
        if self.scatter is not None:
            self.scatter.widget.close()


def make_view(name: str, plotter):
    """A backend's 2D view; raises ImportError if jupyter-scatter is missing."""
    return {"jscatter": JscatterView, "static": StaticView}[name](plotter)
