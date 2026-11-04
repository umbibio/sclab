"""2D plot backends other than plotly: ``static`` (a matplotlib image).

The :class:`~sclab.dataset.plotter.Plotter` stays in charge: it computes a
*frame* (what to draw, see ``Plotter._frame_2d``) and calls ``view.draw(frame)``
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
from ipywidgets import Image, Layout
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from plotly.colors import get_colorscale, hex_to_rgb, sample_colorscale, unlabel_rgb

from ._utils import COLOR_DISCRETE_SEQUENCE, _make_density_heatmap

BACKENDS = ("plotly", "static")
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


def density_lines(xy: np.ndarray, controls) -> list[np.ndarray]:
    """The density contour lines plotly draws, from the same grid."""
    import contourpy  # a matplotlib dependency

    trace = _make_density_heatmap(
        xy,
        controls.density_bandwidth_factor.value,
        controls.density_grid_resolution.value,
        controls.density_line_smoothing.value,
        controls.density_contours.value,
    )
    x, y, z = (np.asarray(trace[k], dtype=float) for k in "xyz")
    generator = contourpy.contour_generator(x, y, z)
    levels = np.linspace(z.min(), z.max(), controls.density_contours.value + 1)[1:-1]
    return [line for level in levels for line in generator.lines(level)]


def render_png(frame: dict) -> bytes:
    """The frame as a matplotlib PNG (no pyplot: no global figure state)."""
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure

    dpi = 100
    width = frame["width"] or 900
    fig = Figure(figsize=(width / dpi, frame["height"] / dpi), dpi=dpi)
    ax = fig.add_subplot()
    x, y = frame["x"], frame["y"]
    diameter = np.broadcast_to(np.asarray(frame["size"], dtype=float), x.shape)
    if frame["selected"] is not None:  # as plotly: the selection sets the size
        area = np.where(frame["selected"], 3.0, 0.5)  # Plotter.update_marker_sizes
        diameter = frame["marker_size"] * np.sqrt(area)
    common = dict(s=(diameter * 72 / dpi) ** 2, alpha=frame["opacity"], linewidths=0)

    spec = frame["color"]
    if spec is None:
        ax.scatter(x, y, c=PALETTE[0], **common)
    elif spec["kind"] == "categorical":
        codes = spec["values"].codes
        order = sorted(spec["colors"], key=lambda c: c != "NA")  # NA drawn first
        for label in order:
            i = spec["values"].categories.get_loc(label)
            mask = codes == i
            ax.scatter(
                x[mask],
                y[mask],
                c=spec["colors"][label],
                label=label,
                **{**common, "s": common["s"][mask]},
            )
        legend = ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9
        )
        # handles are named legendHandles before matplotlib 3.7
        for handle in getattr(legend, "legend_handles", None) or legend.legendHandles:
            handle.set_sizes([30])
            handle.set_alpha(1)
    else:
        values = spec["values"]
        order = np.argsort(np.nan_to_num(values, nan=-np.inf), kind="stable")
        cmap = ListedColormap(spec["colors"])
        cmap.set_bad(NA_COLOR)
        sc = ax.scatter(
            x[order],
            y[order],
            c=values[order],
            cmap=cmap,
            norm=Normalize(*spec["range"]),
            plotnonfinite=True,
            **{**common, "s": common["s"][order]},
        )
        fig.colorbar(sc, ax=ax)

    for line in frame["density"] or []:
        ax.plot(line[:, 0], line[:, 1], color="#555555", linewidth=0.8)

    for set_lim, v in ((ax.set_xlim, x), (ax.set_ylim, y)):
        lo, hi = np.nanmin(v), np.nanmax(v)
        pad = (hi - lo) * 0.1 or 1.0
        set_lim(lo - pad, hi + pad)
    if frame["aspect_equal"]:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(frame["x_title"])
    ax.set_ylabel(frame["y_title"])
    ax.set_title(frame["title"] or "")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
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


def make_view(name: str, plotter):
    """A backend's 2D view; raises ImportError if its package is missing."""
    return {"static": StaticView}[name](plotter)
