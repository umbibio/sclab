"""Pinned plot views: save the plotter's view and get back to it later.

A view spec is a JSON dict, independent of the plotting backend::

    {
        "v": 1,
        "id": "3f2a9c1e",
        "name": "View 1",
        "created": "2026-11-03T10:15:42+01:00",
        "controls": {"data_key": "X_umap", "n_dimensions": "2D", ...},
        "ranges": {"x": [x0, x1], "y": [y0, y1]} | {"camera": {...}} | None,
        "dragmode": "lasso",
        "history_hash": "<sha256 of the steps that changed the data>",
        "n_obs": 1000,
    }

``controls`` holds the value of every value widget in ``PlotterControls.dict``;
``ranges`` the frontend zoom/pan (1D/2D) or 3D camera, as synced into the
figure's layout. The active selection, drawn paths and extra traces are not
part of a view. Views are stored in ``adata.uns["sclab"]["views"]`` as a JSON
string, so they travel with the h5ad file.
"""

import json
from datetime import datetime
from html import escape
from uuid import uuid4

from ipywidgets import HTML, Button, HBox, Layout, Text, VBox
from traitlets import TraitError


UNS_KEY = "sclab"  # as in sclab.dataset.processor._history (circular import)

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
    """The views stored in ``adata.uns`` (malformed entries dropped)."""
    raw = adata.uns.get(UNS_KEY, {}).get("views")
    try:
        views = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []
    if not isinstance(views, list):
        return []
    return [
        v for v in views if isinstance(v, dict) and isinstance(v.get("controls"), dict)
    ]


def dumps_uns(adata, views: list[dict]) -> None:
    adata.uns.setdefault(UNS_KEY, {})["views"] = json.dumps(views)


def _is_stale(plotter, spec: dict) -> bool:
    return (
        spec.get("history_hash") != plotter.data_hash()
        or spec.get("n_obs") != plotter.dataset.adata.n_obs
    )


def capture(plotter, name: str = "") -> dict:
    """The plotter's current view as a spec (see the module docstring)."""
    controls = {
        k: w.value for k, w in plotter.controls.dict.items() if hasattr(w, "value")
    }
    layout = plotter.g.layout
    if controls["n_dimensions"] == "3D":
        camera = layout.scene.camera.to_plotly_json()
        ranges = {"camera": camera} if camera else None
    elif controls["n_dimensions"] in ("1D", "2D"):
        ranges = {"x": layout.xaxis.range, "y": layout.yaxis.range}
    else:
        ranges = None
    spec = {
        "v": 1,
        "id": uuid4().hex[:8],
        "name": name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "controls": controls,
        "ranges": ranges,
        "dragmode": plotter.get_dragmode(),
        "history_hash": plotter.data_hash(),
        "n_obs": plotter.dataset.adata.n_obs,
    }
    from ..processor._history import to_json  # circular import

    return to_json(spec)


def problems(plotter, spec: dict) -> list[str]:
    """What keeps ``spec`` from being restored exactly, as short badges."""
    c, dataset = spec["controls"], plotter.dataset
    if c.get("data_key") not in dataset.data_dict:
        return [f"{c.get('data_key')} missing"]
    found = [
        f"{key} '{c[key]}' missing"
        for key in ("color", "sizeby")
        if c.get(key) is not None and c[key] not in dataset.metadata.columns
    ]
    if _is_stale(plotter, spec):
        found.append("data changed since pin")
    return found


def restore(plotter, spec: dict) -> list[str]:
    """Bring the plotter back to ``spec``; returns warnings (empty: exact).

    The controls are set in one broker batch, so the figure is redrawn once,
    then the saved zoom/camera is applied (each redraw resets it). The zoom is
    skipped if the data changed since the pin (history or cell count).
    """
    if plotter.state.get("drawing"):
        return ["finish drawing before restoring a view"]
    values = spec["controls"]
    if values.get("data_key") not in plotter.dataset.data_dict:
        return [f"{values.get('data_key')} missing"]

    controls, broker = plotter.controls.dict, plotter.broker
    names = [k for k in values if k in controls and hasattr(controls[k], "value")]
    middle = [k for k in names if k not in FIRST and k not in LAST]
    order = [k for k in FIRST + middle + LAST if k in names]
    warnings = []
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
        plotter.make_new_figure()  # also clears gene/custom plots and extra traces
    broker.flush(plotter.make_new_figure)  # if called within an outer batch

    ranges = spec.get("ranges")
    if ranges and _is_stale(plotter, spec):
        warnings.append("data changed since pin: zoom not restored")
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


class ViewGallery(VBox):
    """Strip of pinned-view cards: click to restore, rename, delete."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.views: list[dict] = loads_uns(plotter.dataset.adata)
        self._cards: dict[str, list] = {}  # view id -> card and its widgets
        self.pin_button = Button(
            description="Pin view",
            icon="thumb-tack",
            tooltip="Save the current plot view (not the selection)",
            layout=Layout(width="auto", flex="0 0 auto", margin="2px"),
        )
        self.pin_button.on_click(lambda _: self.pin())
        self.cards = HBox(layout=Layout(width="100%", overflow="auto"))
        self.message = HTML()
        VBox.__init__(
            self, [self.cards, self.message], layout=Layout(width="100%", min_width="0")
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

    def pin(self, name: str | None = None) -> dict | None:
        source = self.plotter.state.get("source", "controls")
        if source != "controls":
            self._say(
                f"{'Gene' if source == 'gene' else 'Custom'} plots cannot be pinned "
                "yet: redraw from the plot settings (Refresh) first"
            )
            return None
        spec = capture(self.plotter, name or f"View {len(self.views) + 1}")
        self.views.append(spec)
        self._sync()
        self._say("")
        return spec

    def restore(self, view_id: str) -> list[str]:
        spec = next(v for v in self.views if v["id"] == view_id)
        warnings = restore(self.plotter, spec)
        self._say("; ".join(warnings))
        return warnings

    def rename(self, view_id: str, name: str):
        next(v for v in self.views if v["id"] == view_id)["name"] = name
        self._sync()

    def delete(self, view_id: str):
        self.views = [v for v in self.views if v["id"] != view_id]
        self._sync()

    def _sync(self):
        dumps_uns(self.plotter.dataset.adata, self.views)
        self._render()

    def _say(self, text: str):
        style = "font-size:12px;color:#b45309"
        self.message.value = f"<span style='{style}'>{escape(text)}</span>"

    def _render(self, *_, **__):
        # cards are kept per view: only their badges follow the data
        for view_id in set(self._cards) - {v["id"] for v in self.views}:
            for widget in self._cards.pop(view_id):
                widget.close()
        for spec in self.views:
            if spec["id"] not in self._cards:
                self._cards[spec["id"]] = self._card(spec)
            *_, restore_button, badge = self._cards[spec["id"]]
            found = problems(self.plotter, spec)
            restore_button.disabled = (
                spec["controls"].get("data_key") not in self.plotter.dataset.data_dict
            )
            style = "font-size:11px;color:#b45309"
            badge.value = f"<span style='{style}'>{escape('; '.join(found))}</span>"
        cards = [self._cards[v["id"]][0] for v in self.views]
        self.cards.children = [self.pin_button, *cards]

    def _card(self, spec: dict) -> list:
        c, view_id = spec["controls"], spec["id"]
        name = Text(spec["name"], continuous_update=False, layout=Layout(width="125px"))
        name.observe(lambda ch: self.rename(view_id, ch["new"]), names="value")
        delete = Button(icon="trash", tooltip="Delete", layout=Layout(width="32px"))
        delete.on_click(lambda _: self.delete(view_id))
        summary = " · ".join(
            str(c[k]) for k in ("data_key", "n_dimensions", "color") if c.get(k)
        )
        restore_button = Button(
            description=summary,
            tooltip=f"Restore: {summary}",
            layout=Layout(width="160px"),
        )
        restore_button.on_click(lambda _: self.restore(view_id))
        badge = HTML()
        row = HBox([name, delete])
        card = VBox(
            [row, restore_button, badge],
            layout=Layout(
                flex="0 0 auto", border="1px solid #ddd", margin="2px", padding="2px"
            ),
        )
        return [card, row, name, delete, restore_button, badge]
