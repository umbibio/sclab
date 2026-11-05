"""Pinned plot views: capture, restore, stale data, adata.uns persistence,
gene and custom-figure pins, reorder, thumbnails, code and PNG export."""

import io
import json
import threading
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

from sclab import SCLabDashboard
from sclab.dataset.plotter import Plotter, _views
from sclab.dataset.plotter._backends import _hex
from sclab.dataset.plotter._views import loads_uns, problems

from ...test_dashboard_redraws import figure_state
from ...test_utils import run_step, simple_loop_adata

STEPS = ["QC", "Preprocess", "PCA", "Neighbors", "UMAP", "Cluster"]
CAMERA = {"eye": {"x": 0.5, "y": -2.0, "z": 0.7}}


def values(plotter) -> dict:
    return {k: w.value for k, w in plotter.controls.dict.items() if hasattr(w, "value")}


@pytest.fixture
def redraws(monkeypatch) -> Counter:
    counts = Counter()
    make_new_figure = Plotter.make_new_figure

    def counting(self, *args, **kwargs):
        counts["figure"] += 1
        return make_new_figure(self, *args, **kwargs)

    monkeypatch.setattr(Plotter, "make_new_figure", counting)
    return counts


@pytest.fixture
def dash(redraws) -> SCLabDashboard:
    d = SCLabDashboard(simple_loop_adata(n_obs=300))
    for step in STEPS:
        run_step(d, step)
    d.plotter.controls.color.value = "leiden"
    return d


def zoom(plotter, x, y):
    # what the frontend syncs back after a zoom/pan
    plotter.g._js2py_relayout = {
        "relayout_data": {"xaxis.range": x, "yaxis.range": y},
        "source_view_id": "test",
    }


def test_restore_controls_figure_and_zoom_with_one_redraw(dash, redraws):
    p, ctrl = dash.plotter, dash.plotter.controls
    zoom(p, [-1.0, 1.0], [0.0, 2.0])
    spec = p.views.pin()
    expected_values, expected_figure = values(p), figure_state(dash)
    assert spec["ranges"] == {"x": [-1.0, 1.0], "y": [0.0, 2.0]}

    ctrl.data_key.value = "X_pca"
    ctrl.n_dimensions.value = "3D"
    ctrl.log_axes.children[0].value = True
    ctrl.color.value = None
    ctrl.marker_opacity.value = 0.4
    redraws.clear()

    assert p.views.restore(spec["id"]) == []

    assert redraws["figure"] == 1
    assert values(p) == expected_values
    assert figure_state(dash) == expected_figure
    assert list(p.g.layout.xaxis.range) == [-1.0, 1.0]
    assert not dash.broker.exceptions_log


def test_restore_order_keeps_log_axes_and_camera(dash, redraws):
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.data_key.value = "X_pca"
    ctrl.n_dimensions.value = "3D"
    z_axis = ctrl.selected_axes.children[2]
    z_axis.value = list(z_axis.options)[3]  # PC4
    ctrl.log_axes.children[2].value = True
    p.g._js2py_relayout = {
        "relayout_data": {"scene.camera": CAMERA},
        "source_view_id": "test",
    }
    spec = p.views.pin()
    expected_values = values(p)

    ctrl.data_key.value = "X_2Dumap"  # forces 2D, resets the axes and logs
    redraws.clear()
    p.views.restore(spec["id"])

    assert redraws["figure"] == 1
    assert values(p) == expected_values
    assert p.g.layout.scene.camera.eye.to_plotly_json() == CAMERA["eye"]


def plot_gene(dash, layer="counts", gene=0, colorscale="Viridis") -> str:
    controls = dash.processor.steps["Gene Expression"].variable_controls
    controls["layer"].value = layer
    controls["colorscale"].value = colorscale
    controls["gene_input"].value = controls["gene_input"].options[gene]
    run_step(dash, "Gene Expression")
    assert dash.plotter.state["source"] == "gene"
    return controls["gene_input"].value.split(" ")[0]


def test_gene_pin_restores_the_gene_plot(dash):
    p = dash.plotter
    gene = plot_gene(dash)
    p.min_size_input.value = 0.1
    spec = p.views.pin()
    assert spec["source"] == "gene"
    assert spec["gene"] == {"gene": gene, "layer": "counts", "colorscale": "Viridis"}
    assert spec["settings"] == {"min_size": 0.1}
    assert problems(p, spec) == []  # plotting a gene does not change the data
    expected_figure, expected_values = figure_state(dash), values(p)
    assert loads_uns(dash.dataset.adata)[0]["gene"]["gene"] == gene

    p.controls.color.value = None  # a plot of the controls
    p.min_size_input.value = 0.3
    assert p.state["source"] == "controls"

    assert p.views.restore(spec["id"]) == []
    assert p.state["source"] == "gene" and p._last_call["colorby"] == gene
    assert p.min_size_input.value == 0.1
    assert values(p) == expected_values
    assert figure_state(dash) == expected_figure
    assert p.views.pin()["gene"] == spec["gene"]  # the restored plot pins again
    assert not dash.broker.exceptions_log


def test_gene_pin_with_a_missing_layer_restores_the_controls(dash):
    p = dash.plotter
    spec = p.views.pin()  # before the gene plot: dash's data is unchanged
    plot_gene(dash)
    gene_spec = p.views.pin()
    del dash.dataset.adata.layers["counts"]
    p.views._render()
    assert problems(p, gene_spec) == ["layer 'counts' missing"]
    assert p.views.restore(gene_spec["id"]) == ["layer 'counts' missing"]
    assert p.state["source"] == "controls"
    assert problems(p, spec) == []


def test_unknown_gene_plots_cannot_be_pinned(dash):
    p = dash.plotter
    metadata = dash.dataset.metadata.assign(other=1.0)
    dash.broker.publish("dplt_plot_figure_request", metadata=metadata, colorby="other")
    assert p.state["source"] == "gene"
    assert p.views.pin() is None
    assert "cannot be pinned" in p.views.message.value
    assert p.views.views == []


def test_v1_specs_are_read_as_controls_views(dash):
    p = dash.plotter
    spec = p.views.pin()
    v1 = {k: spec[k] for k in spec if k not in ("source", "settings")} | {"v": 1}
    dash.dataset.adata.uns["sclab"]["views"] = json.dumps([v1, {"v": 1}])
    new = SCLabDashboard(dash.dataset.adata)

    (read,) = new.plotter.views.views  # the malformed one dropped
    assert read["v"] == 2 and read["source"] == "controls"
    new.plotter.controls.color.value = None
    new.plotter.min_size_input.value = 0.2
    assert new.plotter.views.restore(read["id"]) == []
    assert new.plotter.controls.color.value == "leiden"
    assert new.plotter.min_size_input.value == 0.2  # v1 views keep the setting


def test_reorder_persists_in_uns(dash):
    p = dash.plotter
    a, b, c = (p.views.pin(name)["id"] for name in "abc")
    cards = p.views._cards
    assert cards[a]["left"].disabled and cards[c]["right"].disabled
    cards[c]["left"].click()
    assert [v["name"] for v in loads_uns(dash.dataset.adata)] == ["a", "c", "b"]
    p.views.move(a, 5)  # clamped to the end
    assert [v["name"] for v in loads_uns(dash.dataset.adata)] == ["c", "b", "a"]
    assert cards[c]["left"].disabled and cards[a]["right"].disabled
    assert p.views.cards.children == (
        p.views.header,
        *(cards[i]["card"] for i in (c, b, a)),
    )
    new = SCLabDashboard(dash.dataset.adata)
    assert [v["name"] for v in new.plotter.views.views] == ["c", "b", "a"]


def test_thumbnails_follow_the_data(dash, monkeypatch):
    rendered = []
    render_png = _views.render_png

    def counting(frame, **kwargs):
        rendered.append((frame.get("ndims"), kwargs))
        return render_png(frame, **kwargs)

    monkeypatch.setattr(_views, "render_png", counting)
    p = dash.plotter
    zoom(p, [-1.0, 1.0], [0.0, 2.0])
    spec = p.views.pin()
    thumb = p.views._cards[spec["id"]]["thumb"]
    assert len(rendered) == 1 and rendered[0][1]["minimal"]
    assert thumb.format in ("webp", "png") and len(thumb.value) > 100  # zoomed: empty
    first = thumb.value

    p.views._render()  # e.g. a metadata change: same data, cached
    p.controls.color.value = None  # the view's thumbnail does not follow the plot
    assert len(rendered) == 1

    run_step(dash, "UMAP")  # new data: the zoom is stale and skipped
    assert len(rendered) == 2 and thumb.value != first
    assert dash.dataset.adata.uns["sclab"]["views"].count("thumb") == 0
    p.views.delete(spec["id"])
    assert spec["id"] not in p.views._thumbs


def test_thumbnails_of_1d_3d_and_missing_data(dash):
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.n_dimensions.value = "1D"
    one = p.views.pin()
    ctrl.data_key.value = "X_pca"
    ctrl.n_dimensions.value = "3D"
    p.g._js2py_relayout = {
        "relayout_data": {"scene.camera": CAMERA},
        "source_view_id": "test",
    }
    three = p.views.pin()
    frame = _views.view_frame(p, three)
    assert frame["ndims"] == 3 and frame["camera"] == CAMERA["eye"]
    assert _views.view_frame(p, one)["ndims"] == 1
    for spec in (one, three):
        png = p.views.png(spec["id"], dpi=50)
        assert png.startswith(b"\x89PNG")

    three["controls"]["data_key"] = "X_gone"
    p.views._thumbs.clear()
    p.views._render()
    placeholder = p.views._cards[three["id"]]["thumb"].value
    assert placeholder == _views.placeholder("no preview", _views.thumb_format())


def run_code(dash, spec) -> go.Figure:
    namespace = {"adata": dash.dataset.adata}
    exec(dash.plotter.views.code_for(spec["id"]), namespace)
    return namespace["fig"]


def test_code_reproduces_the_plot(dash):
    p = dash.plotter
    p.controls.show_density.value = True
    zoom(p, [-1.0, 1.0], [0.0, 2.0])
    spec = p.views.pin()
    code = p.views.code_for(spec["id"])
    assert "import plotly.express as px" in code and "px.scatter(" in code

    fig = run_code(dash, spec)
    cells = [t for t in p.g.data if t.type != "contour"]
    assert {t.name for t in fig.data if t.type == "scattergl"} == {
        t.name for t in cells
    }
    assert fig.layout.legend.title.text == "leiden"
    colors = {t.name: _hex(t.marker.color) for t in cells}
    assert all(t.marker.color == colors[t.name] for t in fig.data if t.name in colors)
    assert any(t.type == "histogram2dcontour" for t in fig.data)
    assert list(fig.layout.xaxis.range) == [-1.0, 1.0]

    gene = plot_gene(dash)
    fig = run_code(dash, p.views.pin())
    (trace,) = [t for t in fig.data if t.type == "scattergl"]
    values = _views.gene_values(dash.dataset.adata, gene, "counts")
    assert np.allclose(trace.marker.color, values)
    assert fig.layout.coloraxis.colorbar.title.text == gene

    p.controls.data_key.value = "X_pca"
    p.controls.n_dimensions.value = "3D"
    p.controls.log_axes.children[0].value = True
    fig = run_code(dash, p.views.pin())
    assert {t.type for t in fig.data} == {"scatter3d"}
    assert fig.layout.scene.xaxis.type == "log"

    p.views.show_code(spec["id"])
    assert p.views.code.text == p.views.code_for(spec["id"])
    assert p.views.code.layout.display == ""
    p.views.show_code(spec["id"])
    assert p.views.code.layout.display == "none"


def test_custom_figures_pin_as_snapshots(dash, monkeypatch):
    p = dash.plotter
    obs = dash.dataset.adata.obs
    figure = px.scatter(obs, x="time", y="total_counts", color="leiden", title="mine")
    dash.broker.publish("dplt_plot_figure_request", figure=figure)
    assert p.state["source"] == "custom"
    spec = p.views.pin()
    assert spec["source"] == "custom" and spec["name"] == "Snapshot 1"

    card = p.views._cards[spec["id"]]
    assert card["restore"].disabled and card["code"].disabled
    assert "Snapshot: mine" in card["restore"].description
    assert len(card["thumb"].value) > 1000  # not the placeholder
    assert loads_uns(dash.dataset.adata) == []  # memory only
    assert problems(p, spec) == []
    assert p.views.restore(spec["id"]) == [
        "snapshots of custom figures cannot be restored"
    ]

    sent = []
    monkeypatch.setattr(
        p.download, "send", lambda msg, buffers=None: sent.append((msg, buffers))
    )
    p.views.dpi.value = 100
    card["png"].click()
    ((msg, (png,)),) = sent
    assert msg == {"filename": "SCLab_Dashboard_Snapshot_1.png", "type": "image/png"}
    assert png.startswith(b"\x89PNG")

    p.controls.refresh_button.click()  # back to the plot settings
    controls_spec = p.views.pin()
    assert controls_spec["source"] == "controls"
    assert [v["id"] for v in loads_uns(dash.dataset.adata)] == [controls_spec["id"]]


def test_png_download_is_full_size(dash, monkeypatch):
    p = dash.plotter
    spec = p.views.pin()
    sent = []
    monkeypatch.setattr(
        p.download, "send", lambda msg, buffers=None: sent.append((msg, buffers))
    )
    p.views.dpi.value = 100
    p.views._cards[spec["id"]]["png"].click()
    ((msg, (png,)),) = sent
    from PIL import Image as PILImage

    image = PILImage.open(io.BytesIO(png))
    assert image.size == (900, p.controls.plot_height.value)  # width 0: auto
    assert msg["type"] == "image/png" and msg["filename"].endswith("View_1.png")


def test_missing_color_column_restores_none_with_warning(dash):
    p, adata = dash.plotter, dash.dataset.adata
    adata.obs["tmp"] = pd.Categorical(["a", "b", "c"] * (adata.n_obs // 3))
    dash.broker.publish("dset_metadata_change", dash.dataset.metadata)
    p.controls.color.value = "tmp"
    spec = p.views.pin()

    del adata.obs["tmp"]
    dash.broker.publish("dset_metadata_change", dash.dataset.metadata)
    assert problems(p, spec) == ["color 'tmp' missing"]
    assert "color &#x27;tmp&#x27; missing" in p.views._cards[spec["id"]]["badge"].value

    assert p.views.restore(spec["id"]) == ["color 'tmp' missing"]
    assert p.controls.color.value is None
    assert p.controls.data_key.value == spec["controls"]["data_key"]


def test_missing_data_key_disables_the_card(dash):
    p = dash.plotter
    spec = p.views.pin()
    spec["controls"]["data_key"] = "X_gone"
    p.views._render()
    restore_button = p.views._cards[spec["id"]]["restore"]
    assert restore_button.disabled
    before = values(p)
    assert p.views.restore(spec["id"]) == ["X_gone missing"]
    assert values(p) == before


def test_changed_data_skips_the_zoom(dash):
    p = dash.plotter
    zoom(p, [-1.0, 1.0], [0.0, 2.0])
    spec = p.views.pin()
    run_step(dash, "UMAP")  # recomputed embedding: the zoom would be meaningless

    assert problems(p, spec) == ["data changed since pin"]
    assert "data changed since pin" in p.views._cards[spec["id"]]["badge"].value
    warnings = p.views.restore(spec["id"])
    assert warnings == ["data changed since pin: zoom not restored"]
    assert list(p.g.layout.xaxis.range) != [-1.0, 1.0]
    assert values(p) == spec["controls"]


def test_restore_refused_while_drawing(dash):
    p, ctrl = dash.plotter, dash.plotter.controls
    spec = p.views.pin()
    ctrl.color.value = None
    p.state["drawing"] = True
    try:
        assert "drawing" in p.views.restore(spec["id"])[0]
        assert ctrl.color.value is None
    finally:
        p.state["drawing"] = False


def test_views_travel_with_the_h5ad(dash, tmp_path):
    p = dash.plotter
    zoom(p, [-1.0, 1.0], [0.0, 2.0])
    first = p.views.pin()
    second = p.views.pin()
    p.views.rename(first["id"], "my umap")
    p.views.delete(second["id"])
    assert [v["name"] for v in loads_uns(dash.dataset.adata)] == ["my umap"]

    dash.dataset.adata.write_h5ad(tmp_path / "a.h5ad")
    new = SCLabDashboard(ad.read_h5ad(tmp_path / "a.h5ad"))

    assert new.plotter.views.views == loads_uns(dash.dataset.adata)
    (spec,) = new.plotter.views.views
    assert problems(new.plotter, spec) == []  # same history and cells
    new.plotter.controls.color.value = None
    assert new.plotter.views.restore(spec["id"]) == []
    assert new.plotter.controls.color.value == "leiden"
    assert list(new.plotter.g.layout.xaxis.range) == [-1.0, 1.0]


def test_rename_from_the_card(dash):
    p = dash.plotter
    spec = p.views.pin()
    name_input = p.views._cards[spec["id"]]["name"]
    name_input.value = "renamed"
    assert loads_uns(dash.dataset.adata)[0]["name"] == "renamed"
    p.views._cards[spec["id"]]["delete"].click()  # delete
    assert loads_uns(dash.dataset.adata) == []
    assert p.views.cards.children == (p.views.header,)


def test_save_downloads_in_the_browser_without_writing_files(
    dash, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    sent = []
    monkeypatch.setattr(dash.plotter.download, "send", sent.append)

    dash.plotter.controls.save_button.click()

    (msg,) = sent
    assert msg["filename"].endswith("___colorBy_leiden.html")
    assert msg["type"] == "text/html" and "<html>" in msg["text"]
    assert list(tmp_path.iterdir()) == []


def test_thumbnails_and_pngs_render_off_the_main_thread(dash, monkeypatch):
    from sclab.event import _jobs

    callbacks = []

    class Loop:  # the kernel's IOLoop: callbacks from the worker thread
        def add_callback(self, fn, *args):
            callbacks.append((threading.current_thread(), fn, args))

    monkeypatch.setattr(_jobs, "_kernel_loop", lambda: Loop())
    p = dash.plotter
    spec = p.views.pin()
    thumb = p.views._cards[spec["id"]]["thumb"]
    p.views._pool.shutdown(wait=True)
    p.views._pool = None
    ((thread, show, args),) = callbacks
    assert thread is not threading.main_thread() and len(thumb.value) == 0
    show(*args)  # on the main thread
    assert len(thumb.value) > 100

    sent = []
    monkeypatch.setattr(p.download, "send", lambda msg, buffers=None: sent.append(1))
    p.views.download_png(spec["id"])
    assert "Drawing" in p.views.message.value and not sent
    p.views._pool.shutdown(wait=True)
    _, save, args = callbacks[-1]
    save(*args)
    assert sent == [1] and "Drawing" not in p.views.message.value
