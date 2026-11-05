"""2D plot backends: Settings tab, plotly fallbacks, static images, jupyter-scatter."""

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from ipywidgets import Image
from matplotlib.path import Path

from sclab import SCLabDashboard

from ...test_utils import simple_loop_adata

LASSO = [[-1.0, -1.0], [1.5, -0.5], [1.0, 1.5], [-0.5, 1.0]]


def make_dash(backend="plotly", n_obs=300) -> SCLabDashboard:
    adata = simple_loop_adata(n_obs=n_obs)
    adata.obsm["X_umap"] = np.random.default_rng(0).normal(size=(adata.n_obs, 2))
    d = SCLabDashboard(adata, plot_backend=backend)
    d.plotter.controls.data_key.value = "X_umap"
    assert d.plotter.controls.n_dimensions.value == "2D"
    return d


def shown(dash):
    """The plot widget the user sees: the figure or a backend's view."""
    p = dash.plotter
    first = next(w for w in p.graph_box.children if w.layout.display != "none")
    return p.g if first is p._figure_box else first


def values(plotter) -> dict:
    return {k: w.value for k, w in plotter.controls.dict.items() if hasattr(w, "value")}


def test_settings_tab_switches_the_2d_backend():
    dash = make_dash()
    p = dash.plotter
    assert dash.main_content.titles[-1] == "Settings"
    assert dash.main_content.children[-1] is p.settings
    assert p.controls.full_resolution in p.settings.children
    assert shown(dash) is p.g and p.g.data

    p.backend_dropdown.value = "static"  # as in the Settings tab
    assert p.backend == "static" and isinstance(shown(dash), Image)
    assert not p.g.data  # the hidden plotly traces are freed

    p.backend = "plotly"  # from Python
    assert p.backend_dropdown.value == "plotly"
    assert shown(dash) is p.g and p.g.data
    with pytest.raises(ValueError):
        p.backend = "bokeh"
    assert not dash.broker.exceptions_log


def test_backend_switch_keeps_the_figure_attached():
    # a re-attached plotly view misses the traces sent right after it: plotly
    # stayed empty after static -> plotly, so the figure is hidden, not detached
    dash = make_dash()
    p = dash.plotter
    box = p.graph_box.children[0]
    assert box.children == (p.g,)
    p.backend = "static"
    assert p.graph_box.children[0] is box and box.layout.display == "none"
    p.backend = "plotly"
    assert p.graph_box.children[0] is box and box.layout.display == "block"
    assert not any(isinstance(w, Image) for w in p.graph_box.children)
    assert shown(dash) is p.g and p.g.data


def test_1d_and_3d_always_use_plotly():
    dash = make_dash("static")
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.data_key.value = "Y_array4"
    for ndims in ("1D", "3D"):
        ctrl.n_dimensions.value = ndims
        assert shown(dash) is p.g and p.g.data
    ctrl.n_dimensions.value = "2D"
    assert isinstance(shown(dash), Image)


def test_max_3d_points_setting():
    dash = make_dash()
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.data_key.value = "Y_array4"
    ctrl.n_dimensions.value = "3D"
    p.max_3d_points = 50
    assert p.max_3d_points_input.value == 50
    assert "50 of 300 cells" in p.g.layout.title.text
    ctrl.full_resolution.value = True
    assert p.g.layout.title.text == "Y_array4"


def test_missing_jupyter_scatter_falls_back_to_plotly(monkeypatch):
    monkeypatch.setitem(sys.modules, "jscatter", None)  # import raises ImportError
    dash = make_dash("jscatter")
    p = dash.plotter
    assert p.backend == "plotly" and shown(dash) is p.g and p.g.data
    assert "jupyter-scatter is not installed" in p.notice.value


def test_plotly_only_features_fall_back_with_a_notice():
    dash = make_dash("static")
    p = dash.plotter

    dash.broker.publish(
        "dplt_add_trace_request", trace=go.Scattergl(x=[0, 1], y=[0, 1])
    )
    assert shown(dash) is p.g and len(p.g.data) == 2  # the scatter and the line
    assert "does not support extra traces" in p.notice.value

    p.controls.refresh_button.click()  # a redraw drops extra traces, as with plotly
    assert isinstance(shown(dash), Image) and p.notice.value == ""

    dash.broker.publish("dplt_plot_figure_request", figure=go.Figure(go.Bar(y=[1, 2])))
    assert shown(dash) is p.g and p.g.data[0].type == "bar"
    assert "custom figures" in p.notice.value

    p.controls.refresh_button.click()
    dash.broker.publish("dplt_start_drawing_request")
    assert shown(dash) is p.g and "path drawing" in p.notice.value
    dash.broker.publish("dplt_end_drawing_request")
    p.controls.refresh_button.click()
    assert isinstance(shown(dash), Image)
    assert not dash.broker.exceptions_log


def test_static_images_for_color_density_and_selection():
    dash = make_dash("static")
    p, ctrl = dash.plotter, dash.plotter.controls
    images = {"plain": bytes(shown(dash).value)}
    ctrl.color.value = "quadrant"
    assert p._view.frame["color"]["kind"] == "categorical"
    assert p._view.frame["color"]["colors"]["NA"] == "lightgray"
    images["categorical"] = bytes(shown(dash).value)
    ctrl.color.value = "time"
    assert p._view.frame["color"]["kind"] == "continuous"
    images["continuous"] = bytes(shown(dash).value)
    ctrl.show_density.value = True
    assert p._view.frame["density"]
    images["density"] = bytes(shown(dash).value)
    dash.dataset.selected_rows = dash.dataset.metadata.index[:50]
    assert p._view.frame["selected"].sum() == 50
    images["selection"] = bytes(shown(dash).value)
    ctrl.marker_size.value = 3.0
    ctrl.marker_opacity.value = 0.4
    assert p._view.frame["marker_size"] == 3.0 and p._view.frame["opacity"] == 0.4

    assert all(png.startswith(b"\x89PNG") for png in images.values())
    assert len(set(images.values())) == len(images)  # each state draws differently
    assert not dash.broker.exceptions_log


def test_gene_plot_on_a_backend_keeps_its_colorscale():
    dash = make_dash("static")
    p = dash.plotter
    metadata = dash.dataset.metadata.assign(gene=np.arange(300.0))
    dash.broker.publish(
        "dplt_plot_figure_request",
        metadata=metadata,
        colorby="gene",
        color_continuous_scale="Reds",
    )
    spec = p._view.frame["color"]
    assert spec["kind"] == "continuous" and spec["range"] == (0.0, 299.0)
    assert spec["colors"][0] != spec["colors"][-1]
    p.controls.marker_size.value = 2.0  # redraws the gene plot, not the controls'
    assert p._view.frame["color_title"] == "gene"


@pytest.mark.parametrize("backend", ["static", "jscatter"])
def test_pinned_view_restores_settings(backend):
    if backend == "jscatter":
        pytest.importorskip("jscatter")
    dash = make_dash(backend)
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.color.value = "quadrant"
    ctrl.marker_opacity.value = 0.6
    ctrl.emphasize_values.value = False
    spec = p.views.pin()
    assert spec["backend"] == backend and spec["ranges"] is None
    assert spec["controls"]["emphasize_values"] is False
    expected = values(p)

    ctrl.color.value = "time"
    ctrl.marker_opacity.value = 1.0
    ctrl.emphasize_values.value = True
    ctrl.data_key.value = "Y_array2"
    assert p.views.restore(spec["id"]) == []
    assert values(p) == expected
    assert p._view.frame["color_title"] == "quadrant"
    assert p._view.frame["opacity"] == 0.6
    badge = p.views._cards[spec["id"]][-1].value
    assert ("zoom not saved (jscatter)" in badge) == (backend == "jscatter")
    assert not dash.broker.exceptions_log


def test_plotly_zoom_is_not_restored_on_another_backend():
    dash = make_dash()
    p = dash.plotter
    p.g._js2py_relayout = {
        "relayout_data": {"xaxis.range": [-1, 1], "yaxis.range": [0, 2]},
        "source_view_id": "test",
    }
    spec = p.views.pin()
    p.backend = "static"
    assert p.views.restore(spec["id"]) == ["zoom not restored (static plot)"]


# ---------------------------------------------------------------- jupyter-scatter


@pytest.fixture
def jdash() -> SCLabDashboard:
    pytest.importorskip("jscatter")
    return make_dash("jscatter")


def test_jscatter_draws_and_sends_only_changes(jdash):
    p, ctrl = jdash.plotter, jdash.plotter.controls
    view = p._view
    scatter = view.scatter
    assert shown(jdash) is view.widget and not p.g.data
    assert len(scatter.widget.points) == 300

    ctrl.color.value = "quadrant"
    assert scatter.widget.legend and scatter._color_by == "Custom Color Data"
    ctrl.marker_size.value = 3.0
    assert scatter.widget.size == 3.0
    ctrl.marker_opacity.value = 0.5
    assert scatter.widget.opacity == 0.5
    ctrl.show_density.value = True
    assert len(scatter.widget.annotations) == len(view.frame["density"]) > 0
    jdash.dataset.selected_rows = jdash.dataset.metadata.index[:40]
    assert len(scatter.widget.selection) == 40
    assert view.scatter is scatter  # same points: updated, not rebuilt

    ctrl.data_key.value = "Y_array2"  # new points: a new scatter
    assert view.scatter is not scatter
    assert not jdash.broker.exceptions_log


def inside_lasso(dash) -> np.ndarray:
    return Path(LASSO).contains_points(dash.dataset.adata.obsm["X_umap"])


@pytest.mark.parametrize("polygon_first", [False, True])
def test_jscatter_lasso_selects_and_records_the_shape(jdash, polygon_first):
    widget = jdash.plotter._view.scatter.widget
    inside = inside_lasso(jdash)
    selection = np.flatnonzero(inside).astype("uint32")
    polygon = np.asarray(LASSO, dtype="float32")
    # the frontend sends the selection, then the polygon, as separate updates
    if polygon_first:
        widget.set_trait("lasso_selection_polygon", polygon)
        widget.selection = selection
    else:
        widget.selection = selection
        widget.set_trait("lasso_selection_polygon", polygon)

    assert jdash.dataset.selected_rows_mask.tolist() == inside.tolist()
    jdash.processor.dspr_keep_selected_click_callback()
    entry = jdash.processor.step_history[-1]
    assert entry["status"] == "success", entry["error"]
    assert entry["params"]["shape"]["type"] == "lasso"
    assert entry["params"]["shape"]["polygon"] == LASSO
    stored = jdash.dataset.adata.uns["sclab"]["selections"]["sel1"]
    assert stored["basis"] == "X_umap" and stored["polygon"].shape == (4, 2)
    assert jdash.dataset.adata.n_obs == inside.sum()
    assert not jdash.broker.exceptions_log


def test_jscatter_merged_lasso_or_click_is_recorded_as_a_mask(jdash):
    widget = jdash.plotter._view.scatter.widget
    inside = inside_lasso(jdash)
    merged = inside.copy()
    merged[np.flatnonzero(~inside)[:3]] = True  # shift-lasso: adds to the selection
    widget.selection = np.flatnonzero(merged).astype("uint32")
    widget.set_trait("lasso_selection_polygon", np.asarray(LASSO, dtype="float32"))
    assert jdash.plotter.selection_shape is None
    assert jdash.dataset.selected_rows_mask.tolist() == merged.tolist()

    widget.selection = np.array([7], dtype="uint32")  # a click selects one point
    assert list(jdash.dataset.selected_rows) == [jdash.dataset.metadata.index[7]]
    jdash.processor.dspr_keep_selected_click_callback()
    assert set(jdash.processor.step_history[-1]["params"]) == {"mask", "n"}

    widget.selection = np.array([], dtype="uint32")  # deselect
    assert jdash.dataset._selected_rows is None or not len(
        pd.Index(jdash.dataset.selected_rows)
    )
