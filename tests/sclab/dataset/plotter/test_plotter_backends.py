"""2D plot backends: Settings tab, plotly fallbacks, static images."""

import numpy as np
import plotly.graph_objects as go
import pytest
from ipywidgets import Image

from sclab import SCLabDashboard

from ...test_utils import simple_loop_adata


def make_dash(backend="plotly", n_obs=300) -> SCLabDashboard:
    adata = simple_loop_adata(n_obs=n_obs)
    adata.obsm["X_umap"] = np.random.default_rng(0).normal(size=(adata.n_obs, 2))
    d = SCLabDashboard(adata, plot_backend=backend)
    d.plotter.controls.data_key.value = "X_umap"
    assert d.plotter.controls.n_dimensions.value == "2D"
    return d


def shown(dash):
    return dash.plotter.graph_box.children[0]


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


@pytest.mark.parametrize("backend", ["static"])
def test_pinned_view_restores_settings(backend):
    dash = make_dash(backend)
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.color.value = "quadrant"
    ctrl.marker_opacity.value = 0.6
    spec = p.views.pin()
    assert spec["backend"] == backend and spec["ranges"] is None
    expected = values(p)

    ctrl.color.value = "time"
    ctrl.marker_opacity.value = 1.0
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
