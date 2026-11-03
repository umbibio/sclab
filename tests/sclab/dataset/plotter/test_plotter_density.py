import plotly.graph_objects as go
import pytest

from sclab import SCLabDashboard

from ...test_utils import run_step, simple_loop_adata


@pytest.fixture(scope="module")
def dash() -> SCLabDashboard:
    d = SCLabDashboard(simple_loop_adata(n_obs=200))
    for name in ["QC", "Preprocess", "PCA"]:
        run_step(d, name)
    d.plotter.controls.data_key.value = "X_pca"
    return d


def test_density_contours_change_updates_contour_size(dash):
    # plotly>=6 stores numeric arrays of widget traces as typed-array dicts, so the
    # callback must not read trace.z back from the widget
    plotter, ctrl = dash.plotter, dash.plotter.controls
    n_exc = len(dash.broker.exceptions_log)

    ctrl.show_density.value = True
    contour = next(t for t in plotter.g.data if isinstance(t, go.Contour))
    size_before = contour.contours.size

    ctrl.density_contours.value = ctrl.density_contours.value + 4

    assert not dash.broker.exceptions_log[n_exc:]
    assert contour.contours.size != size_before


def test_frontend_selection_with_density_contours(dash):
    plotter, ctrl = dash.plotter, dash.plotter.controls
    ctrl.show_density.value = True
    scatter = next(t for t in plotter.g.data if not isinstance(t, go.Contour))
    scatter.selectedpoints = [0, 1, 2]
    published = []
    dash.broker.subscribe("dplt_selected_points_change", published.append)

    # what the frontend sends after a box/lasso selection
    plotter.g._js2py_relayout = {
        "relayout_data": {"selections": [{"type": "rect"}]},
        "source_view_id": "test",
    }

    assert len(published) == 1 and len(published[0]) == 3
    assert scatter.selectedpoints is None
