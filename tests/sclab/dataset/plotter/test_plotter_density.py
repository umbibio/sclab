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
