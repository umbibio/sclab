from sclab import SCLabDashboard

from ...test_utils import simple_loop_adata


def test_figure_widget_is_responsive_and_has_no_fixed_width():
    # plotly>=6 only autosizes on first render if plotly.js watches the container
    g = SCLabDashboard(simple_loop_adata(n_obs=50)).plotter.g
    assert g.get_state()["_config"]["responsive"] is True
    assert g.layout.width is None


def test_figure_widget_does_not_sync_a_view_count():
    # its view would send the decrement after the kernel closed the comm: "No such comm"
    g = SCLabDashboard(simple_loop_adata(n_obs=50)).plotter.g
    assert g.get_state()["_view_count"] is None
