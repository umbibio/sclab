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


@pytest.mark.parametrize(
    "event, arg, ref",
    [
        ("dplt_add_vline_request", "vlines", "yref"),
        ("dplt_add_hline_request", "hlines", "xref"),
    ],
)
def test_add_line_requests(dash, event, arg, ref):
    g, broker = dash.plotter.g, dash.broker
    n_exc, n_shapes = len(broker.exceptions_log), len(g.layout.shapes)

    broker.publish(event, **{arg: 0.5})
    broker.publish(event, **{arg: [0.1, 0.2]}, line_dash="dash")

    assert broker.exceptions_log[n_exc:] == []
    new = g.layout.shapes[n_shapes:]
    assert len(new) == 3
    assert all(s[ref].endswith(" domain") for s in new)
    assert new[1].line.dash == "dash"
