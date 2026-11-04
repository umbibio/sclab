"""Emphasize values: numeric colors size the points by rank, zeros small."""

import numpy as np
import pandas as pd
import pytest

from sclab.dataset.plotter._backends import value_size_factors

from .test_plotter_backends import make_dash
from .test_plotter_patch import sent  # noqa: F401 (fixture)


def test_value_size_factors():
    f = value_size_factors([0.0, 3.0, np.nan, 1.0, -2.0, 2.0, 2.0], min_factor=0.3)
    assert f[[0, 2, 4]].tolist() == [0.3] * 3  # zero, NaN, negative
    assert 0.6 < f[3] < f[5] == f[6] < f[1] < 1.5  # rank order, ties equal
    assert (value_size_factors(np.zeros(5)) == 0.3).all()  # all-zero column
    assert value_size_factors(pd.Series([1.0, None], dtype="Float64")).tolist() == [
        1.05,
        0.3,
    ]
    many = value_size_factors(np.arange(1000.0))
    assert many[0] == 0.3 and 0.6 < many[1] < 0.61 and 1.49 < many[-1] < 1.5


def sizes(plotter) -> pd.Series:
    """Each cell's plotly marker.size (area units: 1.0 is the marker size)."""
    parts = [
        pd.Series(np.broadcast_to(t.marker.size, rows.shape), plotter._rows[rows])
        for t, rows in zip(plotter.g.data, plotter._trace_rows)
        if rows is not None
    ]
    return pd.concat(parts).sort_index()


@pytest.fixture
def dash():
    d = make_dash()
    d.dataset.adata.obs["sparse"] = np.where(
        np.arange(300) % 3 == 0, 0.0, np.arange(300)
    )
    d.broker.publish("dset_metadata_change", d.dataset.metadata)
    return d


def test_plotly_sizes_follow_selection_then_size_by_then_values(dash, sent):  # noqa: F811
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.color.value = "sparse"
    assert ctrl.emphasize_values.value  # on by default
    sparse = dash.dataset.metadata["sparse"]
    expected = pd.Series(value_size_factors(sparse) ** 2, sparse.index).sort_index()
    np.testing.assert_allclose(sizes(p), expected, rtol=1e-6)
    assert all(t.marker.sizeref == 1 / ctrl.marker_size.value**2 for t in p.g.data)

    sent.clear()
    ctrl.emphasize_values.value = False  # patched: only the sizes are sent
    assert (sizes(p) == 1.0).all()
    assert sent["restyle marker.size"] and not sent["add"] and not sent["restyle x"]
    ctrl.emphasize_values.value = True

    p.min_size_input.value = 0.5  # the Settings tab's minimum
    zero = dash.dataset.metadata.index[0]
    assert sizes(p)[zero] == pytest.approx(0.25)

    ctrl.sizeby.value = "poisson"  # Size by wins
    by = sizes(p)
    assert (by == dash.dataset.metadata["poisson"].reindex(by.index)).all()
    ctrl.sizeby.value = None

    dash.dataset.selected_rows = dash.dataset.metadata.index[:10]  # selection wins
    assert set(sizes(p)) == {0.5, 3.0}
    dash.dataset.selected_rows = None
    assert sizes(p)[zero] == pytest.approx(0.25)

    ctrl.color.value = "quadrant"  # categorical: uniform
    assert (sizes(p) == 1.0).all()
    assert not dash.broker.exceptions_log


def test_gene_plot_sizes_survive_size_controls(dash):
    p = dash.plotter
    gene = np.where(np.arange(300) < 200, 0.0, np.arange(300.0))
    metadata = dash.dataset.metadata.assign(gene=gene)
    dash.broker.publish("dplt_plot_figure_request", metadata=metadata, colorby="gene")
    expected = pd.Series(value_size_factors(gene) ** 2, metadata.index)
    np.testing.assert_allclose(sizes(p), expected.sort_index(), rtol=1e-6)
    p.controls.marker_size.value = 3.0  # resizes the gene's plot, not the controls'
    np.testing.assert_allclose(sizes(p), expected.sort_index(), rtol=1e-6)
    assert not dash.broker.exceptions_log


@pytest.mark.parametrize("backend", ["static", "jscatter"])
def test_backends_draw_the_same_sizes(backend):
    if backend == "jscatter":
        pytest.importorskip("jscatter")
    dash = make_dash(backend)
    p, ctrl = dash.plotter, dash.plotter.controls
    ctrl.color.value = "poisson"
    frame = p._view.frame
    factors = value_size_factors(dash.dataset.metadata["poisson"])
    np.testing.assert_allclose(frame["size"], ctrl.marker_size.value * factors)
    ctrl.emphasize_values.value = False
    assert p._view.frame["size"] == ctrl.marker_size.value
    if backend == "jscatter":
        assert p._view.scatter.widget.size == ctrl.marker_size.value
    ctrl.emphasize_values.value = True
    ctrl.sizeby.value = "time"  # Size by wins
    time = dash.dataset.metadata["time"].to_numpy()
    expected = ctrl.marker_size.value * np.sqrt(time / time.max())
    np.testing.assert_allclose(p._view.frame["size"], expected)
    assert not dash.broker.exceptions_log
