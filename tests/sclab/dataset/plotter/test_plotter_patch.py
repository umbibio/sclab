"""Patched redraws, the cell index map, and 3D subsampling/chunking."""

import contextlib
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from sclab import SCLabDashboard
from sclab.dataset.plotter import _plotter

from ...test_utils import simple_loop_adata


@pytest.fixture
def dash() -> SCLabDashboard:
    d = SCLabDashboard(simple_loop_adata(n_obs=300))
    d.plotter.controls.data_key.value = "Y_array2"
    return d


@pytest.fixture
def sent(dash) -> Counter:
    """Messages to the frontend: add/delete traces, and the restyled properties."""
    g, counts = dash.plotter.g, Counter()
    add, delete, update = (
        g._send_addTraces_msg,
        g._send_deleteTraces_msg,
        g._send_update_msg,
    )

    def on_add(*args):
        counts["add"] += 1
        return add(*args)

    def on_delete(*args):
        counts["delete"] += 1
        return delete(*args)

    def on_update(*args, **kwargs):
        counts.update(f"restyle {k}" for k in kwargs.get("restyle_data", {}))
        return update(*args, **kwargs)

    g._send_addTraces_msg = on_add
    g._send_deleteTraces_msg = on_delete
    g._send_update_msg = on_update
    return counts


def cells(plotter) -> list:
    return [
        list(plotter._rows[rows]) for rows in plotter._trace_rows if rows is not None
    ]


def test_index_map_and_hover_ids(dash):
    p = dash.plotter
    p.controls.color.value = "quadrant"
    shown = [c for trace in cells(p) for c in trace]
    assert sorted(shown) == sorted(dash.dataset.adata.obs_names)
    for trace, rows in zip(p.g.data, p._trace_rows):
        assert trace.hovertext is None
        assert list(trace.customdata) == list(p._rows[rows])
        assert trace.hovertemplate.startswith("<b>%{customdata}</b>")
        assert np.asarray(trace.x).dtype == np.float32


@pytest.mark.parametrize("ndims", ["1D", "2D"])
def test_missing_categories_are_drawn_as_na(dash, ndims):
    p = dash.plotter
    p.controls.n_dimensions.value = ndims
    missing = dash.dataset.metadata["quadrant"].isna()
    assert missing.sum() == 150
    # pandas 3's str dtype (pandas 2: future.infer_string) keeps NaN in astype(str)
    infer_string = (
        contextlib.nullcontext()
        if int(pd.__version__.split(".")[0]) >= 3
        else pd.option_context("future.infer_string", True)
    )
    with infer_string:
        p.controls.color.value = "quadrant"
    (na,) = [t for t in p.g.data if t.name == "NA"]
    if ndims == "2D":
        shown = [c for trace in cells(p) for c in trace]
        assert sorted(shown) == sorted(dash.dataset.adata.obs_names)
        assert sorted(na.customdata) == sorted(missing.index[missing])
    else:  # histogram bars: the NA trace counts the missing cells
        assert len(na.x) == 150


def test_selected_points_from_trace_selections(dash):
    p = dash.plotter
    p.controls.color.value = "quadrant"
    first, second = p.g.data[0], p.g.data[1]
    first.selectedpoints, second.selectedpoints = [0, 2], [1]
    expected = {cells(p)[0][0], cells(p)[0][2], cells(p)[1][1]}
    assert set(p.selected_points) == expected


def test_point_click_grows_the_marker(dash):
    p = dash.plotter
    name = cells(p)[0][5]
    dash.broker.publish("dplt_point_click", row_name=name, device_state=None)
    sizes = np.asarray(p.g.data[0].marker.size)
    assert sizes.shape == (len(cells(p)[0]),)
    assert sizes[5] == p.controls.marker_size.value
    assert (np.delete(sizes, 5) == 1.0).all()
    assert not dash.broker.exceptions_log


def test_marker_size_is_scalar_without_size_by_or_selection(dash):
    p = dash.plotter
    assert all(t.marker.size == 1.0 for t in p.g.data)
    dash.dataset.selected_rows = dash.dataset.metadata.index[:10]
    sizes = np.concatenate([np.asarray(t.marker.size) for t in p.g.data])
    assert sizes.dtype == np.float32 and set(sizes) == {0.5, 3.0}
    dash.dataset.selected_rows = None
    assert all(t.marker.size == 1.0 for t in p.g.data)


def test_size_and_selection_only_send_sizes(dash, sent):
    p = dash.plotter
    p.controls.marker_size.value = 3.0
    dash.dataset.selected_rows = dash.dataset.metadata.index[:10]
    assert not sent["add"] and not sent["delete"]
    assert not sent["restyle x"] and not sent["restyle customdata"]
    assert sent["restyle marker.sizeref"] and sent["restyle marker.size"]


def test_recolor_patches_traces_of_the_same_kind(dash, sent):
    p = dash.plotter
    p.controls.color.value = "time"
    sent.clear()
    p.controls.color.value = "poisson"  # continuous -> continuous: patched
    assert not sent["add"] and not sent["delete"]
    assert sent["restyle marker.color"]

    p.controls.color.value = "quadrant"  # continuous -> categorical: rebuilt
    assert sent["add"] == 1 and sent["delete"] == 1


def test_patched_figure_equals_a_rebuilt_one(dash):
    p = dash.plotter
    p.controls.color.value = "time"
    p.controls.color.value = "poisson"  # patched
    patched = p.g.to_plotly_json()
    p._structure = None
    p.make_new_figure()  # rebuilt
    rebuilt = p.g.to_plotly_json()
    for a, b in zip(patched["data"], rebuilt["data"]):
        a.pop("uid"), b.pop("uid")
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_equal(a[key], b[key])


def test_extra_traces_force_a_rebuild(dash, sent):
    p = dash.plotter
    dash.broker.publish(
        "dplt_add_data_as_line_trace_request",
        "Y_array2",
        dash.dataset.data.iloc[:5],
    )
    n = len(p.g.data)
    sent.clear()
    p.controls.refresh_button.click()
    assert sent["add"] == 1 and len(p.g.data) == n - 1


@pytest.fixture
def dash3d(dash, monkeypatch) -> SCLabDashboard:
    dash.plotter.max_3d_points = 50
    monkeypatch.setattr(_plotter, "MAX_TRACE_POINTS", 40)
    dash.plotter.controls.data_key.value = "Y_array4"
    dash.plotter.controls.n_dimensions.value = "3D"
    return dash


def test_3d_draws_a_stable_seeded_subsample(dash3d):
    p = dash3d.plotter
    shown = sorted(c for trace in cells(p) for c in trace)
    assert len(shown) == 50
    assert "50 of 300 cells" in p.g.layout.title.text

    p.controls.color.value = "time"  # same cells after a recolor
    assert sorted(c for trace in cells(p) for c in trace) == shown
    assert not dash3d.broker.exceptions_log


def test_3d_subsample_keeps_the_selection(dash3d):
    p, obs_names = dash3d.plotter, dash3d.dataset.metadata.index
    p.controls.full_resolution.value = True  # all 300 cells, 3D shows selected only
    dash3d.dataset.selected_rows = obs_names[:80]
    assert sorted(c for t in cells(p) for c in t) == sorted(obs_names[:80])

    p.controls.full_resolution.value = False
    shown = [c for t in cells(p) for c in t]
    assert len(shown) == 50 and set(shown) <= set(obs_names[:80])
    assert all(set(np.asarray(t.marker.size)) == {3.0} for t in p.g.data)


def test_3d_full_resolution_splits_traces_into_chunks(dash3d):
    p = dash3d.plotter
    p.controls.full_resolution.value = True
    sizes = [len(rows) for rows in p._trace_rows]
    assert sum(sizes) == 300 and max(sizes) <= 40
    assert [t.showlegend for t in p.g.data][1:] == [False] * (len(sizes) - 1)
    assert p.g.layout.title.text == "Y_array4"


def test_3d_camera_survives_a_redraw_on_the_same_basis(dash3d):
    p = dash3d.plotter
    camera = {"eye": {"x": 0.5, "y": -2.0, "z": 0.7}}
    p.g._js2py_relayout = {
        "relayout_data": {"scene.camera": camera},
        "source_view_id": "test",
    }
    p.controls.color.value = "quadrant"
    assert p.g.layout.scene.camera.eye.to_plotly_json() == camera["eye"]
    assert p.g.layout.scene.uirevision == "Y_array4"
