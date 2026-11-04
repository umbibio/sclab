"""Keep/Drop/Label record how cells were selected; the emitted code reproduces it."""

import logging
import tempfile

import anndata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.path import Path

import sclab
from sclab import SCLabDashboard
from sclab.dataset.processor._selection_code import (
    MAX_INLINE_VERTICES,
    SELECTIONS_FILE,
    apply_label_code,
    filter_rows_code,
)

from .test_processor import make_adata

CODE = {"filter_rows": filter_rows_code, "apply_label": apply_label_code}
LASSO = [[-1.0, -1.0], [1.5, -0.5], [1.0, 1.5], [-0.5, 1.0]]


def original() -> anndata.AnnData:
    adata = make_adata()
    rng = np.random.default_rng(0)
    adata.obsm["X_umap"] = rng.normal(size=(adata.n_obs, 2))
    return adata


@pytest.fixture
def dash() -> SCLabDashboard:
    d = SCLabDashboard(original())
    d.plotter.controls.data_key.value = "X_umap"
    assert d.plotter.controls.n_dimensions.value == "2D"
    return d


def plot_select(dash, selections: list[dict], inside: np.ndarray):
    """What the frontend sends on a box/lasso: selectedpoints, then a relayout."""
    names = set(dash.dataset.adata.obs_names[inside])
    for trace in dash.plotter.g.data:
        if not isinstance(trace, go.Contour):
            ids = trace.hovertext
            trace.selectedpoints = [i for i, n in enumerate(ids) if n in names]
    dash.plotter.g._js2py_relayout = {
        "relayout_data": {"selections": selections},
        "source_view_id": "test",
    }


def rect(x, y) -> dict:
    return {"type": "rect", "x0": x[1], "x1": x[0], "y0": y[0], "y1": y[1]}


def box(dash, x, y):
    # selectedpoints stay empty: as for lassos, they may lag behind the relayout
    none = np.zeros(dash.dataset.adata.n_obs, bool)
    plot_select(dash, [{**rect(x, y), "xref": "x", "yref": "y"}], none)


def lasso(dash, polygon=LASSO, inside=None):
    if inside is None:
        inside = np.zeros(dash.dataset.adata.n_obs, bool)
    path = "M" + "L".join(f"{x},{y}" for x, y in polygon) + "Z"
    plot_select(dash, [{"type": "path", "path": path}], inside)


def emitted_code(dash) -> str:
    entries = dash.processor.step_history
    return "\n".join(CODE[e["step"]](e["params"]) for e in entries)


def run_code(code: str, selections: dict) -> anndata.AnnData:
    saved = original()  # what the dashboard saved: its uns holds long polygons, masks
    saved.uns["sclab"] = {"selections": selections}
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/saved.h5ad"
        saved.write_h5ad(path)
        code = code.replace(f'"{SELECTIONS_FILE}"', f'"{path}"')
        namespace = {"adata": original(), "np": np, "pd": pd, "sclab": sclab}
        exec(code, namespace)
    return namespace["adata"]


def assert_reproduced(dash, label_columns=()) -> anndata.AnnData:
    adata = dash.dataset.adata
    result = run_code(emitted_code(dash), adata.uns["sclab"].get("selections", {}))
    assert result.obs_names.tolist() == adata.obs_names.tolist()
    for column in label_columns:
        pd.testing.assert_series_equal(result.obs[column], adata.obs[column])
    return result


def keep(dash):
    n_hist = len(dash.processor.step_history)
    dash.processor.dspr_keep_selected_click_callback()
    assert len(dash.processor.step_history) == n_hist + 1
    entry = dash.processor.step_history[-1]
    assert entry["status"] == "success", entry["error"]
    return entry["params"]


def label(dash, column, value, new=True):
    lab = dash.processor.selection_labeling_controls_dict
    lab["create_new_key_checkbox"].value = not new
    lab["create_new_key_checkbox"].value = new  # refreshes the key options
    key, val = (
        ("new_medatadata_key", "new_label")
        if new
        else ("existing_metadata_key", "existing_label")
    )
    lab[key].value = column
    lab[val].value = value
    dash.processor.dspr_apply_label_click_callback()
    entry = dash.processor.step_history[-1]
    assert entry["step"] == "apply_label" and entry["status"] == "success"
    return entry["params"]


def test_category_filter(dash):
    dash.processor.selection_controls_dict["cat"].value = ("a", "c")
    params = keep(dash)
    assert params == {"filters": {"cat": {"isin": ["a", "c"]}}, "n": 20}
    assert 'adata.obs["cat"].isin(["a", "c"])' in emitted_code(dash)
    assert_reproduced(dash)


def test_numeric_ranges_are_inclusive(dash):
    c = dash.processor.selection_controls_dict
    c["num_i"].value = (5, 30)
    c["num_f"].value = (2.0, 6.5)
    params = keep(dash)
    assert params["filters"] == {
        "num_f": {"between": [2.0, 6.5]},
        "num_i": {"between": [5, 30]},
    }
    assert 'adata.obs["num_i"].between(5, 30)' in emitted_code(dash)
    assert_reproduced(dash)


def test_bool_filter_drop(dash):
    dash.processor.selection_controls_dict["flag"].value = (True,)
    dash.processor.dspr_drop_selected_click_callback()
    assert dash.processor.step_history[-1]["params"]["drop"] is True
    assert dash.dataset.adata.n_obs == 20 and not dash.dataset.adata.obs["flag"].any()
    assert "adata = adata[~mask].copy()" in emitted_code(dash)
    assert_reproduced(dash)


def test_box(dash):
    box(dash, x=(-0.5, 1.0), y=(-1.0, 0.8))
    params = keep(dash)
    assert params["shape"] == {
        "id": "sel1",
        "type": "box",
        "basis": "X_umap",
        "dims": [0, 1],
        "x": [-0.5, 1.0],
        "y": [-1.0, 0.8],
    }
    code = emitted_code(dash)
    assert 'select_box(adata, "X_umap", x=(-0.5, 1.0), y=(-1.0, 0.8))' in code
    assert "read_adata" not in code  # a box is written out in full
    assert_reproduced(dash)


def test_box_label_on_existing_column(dash):
    box(dash, x=(-2.0, 0.0), y=(-2.0, 2.0))
    label(dash, "cat", "a", new=False)  # existing category
    box(dash, x=(0.0, 2.0), y=(-2.0, 2.0))
    params = label(dash, "cat", "new", new=True)  # same key: adds a category
    assert params["shape"]["id"] == "sel2" and params["label"] == "new"
    assert_reproduced(dash, ["cat"])


def test_lasso_keep_and_label(dash):
    lasso(dash)
    params = label(dash, "lab", "inside")
    assert params["shape"]["type"] == "lasso" and params["shape"]["polygon"] == LASSO
    keep(dash)  # same lasso: the stored shape is reused
    assert list(dash.dataset.adata.uns["sclab"]["selections"]) == ["sel1"]
    code = emitted_code(dash)
    assert "polygon = [\n    [-1.0, -1.0],\n" in code  # short: inlined
    assert "read_adata" not in code
    result = assert_reproduced(dash, ["lab"])
    assert (result.obs["lab"] == "inside").all()


def test_long_lasso_is_read_from_the_saved_adata(dash):
    t = np.linspace(0, 2 * np.pi, MAX_INLINE_VERTICES + 5, endpoint=False)
    lasso(dash, np.c_[1.5 * np.cos(t), 1.5 * np.sin(t)].round(3).tolist())
    label(dash, "lab", "inside")
    code = emitted_code(dash)
    assert f"# lasso with {len(t)} vertices" in code
    assert '.uns["sclab"]["selections"]["sel1"]["polygon"]' in code
    assert code.count("read_adata") == 1 and SELECTIONS_FILE in code
    result = assert_reproduced(dash, ["lab"])
    assert (result.obs["lab"] == "inside").sum() > 0


def test_filters_intersect_lasso(dash):
    dash.processor.selection_controls_dict["cat"].value = ("a", "b")
    lasso(dash)
    xy = original().obsm["X_umap"]
    expected = Path(LASSO).contains_points(xy) & original().obs["cat"].isin(["a", "b"])
    assert dash.dataset.selected_rows_mask.tolist() == expected.tolist()
    params = keep(dash)
    assert params["filters"] == {"cat": {"isin": ["a", "b"]}} and "shape" in params
    assert params["n"] == expected.sum() > 0
    code = emitted_code(dash)
    assert (
        'mask = adata.obs["cat"].isin(["a", "b"])\nmask &= sclab.tools.select_p' in code
    )
    assert_reproduced(dash)


def test_point_picks_are_recorded_as_a_mask(dash):
    picked = ["c03", "c07", "c11"]
    dash.dataset.selected_rows = pd.Index(picked)  # e.g. set by code
    params = keep(dash)
    assert params == {"mask": "sel1", "n": 3}
    stored = dash.dataset.adata.uns["sclab"]["selections"]["sel1"]
    assert stored["type"] == "mask" and stored["mask"].dtype == bool
    assert dash.dataset.adata.obs_names.tolist() == picked
    assert_reproduced(dash)


def test_several_selections_fall_back_to_mask(dash):
    # no single shape: the traces' selectedpoints are used, recorded as a mask
    inside = np.zeros(40, dtype=bool)
    inside[:5] = True
    plot_select(dash, [rect((-9, 0), (-9, 9)), rect((0, 9), (-9, 9))], inside)
    assert dash.processor._plot_shape is None
    params = keep(dash)
    assert set(params) == {"mask", "n"} and params["n"] == 5
    assert_reproduced(dash)


def test_box_on_obs_columns(dash):
    ctrl = dash.plotter.controls
    ctrl.data_key.value = "metadata"
    ctrl.selected_axes.children[0].value = "num_f"
    ctrl.selected_axes.children[1].value = "num_i"
    obs = dash.dataset.adata.obs
    inside = (
        obs["num_f"].between(1, 5).to_numpy() & obs["num_i"].between(0, 15).to_numpy()
    )
    plot_select(dash, [rect((1, 5), (0, 15))], np.zeros(40, bool))
    params = keep(dash)
    assert params["n"] == inside.sum() > 0
    assert (
        params["shape"]["dims"] == ["num_f", "num_i"] and "basis" not in params["shape"]
    )
    assert (
        "select_box(\n    adata,\n    None,\n    x=(1.0, 5.0),\n    y=(0.0, 15.0),\n"
        '    dims=("num_f", "num_i"),\n)' in emitted_code(dash)
    )
    assert_reproduced(dash)


def test_log_axes_selection_has_no_shape(dash):
    dash.plotter.controls.log_axes.children[0].value = True
    inside = np.zeros(40, dtype=bool)
    inside[np.flatnonzero(original().obsm["X_umap"][:, 0] > 0)[:2]] = True  # plotted
    lasso(dash, inside=inside)
    assert len(dash.processor._plot_selection) == 2
    assert dash.processor._plot_shape is None
    assert keep(dash) == {"mask": "sel1", "n": 2}


def test_keep_without_selection_does_nothing(dash):
    dash.processor.dspr_keep_selected_click_callback()
    assert dash.dataset.adata.n_obs == 40 and not dash.processor.step_history


def test_sequence_of_steps_and_h5ad_round_trip(dash, tmp_path):
    dash.processor.selection_controls_dict["cat"].value = ("a", "b", "c")
    keep(dash)
    dash.processor.dspr_clear_selection_click_callback()
    lasso(dash)
    label(dash, "lab", "L")
    dash.dataset.selected_rows = dash.dataset.adata.obs_names[::2]
    dash.processor.dspr_drop_selected_click_callback()

    path = tmp_path / "a.h5ad"
    dash.dataset.adata.write_h5ad(path)
    read = anndata.read_h5ad(path)
    selections = read.uns["sclab"]["selections"]
    assert set(selections) == {"sel1", "sel2"}
    np.testing.assert_array_equal(
        selections["sel1"]["polygon"],
        dash.dataset.adata.uns["sclab"]["selections"]["sel1"]["polygon"],
    )
    result = run_code(emitted_code(dash), selections)
    assert result.obs_names.tolist() == read.obs_names.tolist()
    pd.testing.assert_series_equal(result.obs["lab"], read.obs["lab"])


def test_history_file_replay_and_count_check(dash, tmp_path, caplog):
    dash.processor.selection_controls_dict["num_i"].value = (0, 30)
    lasso(dash)
    keep(dash)
    path = tmp_path / "history.json"
    dash.processor.save_step_history(path)

    d2 = SCLabDashboard(original())
    d2.processor.load_step_history(path)
    d2.processor.apply_step_history()  # shape geometry travels in the params
    assert d2.dataset.adata.obs_names.equals(dash.dataset.adata.obs_names)
    assert emitted_code(d2) == emitted_code(dash)
    assert_reproduced(d2)  # the replay stored the lasso in uns for the code

    other = original()
    other.obs["num_i"] = other.obs["num_i"][::-1].to_numpy()
    d3 = SCLabDashboard(other)
    d3.processor.load_step_history(path)
    with caplog.at_level(logging.WARNING):
        d3.processor.apply_step_history()
    assert "recorded" in caplog.text


def test_legacy_entries_and_mask_without_uns(dash, tmp_path):
    assert filter_rows_code({"index": ["c00"]}).startswith("# filter_rows:")
    assert apply_label_code({"index": ["c00"], "column": "a", "label": "b"})[0] == "#"
    dash.dataset.selected_rows = pd.Index(["c00", "c01"])
    keep(dash)
    path = tmp_path / "history.json"
    dash.processor.save_step_history(path)
    d2 = SCLabDashboard(original())  # its uns lacks the mask
    d2.processor.load_step_history(path)
    d2.processor.apply_step_history()
    entry = d2.processor.step_history[-1]
    assert entry["status"] == "failed" and "'sel1' not in adata.uns" in entry["error"]
