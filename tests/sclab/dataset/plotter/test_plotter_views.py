"""Pinned plot views: capture, restore, stale data, adata.uns persistence."""

from collections import Counter

import anndata as ad
import pandas as pd
import pytest

from sclab import SCLabDashboard
from sclab.dataset.plotter import Plotter
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


def test_restore_clears_a_gene_plot_that_cannot_be_pinned(dash):
    p = dash.plotter
    spec = p.views.pin()
    controls = dash.processor.steps["Gene Expression"].variable_controls
    controls["layer"].value = "counts"
    controls["gene_input"].value = controls["gene_input"].options[0]
    run_step(dash, "Gene Expression")
    assert p.state["source"] == "gene"

    assert p.views.pin() is None  # phase 1: gene/custom plots are not pinned
    assert "cannot be pinned" in p.views.message.value
    assert len(p.views.views) == 1
    assert problems(p, spec) == []  # plotting a gene does not change the data

    p.views.restore(spec["id"])
    assert p.state["source"] == "controls"
    assert values(p) == spec["controls"]


def test_missing_color_column_restores_none_with_warning(dash):
    p, adata = dash.plotter, dash.dataset.adata
    adata.obs["tmp"] = pd.Categorical(["a", "b", "c"] * (adata.n_obs // 3))
    dash.broker.publish("dset_metadata_change", dash.dataset.metadata)
    p.controls.color.value = "tmp"
    spec = p.views.pin()

    del adata.obs["tmp"]
    dash.broker.publish("dset_metadata_change", dash.dataset.metadata)
    assert problems(p, spec) == ["color 'tmp' missing"]
    assert "color &#x27;tmp&#x27; missing" in p.views._cards[spec["id"]][-1].value

    assert p.views.restore(spec["id"]) == ["color 'tmp' missing"]
    assert p.controls.color.value is None
    assert p.controls.data_key.value == spec["controls"]["data_key"]


def test_missing_data_key_disables_the_card(dash):
    p = dash.plotter
    spec = p.views.pin()
    spec["controls"]["data_key"] = "X_gone"
    p.views._render()
    restore_button = p.views._cards[spec["id"]][-2]
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
    assert "data changed since pin" in p.views._cards[spec["id"]][-1].value
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
    name_input = p.views._cards[spec["id"]][2]
    name_input.value = "renamed"
    assert loads_uns(dash.dataset.adata)[0]["name"] == "renamed"
    p.views._cards[spec["id"]][3].click()  # delete
    assert loads_uns(dash.dataset.adata) == []
    assert p.views.cards.children == (p.views.pin_button,)


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
