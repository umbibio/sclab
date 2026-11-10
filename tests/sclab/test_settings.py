"""The Settings tab: each setting applies, keeps user choices, and persists."""

import anndata
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.io as pio
import pytest

from sclab import SCLabDashboard, Workspace
from sclab.dataset.plotter import Plotter

from .test_background_jobs import FakeLoop
from .test_utils import run_step, simple_loop_adata


def make_adata(n_obs=200):
    adata = simple_loop_adata(n_obs=n_obs, n_vars=40)
    rng = np.random.default_rng(0)
    adata.layers["raw"] = adata.X.copy()
    adata.obs["sample"] = pd.Categorical(rng.choice(["s1", "s2"], adata.n_obs))
    adata.obs["donor"] = pd.Categorical(rng.choice(["d1", "d2", "d3"], adata.n_obs))
    return adata


def layer_of(dash, step):
    return dash.processor.steps[step].variable_controls["layer"].value


def test_counts_layer_reaches_the_steps_and_their_validation():
    dash = SCLabDashboard(make_adata())
    steps = dash.processor.steps
    de = steps["Differential Expression"]
    de.variable_controls["method"].value = "pseudobulk_limma"
    de.variable_controls["groupby"].value = "donor"
    assert de.variable_controls["override_layer"].description.endswith("(counts)")

    dash.counts_layer_dropdown.value = "raw"
    assert dash.dataset.counts_layer == "raw"
    assert layer_of(dash, "QC") == layer_of(dash, "Preprocess") == "raw"
    if "Doublet Detection" in steps:
        assert layer_of(dash, "Doublet Detection") == "raw"
    assert de.variable_controls["override_layer"].description.endswith("(raw)")
    assert layer_of(dash, "Differential Expression") == "raw"
    assert steps["PCA"].resolved_params({})["counts_layer"] == "raw"

    # validation re-runs: the counts layer is gone, Doublet Detection cannot run
    del dash.dataset.adata.layers["raw"]
    dash.broker.publish("dset_anndata_layers_change")
    assert dash.counts_layer_dropdown.value == "raw"  # kept: not a new setting
    if "Doublet Detection" in steps:
        assert "No counts layer 'raw'" in steps["Doublet Detection"].status.value
    dash.dataset.adata.layers["lognorm"] = np.log1p(dash.dataset.adata.layers["counts"])
    dash.broker.publish("dset_anndata_layers_change")
    dash.counts_layer_dropdown.value = "lognorm"
    assert "does not look like raw counts" in de.status.value
    assert not dash.broker.exceptions_log


def test_counts_layer_keeps_the_layers_the_user_chose():
    adata = make_adata()
    adata.layers["other"] = adata.X.copy()
    dash = SCLabDashboard(adata)
    qc = dash.processor.steps["QC"].variable_controls["layer"]
    qc.value = "raw"  # chosen: differs from the setting
    dash.counts_layer_dropdown.value = "other"
    assert qc.value == "raw"
    assert layer_of(dash, "Preprocess") == "other"  # followed the setting


def test_batch_key_reaches_integration_and_pca():
    dash = SCLabDashboard(make_adata(), batch_key="sample")
    group_by = dash.processor.steps["Integration"].variable_controls["group_by"]
    reference = dash.processor.steps["PCA"].variable_controls["reference_batch"]
    assert group_by.value == "sample" and dash.batch_key_dropdown.value == "sample"

    dash.batch_key_dropdown.value = "donor"
    assert dash.processor.batch_key == "donor"
    assert group_by.value == "donor"
    assert list(reference.options) == ["", "d1", "d2", "d3"]
    assert dash.processor.steps["Neighbors"].resolved_params({})["batch_key"] == "donor"

    group_by.value = "sample"  # chosen: kept
    dash.batch_key_dropdown.value = ""
    assert dash.processor.batch_key is None and group_by.value == "sample"
    assert not dash.broker.exceptions_log


def test_background_off_runs_steps_on_the_main_thread():
    dash = SCLabDashboard(make_adata())
    run_step(dash, "QC")
    run_step(dash, "Preprocess")
    loop = dash.broker.jobs.loop = FakeLoop()  # as in a kernel: steps go background
    dash.background_checkbox.value = False
    pca = dash.processor.steps["PCA"]
    assert pca.background
    pca.run_button.click()
    assert not dash.broker.jobs.busy and not loop.timers  # done, synchronously
    assert dash.processor.step_history[-1]["step"] == "PCA"
    assert "X_pca" in dash.dataset.adata.obsm

    dash.background_checkbox.value = True
    pca.run_button.click()
    assert dash.broker.jobs.busy  # waits for the loop again
    loop.pump(dash.broker.jobs)
    assert not dash.broker.exceptions_log


def test_seed_sets_random_state_unless_the_user_did():
    dash = SCLabDashboard(make_adata())
    steps = dash.processor.steps
    seeded = ["PCA", "Neighbors", "UMAP", "Cluster", "Integration"]
    seeded += ["Doublet Detection", "Differential Expression"]
    dash.seed_input.value = "42"
    for name in seeded:
        assert steps[name].fixed_params["random_state"] == 42, name
    assert "random_state" not in steps["QC"].fixed_params

    steps["UMAP"].fixed_params["random_state"] = 7  # set by the user
    dash.seed_input.value = "5"
    assert steps["UMAP"].fixed_params["random_state"] == 7
    assert steps["PCA"].fixed_params["random_state"] == 5

    run_step(dash, "QC")
    run_step(dash, "Preprocess")
    run_step(dash, "PCA")
    assert dash.processor.step_history[-1]["params"]["random_state"] == 5
    assert "random_state=5" in dash.generate_code()

    dash.seed_input.value = ""  # empty: back to each step's own default
    assert "random_state" not in steps["PCA"].fixed_params
    assert steps["UMAP"].fixed_params["random_state"] == 7  # still the user's


def test_seed_zero_is_a_seed_and_invalid_text_does_not_apply():
    dash = SCLabDashboard(make_adata())
    pca = dash.processor.steps["PCA"].fixed_params
    dash.seed_input.value = "0"
    assert pca["random_state"] == 0
    assert dash.dataset.adata.uns["sclab"]["settings"]["random_state"] == 0
    dash.seed_input.value = "4"
    assert pca["random_state"] == 4  # 0 was applied, so it was followed

    dash.seed_input.value = "four"
    assert pca["random_state"] == 4 and dash.seed_message.value
    assert dash.dataset.adata.uns["sclab"]["settings"]["random_state"] == 4
    dash.seed_input.value = " 5 "
    assert pca["random_state"] == 5 and not dash.seed_message.value

    pca["random_state"] = 0  # chosen by the user while the setting is empty
    dash.seed_input.value = ""
    dash.seed_input.value = "9"
    assert pca["random_state"] == 0


@pytest.fixture
def redraws(monkeypatch):
    calls = []
    make_new_figure = Plotter.make_new_figure

    def counting(self, *args, **kwargs):
        calls.append(1)
        return make_new_figure(self, *args, **kwargs)

    monkeypatch.setattr(Plotter, "make_new_figure", counting)
    return calls


def test_theme_colorscale_and_palette_reach_the_figure(redraws):
    dash = SCLabDashboard(make_adata())
    p = dash.plotter
    p.controls.data_key.value = "Y_array2"
    p.controls.color.value = "time"  # numeric
    redraws.clear()

    p.template_dropdown.value = "plotly_dark"
    assert len(redraws) == 1
    dark = pio.templates["plotly_dark"].layout.paper_bgcolor
    assert p.g.layout.template.layout.paper_bgcolor == dark

    p.colorscale_dropdown.value = "plasma"
    assert len(redraws) == 2
    assert p.g.layout.coloraxis.colorscale[0][1] == pc.sequential.Plasma[0]

    p.controls.color.value = "donor"
    redraws.clear()
    p.palette_dropdown.value = "D3"
    assert len(redraws) == 1
    colors = {t.name: t.marker.color for t in p.g.data if t.name}
    assert colors["d1"] == pc.qualitative.D3[0]
    assert colors["d3"] == pc.qualitative.D3[2]

    p.backend = "static"  # the palette and colorscale reach the other backends
    spec = p._view.frame["color"]
    assert spec["colors"]["d1"] == pc.qualitative.D3[0]
    p.controls.color.value = "time"
    assert p._view.frame["color"]["colors"][0] == pc.sequential.Plasma[0]
    assert not dash.broker.exceptions_log


def test_keep_tables_toggles_unload():
    dash = SCLabDashboard(make_adata())
    de = dash.processor.steps["Differential Expression"].results.table_output
    outputs = [dash.dataset.obs_table_output, dash.dataset.var_table_output, de]
    assert all(o.unload for o in outputs)
    dash.main_content.selected_index = 2  # Observations: rendered
    dash.keep_tables_checkbox.value = True
    assert not any(o.unload for o in outputs)
    dash.main_content.selected_index = 0
    assert dash.dataset.obs_table_output.outputs  # kept while hidden
    dash.keep_tables_checkbox.value = False
    assert all(o.unload for o in outputs)
    assert not dash.dataset.obs_table_output.outputs  # hidden: unloaded now


def change_settings(dash):
    dash.counts_layer_dropdown.value = "raw"
    dash.batch_key_dropdown.value = "donor"
    dash.background_checkbox.value = False
    dash.seed_input.value = "3"
    dash.plotter.backend = "static"
    dash.plotter.template_dropdown.value = "plotly_white"
    dash.plotter.colorscale_dropdown.value = "plasma"
    dash.plotter.palette_dropdown.value = "D3"
    dash.plotter.min_size_input.value = 0.5
    dash.keep_tables_checkbox.value = True


def assert_restored(dash):
    p = dash.plotter
    assert dash.dataset.counts_layer == "raw" and layer_of(dash, "QC") == "raw"
    assert dash.processor.batch_key == "donor"
    integration = dash.processor.steps["Integration"]
    assert integration.variable_controls["group_by"].value == "donor"
    assert dash.broker.jobs.threaded is False
    assert dash.processor.steps["PCA"].fixed_params["random_state"] == 3
    assert p.backend == "static"
    assert (p.template, p.colorscale, p.palette_dropdown.value) == (
        "plotly_white",
        "plasma",
        "D3",
    )
    assert p.min_size_input.value == 0.5
    assert dash.dataset.obs_table_output.unload is False


def test_settings_round_trip_through_h5ad(tmp_path):
    adata = make_adata()
    dash = SCLabDashboard(adata)
    assert "settings" not in adata.uns.get("sclab", {})  # opening is no change
    n_history = len(dash.processor.step_history)
    change_settings(dash)
    settings = adata.uns["sclab"]["settings"]
    assert settings["template"] == "plotly_white" and settings["background"] is False
    assert len(dash.processor.step_history) == n_history  # not history entries

    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    assert_restored(SCLabDashboard(anndata.read_h5ad(path)))


@pytest.mark.parametrize("seed, stored", [("", None), ("0", 0)])
def test_seed_round_trips_through_h5ad(tmp_path, seed, stored):
    adata = make_adata()
    dash = SCLabDashboard(adata)
    dash.seed_input.value = "1"  # a change is needed to store anything
    dash.seed_input.value = seed
    assert adata.uns["sclab"]["settings"]["random_state"] == stored
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert back.uns["sclab"]["settings"]["random_state"] == stored

    dash = SCLabDashboard(back)
    assert dash.seed_input.value == seed
    fixed = dash.processor.steps["PCA"].fixed_params
    assert fixed.get("random_state") == stored  # empty: the key is absent


def test_arguments_win_over_stored_settings():
    adata = make_adata()
    change_settings(SCLabDashboard(adata))
    dash = SCLabDashboard(
        adata, counts_layer="counts", batch_key="sample", plot_backend="plotly"
    )
    assert dash.dataset.counts_layer == "counts"
    assert dash.processor.batch_key == "sample"
    assert dash.plotter.backend == "plotly"
    assert dash.plotter.template == "plotly_white"  # not an argument: stored

    adata.obs = adata.obs.drop(columns="donor")  # stale settings: the defaults
    del adata.layers["raw"]
    dash = SCLabDashboard(adata)
    assert dash.dataset.counts_layer == "counts"
    assert dash.processor.batch_key is None
    assert "raw" not in dash.dataset.adata.layers  # not made from X


def test_workspace_reopen_restores_settings(tmp_path):
    paths = {}
    for name in ("a", "b"):
        paths[name] = tmp_path / f"{name}.h5ad"
        make_adata().write_h5ad(paths[name])
    ws = Workspace(paths, workspace_dir=tmp_path / "ws")
    change_settings(ws.dash)
    assert ws.is_dirty("a")
    ws.activate("b")
    assert ws.datasets["a"].adata is None  # written and freed
    assert ws.dash.plotter.template == "simple_white"  # b keeps its own
    ws.activate("a")
    assert_restored(ws.dash)
