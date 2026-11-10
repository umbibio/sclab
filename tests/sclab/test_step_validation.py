"""Steps refuse to run without their inputs, and show their errors in their panel."""

import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import sclab.preprocess
from sclab import SCLabDashboard

from .test_utils import run_step, simple_loop_adata


@pytest.fixture
def raw():
    """A dashboard on raw counts: nothing computed, no batch column."""
    return SCLabDashboard(simple_loop_adata(n_obs=120, n_vars=40))


def assert_refused(dash, step_name: str, reason: str, **params):
    step = dash.processor.steps[step_name]
    n_hist = len(dash.processor.step_history)
    assert step.run_button.disabled
    assert reason in step.status.value
    with pytest.raises(ValueError, match=reason):
        step.run(**params)
    step.run_button.click()  # e.g. a stale frontend: refused too, and shown
    assert len(dash.processor.step_history) == n_hist  # nothing recorded
    assert "details in Logs" in step.status.value


@pytest.mark.parametrize(
    "step_name, reason",
    [
        ("PCA", "Run the Preprocess step first"),
        ("Integration", "Choose a 'GroupBy' column with at least 2 batches"),
        ("Neighbors", "Choose a 'Use rep.' representation"),
        ("UMAP", "Run the Neighbors step first"),
        ("Cluster", "Run the Neighbors step first"),
        ("Transfer Metadata", "Choose a 'Group Key'"),
        ("Differential Expression", "Choose a 'Group by' column"),
        ("Gene Expression", "Choose a gene"),
        ("Guided Pseudotime", "Choose a 'Use rep.' representation"),
    ],
)
def test_missing_inputs_disable_the_step(raw, step_name, reason):
    assert_refused(raw, step_name, reason)


def test_steps_with_valid_defaults_are_enabled(raw):
    for name in ["QC", "Preprocess", "Doublet Detection"]:
        step = raw.processor.steps.get(name)
        if step is None:  # no doublet detection backend installed
            continue
        assert not step.run_button.disabled, name
        assert step.status.value == "", name


def test_layer_and_column_params_are_checked(raw):
    with pytest.raises(ValueError, match="Choose a 'Layer' with raw counts"):
        raw.processor.steps["QC"].run(layer="nope")
    with pytest.raises(ValueError, match="No column 'nope'"):
        raw.processor.steps["Preprocess"].run(group_by="nope")
    if "Doublet Detection" in raw.processor.steps:
        step = raw.processor.steps["Doublet Detection"]
        with pytest.raises(ValueError, match="Choose a layer with raw counts"):
            step.run(override_layer=True, layer=None)
    assert raw.processor.step_history == []


def test_differential_expression_needs_two_groups_and_a_valid_layer(raw):
    adata = raw.dataset.adata
    adata.obs["one"] = pd.Categorical(["a"] * adata.n_obs)
    adata.obs["two"] = pd.Categorical(["a", "b"] * (adata.n_obs // 2))
    raw.broker.publish("dset_metadata_change", raw.dataset.metadata)
    step = raw.processor.steps["Differential Expression"]
    c = step.variable_controls

    c["groupby"].value = "one"
    assert step.run_button.disabled and "single group" in step.status.value
    c["groupby"].value = "two"  # wilcoxon on adata.X, which holds raw counts here
    assert step.run_button.disabled
    assert "adata.X looks like raw counts" in step.status.value
    c["method"].value = "pseudobulk_edger"  # the counts layer: valid
    assert not step.run_button.disabled and step.status.value == ""
    c["groups"].value = ("a",)
    c["reference"].value = "a"
    assert "other than the reference" in step.status.value
    c["reference"].value = "b"  # one group plus a reference
    assert not step.run_button.disabled
    with pytest.raises(ValueError, match="No layer 'nope'"):
        step.run(override_layer=True, layer="nope")
    with pytest.raises(ValueError, match="Choose a layer"):
        step.run(override_layer=True, layer=None)
    with pytest.raises(ValueError, match="No column 'nope'"):
        step.run(sample_key="nope")
    with pytest.raises(ValueError, match="No column 'nope'"):
        step.run(cell_identity="nope")
    assert raw.processor.step_history == []


def test_differential_expression_layer_must_suit_the_method(raw):
    adata = raw.dataset.adata
    adata.obs["two"] = pd.Categorical(["a", "b"] * (adata.n_obs // 2))
    adata.layers["lognorm"] = np.log1p(adata.layers["counts"])
    raw.broker.publish("dset_metadata_change", raw.dataset.metadata)
    raw.broker.publish("dset_anndata_layers_change")
    step = raw.processor.steps["Differential Expression"]
    c = step.variable_controls
    c["groupby"].value = "two"

    c["override_layer"].value = True
    c["layer"].value = "lognorm"  # wilcoxon on log-normalized data: valid
    assert not step.run_button.disabled, step.status.value
    c["layer"].value = "counts"
    assert "Layer 'counts' looks like raw counts" in step.status.value
    c["method"].value = "pseudobulk_limma"
    assert not step.run_button.disabled, step.status.value
    c["layer"].value = "lognorm"
    assert "Layer 'lognorm' does not look like raw counts" in step.status.value
    assert "pseudobulk_limma needs" in step.status.value

    c["override_layer"].value = False
    del adata.layers["counts"]  # no counts layer: pseudobulk cannot run
    raw.broker.publish("dset_anndata_layers_change")
    assert step.run_button.disabled
    assert "No counts layer 'counts': tick 'Override layer'" in step.status.value


def test_differential_expression_pseudobulk_needs_the_de_extra(raw, monkeypatch):
    import sclab.examples.processor_steps._differential_expression as de_step

    adata = raw.dataset.adata
    adata.obs["two"] = pd.Categorical(["a", "b"] * (adata.n_obs // 2))
    raw.broker.publish("dset_metadata_change", raw.dataset.metadata)
    step = raw.processor.steps["Differential Expression"]
    c = step.variable_controls
    c["groupby"].value = "two"

    monkeypatch.setattr(de_step, "edgepython_is_available", lambda: False)
    monkeypatch.setattr(de_step, "pylimma_is_available", lambda: False)
    assert set(c["method"].options) == {  # offered, and disabled with the reason
        "wilcoxon",
        "t-test",
        "logreg",
        "pseudobulk_limma",
        "pseudobulk_edger",
    }
    c["method"].value = "pseudobulk_edger"
    assert step.run_button.disabled
    assert "pseudobulk_edger needs edgepython" in step.status.value
    assert "sclab[de-edger]" in step.status.value
    c["method"].value = "pseudobulk_limma"
    assert "pseudobulk_limma needs pylimma" in step.status.value

    # pylimma without edgepython: edgeR preprocessing by sclab's own port
    monkeypatch.setattr(de_step, "pylimma_is_available", lambda: True)
    assert step.validate(step._control_params()) is None
    resolved = step.resolved_params(step._control_params())
    assert "edger_preprocessing" not in step.to_code(resolved)


def test_integration_from_the_report(raw):
    """No batches, no group-by: disabled with the reason; enabled with 2 batches."""
    step = raw.processor.steps["Integration"]
    c = step.variable_controls
    assert c["group_by"].value is None
    assert_refused(raw, "Integration", "with at least 2 batches")

    adata = raw.dataset.adata
    adata.obs["sample"] = pd.Categorical(["s1"] * adata.n_obs)
    raw.broker.publish("dset_metadata_change", raw.dataset.metadata)
    c["group_by"].value = "sample"
    assert step.run_button.disabled
    assert "'sample' has a single batch" in step.status.value

    adata.obs["sample"] = pd.Categorical(["s1", "s2"] * (adata.n_obs // 2))
    raw.broker.publish("dset_metadata_change", raw.dataset.metadata, "sample")
    assert not step.run_button.disabled  # cca on X
    assert step.status.value == ""

    c["flavor"].value = "harmony"
    assert "Choose a 'Use rep.'" in step.status.value
    adata.obsm["X_emb"] = np.random.default_rng(0).normal(size=(adata.n_obs, 3))
    raw.broker.publish("dset_data_dict_change", raw.dataset.data_dict)
    c["use_rep"].value = "X_emb"
    assert not step.run_button.disabled
    c["reference_batch"].value = "s1"
    assert not step.run_button.disabled
    c["flavor"].value = "cca"
    assert "CCA does not take a reference batch" in step.status.value


def test_integration_defaults_to_the_batch_key():
    adata = simple_loop_adata(n_obs=60, n_vars=20)
    adata.obs["sample"] = pd.Categorical(["s1", "s2"] * 30)
    d = SCLabDashboard(adata, batch_key="sample")
    step = d.processor.steps["Integration"]
    assert step.variable_controls["group_by"].value == "sample"
    assert not step.run_button.disabled


def test_neighbors_enable_the_graph_steps(raw):
    adata = raw.dataset.adata
    adata.uns["neighbors"] = {}
    adata.obsp["connectivities"] = sp.identity(adata.n_obs, format="csr")
    raw.broker.publish("dset_anndata_neighbors_change")
    for name in ["UMAP", "Cluster"]:
        assert not raw.processor.steps[name].run_button.disabled, name


@pytest.fixture
def batched():
    adata = simple_loop_adata(n_obs=120, n_vars=40)
    adata.obs["batch"] = pd.Categorical(["b1", "b2"] * 60)
    adata.obsm["X_emb"] = np.random.default_rng(0).normal(size=(120, 3))
    d = SCLabDashboard(adata)
    d.processor.steps["Integration"].variable_controls["use_rep"].value = "X_emb"
    return d


def test_sync_error_is_shown_in_the_panel_and_cleared(batched, monkeypatch):
    def boom(adata, **kwargs):
        raise RuntimeError("integration exploded\nsecond line")

    monkeypatch.setattr(sclab.preprocess, "cca_integrate", boom)
    step = batched.processor.steps["Integration"]
    assert not step.run_button.disabled
    step.run_button.click()
    assert step.run_button.button_style == "danger"
    assert not step.run_button.disabled  # inputs still valid: can retry
    assert "RuntimeError: integration exploded (details in Logs)" in step.status.value
    assert "second line" not in step.status.value
    assert batched.processor.step_history[-1]["status"] == "failed"

    step.variable_controls["max_iters"].value = 5  # the inputs changed
    assert step.status.value == ""

    step.run_button.click()  # fails again
    assert "exploded" in step.status.value
    monkeypatch.undo()
    run_step(batched, "Integration")
    assert step.status.value == ""


def test_event_callback_error_is_shown_in_its_step(batched):
    step = batched.processor.steps["PCA"]

    def fail(self, *args, **kwargs):
        raise KeyError("in a callback")

    # a step's dataset callback: the broker shows its error in the step's panel
    batched.broker.subscribe("dset_total_vars_change", types.MethodType(fail, step))
    batched.broker.publish("dset_total_vars_change", batched.dataset.adata.var)
    assert any("in a callback" in e for e in batched.broker.exceptions_log)
    assert "KeyError: 'in a callback' (details in Logs)" in step.status.value


def test_control_callback_error_is_shown(batched):
    step = batched.processor.steps["Gene Expression"]
    c = step.variable_controls
    c["layer"].value = "counts"
    del batched.dataset.adata.layers["counts"]  # behind the dashboard's back
    c["gene_input"].value = c["gene_input"].options[0]  # replots: fails
    assert "KeyError" in step.status.value and "details in Logs" in step.status.value
    assert batched.broker.exceptions_log


def test_selection_action_error_is_shown_below_the_buttons(batched, monkeypatch):
    processor = batched.processor
    batched.dataset.selected_rows = batched.dataset.adata.obs_names[:10]

    def fail(*args, **kwargs):
        raise RuntimeError("cannot filter")

    monkeypatch.setattr(batched.dataset, "filter_rows", fail)
    keep, _ = processor.selection_buttons_dict["keep_drop_buttons"].children
    keep.click()
    assert (
        "RuntimeError: cannot filter (details in Logs)"
        in processor.selection_status.value
    )
    assert processor.step_history[-1]["status"] == "failed"

    monkeypatch.undo()
    keep.click()
    assert processor.selection_status.value == ""
    assert batched.dataset.adata.n_obs == 10


def test_output_widgets_do_not_swallow_step_errors_in_a_kernel(batched, monkeypatch):
    """In a kernel ``with Output():`` swallows exceptions: the user's report."""
    import ipywidgets.widgets.widget_output as widget_output

    class FakeShell:  # get_ipython() in a kernel: Output.__exit__ shows and swallows
        def showtraceback(self, *args, **kwargs):
            pass

    monkeypatch.setattr(widget_output, "get_ipython", lambda: FakeShell())

    def boom(adata, **kwargs):
        raise RuntimeError("integration exploded")

    monkeypatch.setattr(sclab.preprocess, "cca_integrate", boom)
    step = batched.processor.steps["Integration"]
    step.run_button.click()
    assert batched.processor.step_history[-1]["status"] == "failed"
    assert "RuntimeError: integration exploded" in step.status.value
    assert "X_cca" not in batched.dataset.adata.obsm
