"""Drive the example processor steps through a real SCLabDashboard.

``ProcessorStepBase.run`` swallows exceptions (it only colours the run button
and logs the traceback), so every test here asserts on ``run_button.button_style``
and on the resulting ``adata`` state instead of relying on "did not raise".
"""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from sclab import SCLabDashboard
from sclab.preprocess import pca as library_pca
from sclab.preprocess import preprocess as library_preprocess
from sclab.preprocess import qc as library_qc
from sclab.utils import layer_keys
from sclab.examples.processor_steps._differential_expression import (
    _rank_genes_groups_df,
)

from ..test_utils import run_step as strict_run_step
from ..test_utils import simple_loop_adata


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_adata(n_obs=400, n_vars=400) -> AnnData:
    adata = simple_loop_adata(n_obs=n_obs, n_vars=n_vars)
    adata.obs["batch"] = pd.Categorical(
        np.where(np.arange(adata.n_obs) % 2 == 0, "b1", "b2")
    )
    adata.obs["grp"] = pd.cut(adata.obs["time"], bins=3, labels=["A", "B", "C"]).astype(
        "category"
    )
    return adata


def run_step(dash: SCLabDashboard, step_name: str, **params):
    name = step_name
    step = dash.processor.steps[name]
    n_hist = len(dash.processor.step_history)
    step.run(**params)
    ok = step.run_button.button_style == "success"
    if ok:
        assert len(dash.processor.step_history) == n_hist + 1
        assert dash.processor.step_history[-1]["step"] == name
    return step, ok


def capture(dash: SCLabDashboard, event: str) -> list:
    """Record every ``publish(event, ...)``.

    A spy on ``publish`` is used rather than a subscriber: a failing earlier
    subscriber aborts the dispatch loop, which would hide the event.
    """
    calls = []
    original = dash.broker.publish

    def spy(ev, *args, **kwargs):
        if ev == event:
            calls.append((args, kwargs))
        return original(ev, *args, **kwargs)

    dash.broker.publish = spy
    return calls


@pytest.fixture(scope="module")
def dash() -> SCLabDashboard:
    d = SCLabDashboard(make_adata())
    for name in ["QC", "Preprocess", "PCA"]:
        _, ok = run_step(d, name)
        assert ok, name
    return d


# --------------------------------------------------------------------------
# Layer dropdowns
# --------------------------------------------------------------------------
def test_layer_dropdown_options_never_include_none(dash):
    # anndata>=0.13 exposes X as adata.layers[None]; it must not leak into options
    assert None not in layer_keys(dash.dataset.adata)
    dash.processor.dset_anndata_layers_change_callback()
    seen = 0
    for step in dash.processor.steps.values():
        for control in step.variable_controls.values():
            if getattr(control, "description", "").strip(" :.").lower() != "layer":
                continue
            seen += 1
            values = list(control.options)
            assert None not in values
            assert set(values) <= set(layer_keys(dash.dataset.adata))
    assert seen


# --------------------------------------------------------------------------
# QC / Preprocess: thin wrappers over the library functions
# --------------------------------------------------------------------------
def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class TestQCPreprocessMatchLibrary:
    @pytest.mark.parametrize(
        "step_params, library_params",
        [
            ({}, {}),
            (
                {"group_by": "batch", "normalization_method": "Weighted library size"},
                {"group_by": "batch", "normalization_method": "weighted"},
            ),
            (
                {
                    "normalization_method": "No normalization",
                    "log1p": False,
                    "scale": False,
                    "regress_n_genes": True,
                },
                {
                    "normalization_method": "none",
                    "log1p": False,
                    "scale": False,
                    "regress_n_genes": True,
                },
            ),
        ],
    )
    def test_step_output_equals_library_call(self, step_params, library_params):
        d = SCLabDashboard(make_adata(300, 300))
        expected = d.dataset.adata.copy()
        qc_params = dict(min_counts=50, min_genes=5, min_cells=0, max_rank=250)

        strict_run_step(d, "QC", **qc_params)
        library_qc(expected, counts_layer="counts", **qc_params)
        strict_run_step(d, "Preprocess", **step_params)
        library_preprocess(expected, counts_layer="counts", **library_params)

        got = d.dataset.adata
        assert got.obs_names.equals(expected.obs_names)
        assert got.var_names.equals(expected.var_names)
        np.testing.assert_array_equal(_dense(got.X), _dense(expected.X))
        assert got.X.dtype == expected.X.dtype
        assert layer_keys(got) == layer_keys(expected)
        for key in layer_keys(got):
            np.testing.assert_array_equal(
                _dense(got.layers[key]), _dense(expected.layers[key])
            )
        pd.testing.assert_series_equal(
            got.var["highly_variable"], expected.var["highly_variable"]
        )

    def test_history_records_resolved_layer(self):
        d = SCLabDashboard(make_adata(300, 300))
        strict_run_step(d, "QC", layer=None)
        strict_run_step(d, "Preprocess", layer=None)
        qc_entry, pp_entry = d.processor.step_history
        assert qc_entry["params"]["layer"] == pp_entry["params"]["layer"] == "counts"
        assert pp_entry["shape_before"] == qc_entry["shape_after"]

    def test_unknown_normalization_method_fails(self):
        d = SCLabDashboard(make_adata(300, 300))
        _, ok = run_step(d, "Preprocess", normalization_method="library")
        assert not ok
        assert d.processor.step_history[-1]["status"] == "failed"

    def test_qc_filtering_refreshes_row_and_gene_dependents(self):
        d = SCLabDashboard(make_adata(300, 300))
        totals = np.asarray(d.dataset.adata.layers["counts"].sum(axis=1)).ravel()
        rows = capture(d, "dset_total_rows_change")
        strict_run_step(d, "QC", min_counts=int(np.median(totals)))
        assert 0 < d.dataset.adata.n_obs < 300 and len(rows) == 1


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------
class TestPCA:
    def test_history_records_implicit_inputs(self, dash):
        (entry,) = [e for e in dash.processor.step_history if e["step"] == "PCA"]
        assert entry["params"]["batch_key"] == "batch"
        assert entry["params"]["counts_layer"] == "counts"

    def test_default_run_projects_and_enables_plot_button(self, dash):
        adata = dash.dataset.adata
        step = dash.processor.steps["PCA"]
        _, ok = run_step(dash, "PCA")
        assert ok
        X_pca, PCs = adata.obsm["X_pca"], adata.varm["PCs"]
        assert X_pca.shape == (adata.n_obs, 30)
        assert PCs.shape == (adata.n_vars, 30)
        # step stores the raw (uncentered) projection X @ PCs
        np.testing.assert_allclose(X_pca, adata.X.dot(PCs), atol=1e-6)
        assert adata.uns["pca"]["variance_ratio"].shape == (30,)
        assert (np.diff(adata.uns["pca"]["variance"]) <= 1e-9).all()
        assert step.plot_variance_ratio_button.disabled is False
        # PCs are orthonormal
        np.testing.assert_allclose(PCs.T @ PCs, np.eye(30), atol=1e-4)

    def test_zero_center_removes_column_means(self, dash):
        adata = dash.dataset.adata
        _, ok = run_step(dash, "PCA", n_comps=8, zero_center=True)
        assert ok
        X_pca = adata.obsm["X_pca"]
        assert X_pca.shape[1] == 8
        np.testing.assert_allclose(X_pca.mean(axis=0), 0, atol=1e-5)
        raw = adata.X.dot(adata.varm["PCs"])
        np.testing.assert_allclose(X_pca, raw - raw.mean(axis=0), atol=1e-4)

    def test_mask_var_zeroes_loadings_of_masked_genes(self, dash):
        adata = dash.dataset.adata
        hv = adata.var["highly_variable"].to_numpy()
        assert 0 < hv.sum() < adata.n_vars
        _, ok = run_step(dash, "PCA", n_comps=5, mask_var="highly_variable")
        assert ok
        PCs = adata.varm["PCs"]
        assert np.all(PCs[~hv] == 0)
        assert np.any(PCs[hv] != 0)

    def test_reference_batch_fits_on_reference_only(self, dash):
        adata = dash.dataset.adata
        _, ok = run_step(dash, "PCA", n_comps=6)
        assert ok
        full_pcs = adata.varm["PCs"].copy()

        _, ok = run_step(dash, "PCA", n_comps=6, reference_batch="b1")
        assert ok
        assert adata.uns["pca"]["reference_batch"] == "b1"
        PCs = adata.varm["PCs"]
        assert PCs.shape == full_pcs.shape
        # fitted on a subset -> different loadings; embedding is still X @ PCs
        assert not np.allclose(np.abs(PCs), np.abs(full_pcs))
        np.testing.assert_allclose(adata.obsm["X_pca"], adata.X.dot(PCs), atol=1e-6)

        # matches a direct reference-only PCA
        ref = adata[adata.obs["batch"] == "b1"].copy()
        sc.pp.pca(ref, n_comps=6, mask_var=None, svd_solver="arpack")
        sim = np.abs(np.sum(ref.varm["PCs"] * PCs, axis=0))
        np.testing.assert_allclose(sim, 1.0, atol=1e-3)

    def test_reference_batch_with_hvg_mask_unions_two_flavors(self, dash):
        adata = dash.dataset.adata
        _, ok = run_step(
            dash, "PCA", n_comps=5, mask_var="highly_variable", reference_batch="b2"
        )
        assert ok
        assert adata.uns["pca"]["reference_batch"] == "b2"
        assert adata.varm["PCs"].shape == (adata.n_vars, 5)
        # loadings are zero outside the (recomputed) HVG union
        assert (np.abs(adata.varm["PCs"]).sum(axis=1) == 0).any()

    def test_reference_batch_hvg_without_log1p_layer(self):
        d = SCLabDashboard(make_adata())
        for name in ["QC", "Preprocess"]:
            strict_run_step(d, name)
        adata = d.dataset.adata
        del adata.layers["counts_log1p"]

        strict_run_step(
            d, "PCA", n_comps=5, mask_var="highly_variable", reference_batch="b1"
        )
        assert adata.uns["pca"]["reference_batch"] == "b1"
        assert "counts_log1p" not in adata.layers

    @pytest.mark.parametrize("reference_batch", [None, "b1"])
    @pytest.mark.parametrize("mask_var", [None, "highly_variable"])
    def test_matches_library_pca(self, mask_var, reference_batch):
        d = SCLabDashboard(make_adata())
        for name in ["QC", "Preprocess"]:
            strict_run_step(d, name)
        adata = d.dataset.adata
        expected = adata.copy()
        library_pca(
            expected,
            n_comps=5,
            mask_var=mask_var,
            batch_key="batch",
            reference_batch=reference_batch,
            zero_center=True,
        )

        strict_run_step(
            d,
            "PCA",
            n_comps=5,
            mask_var=mask_var,
            reference_batch=reference_batch,
            zero_center=True,
        )

        np.testing.assert_allclose(adata.obsm["X_pca"], expected.obsm["X_pca"])
        np.testing.assert_allclose(adata.varm["PCs"], expected.varm["PCs"])
        assert adata.uns["pca"].get("reference_batch") == reference_batch

    def test_plot_variance_ratio_publishes_figure(self, dash):
        step = dash.processor.steps["PCA"]
        step.variable_controls["n_comps"].value = 10
        step.variable_controls["mask_var"].value = None
        step.variable_controls["reference_batch"].value = None
        _, ok = run_step(dash, "PCA")
        assert ok
        calls = capture(dash, "dplt_plot_figure_request")
        step.plot_variance_ratio_callback()
        assert len(calls) == 1
        fig = calls[0][1]["figure"]
        y = np.asarray(fig.data[0].y)
        np.testing.assert_allclose(y, dash.dataset.adata.uns["pca"]["variance_ratio"])
        assert list(fig.data[0].x) == list(range(1, 11))

    def test_plot_variance_ratio_after_programmatic_run(self, dash):
        step = dash.processor.steps["PCA"]
        step.variable_controls["n_comps"].value = 30
        _, ok = run_step(dash, "PCA", n_comps=7)
        assert ok
        calls = capture(dash, "dplt_plot_figure_request")
        step.plot_variance_ratio_callback()
        fig = calls[0][1]["figure"]
        assert list(fig.data[0].x) == list(range(1, 8))
        assert len(fig.data[0].y) == 7

    def test_refuses_raw_counts_in_X(self):
        d = SCLabDashboard(make_adata())
        step = d.processor.steps["PCA"]
        _, ok = run_step(d, "PCA")
        assert not ok
        assert "Run the Preprocess step first" in d.broker.exceptions_log[-1]
        assert "X_pca" not in d.dataset.adata.obsm
        assert step.plot_variance_ratio_button.disabled is True

    def test_var_dataframe_change_refreshes_mask_options(self, dash):
        step = dash.processor.steps["PCA"]
        adata = dash.dataset.adata
        adata.var["my_flag"] = adata.var["highly_variable"]
        step.dset_var_dataframe_change_callback()
        control = step.variable_controls["mask_var"]
        assert "my_flag" in control.options and "highly_variable" in control.options
        control.value = "my_flag"
        del adata.var["my_flag"]
        step.dset_var_dataframe_change_callback()
        assert "my_flag" not in control.options
        assert control.value is None

    def test_initial_controls(self, dash):
        controls = dash.processor.steps["PCA"].variable_controls
        assert controls["n_comps"].value == 30
        assert set(controls) == {
            "n_comps",
            "mask_var",
            "reference_batch",
            "zero_center",
        }
        assert set(controls["reference_batch"].options) >= {"b1", "b2"}


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shifted_embedding(dash):
    """A 5-d embedding with a strong, known batch offset on every axis."""
    adata = dash.dataset.adata
    rng = np.random.default_rng(0)
    is_b2 = (adata.obs["batch"] == "b2").to_numpy()
    emb = rng.normal(size=(adata.n_obs, 5)) + is_b2[:, None] * 3.0
    adata.obsm["X_emb"] = emb
    return emb, is_b2


def _batch_gap(Z, is_b2):
    return np.linalg.norm(Z[is_b2].mean(axis=0) - Z[~is_b2].mean(axis=0))


class TestIntegration:
    def test_harmony_reduces_batch_offset(self, dash, shifted_embedding):
        emb, is_b2 = shifted_embedding
        calls = capture(dash, "dset_data_dict_change")
        _, ok = run_step(
            dash,
            "Integration",
            use_rep="X_emb",
            group_by="batch",
            flavor="harmony",
            reference_batch=None,
            max_iters=10,
        )
        assert ok
        Z = dash.dataset.adata.obsm["X_emb_harmony"]
        assert Z.shape == emb.shape and np.isfinite(Z).all()
        assert _batch_gap(Z, is_b2) < 0.5 * _batch_gap(emb, is_b2)
        # input untouched
        np.testing.assert_array_equal(dash.dataset.adata.obsm["X_emb"], emb)
        assert calls and calls[-1][0][1] == "X_emb_harmony"

    def test_harmony_with_reference_batch(self, dash, shifted_embedding):
        emb, is_b2 = shifted_embedding
        _, ok = run_step(
            dash,
            "Integration",
            use_rep="X_emb",
            group_by="batch",
            flavor="harmony",
            reference_batch="b1",
            max_iters=10,
        )
        assert ok
        Z = dash.dataset.adata.obsm["X_emb_harmony"]
        assert _batch_gap(Z, is_b2) < _batch_gap(emb, is_b2)

    def test_cca_adds_embedding_with_one_row_per_cell(self, dash, shifted_embedding):
        emb, is_b2 = shifted_embedding
        _, ok = run_step(
            dash,
            "Integration",
            use_rep="X_emb",
            group_by="batch",
            flavor="cca",
            reference_batch=None,
            max_iters=5,
        )
        assert ok
        Z = dash.dataset.adata.obsm["X_emb_cca"]
        assert Z.shape[0] == emb.shape[0] and np.isfinite(Z).all()
        assert _batch_gap(Z, is_b2) < _batch_gap(emb, is_b2)

    def test_cca_with_reference_batch_fails(self, dash, shifted_embedding):
        before = set(dash.dataset.adata.obsm)
        n_exc = len(dash.broker.exceptions_log)
        step, ok = run_step(
            dash,
            "Integration",
            use_rep="X_emb",
            group_by="batch",
            flavor="cca",
            reference_batch="b1",
            max_iters=5,
        )
        assert not ok
        assert step.run_button.button_style == "danger"
        assert set(dash.dataset.adata.obsm) == before
        errors = dash.broker.exceptions_log[n_exc:]
        assert any("reference_batch is not supported" in e for e in errors)

    def test_unknown_flavor_fails_without_side_effects(self, dash, shifted_embedding):
        before = set(dash.dataset.adata.obsm)
        hist = len(dash.processor.step_history)
        step, ok = run_step(
            dash,
            "Integration",
            use_rep="X_emb",
            group_by="batch",
            flavor="bogus",
            reference_batch=None,
            max_iters=5,
        )
        assert not ok
        assert step.run_button.button_style == "danger"
        assert set(dash.dataset.adata.obsm) == before
        assert len(dash.processor.step_history) == hist + 1  # recorded as failed
        assert dash.processor.step_history[-1]["status"] == "failed"
        assert dash.processor.step_history[-1]["error"] == (
            "ValueError: Unknown flavor: bogus"
        )
        assert "Unknown flavor: bogus" in dash.broker.exceptions_log[-1]

    def test_missing_group_column_fails(self, dash, shifted_embedding):
        step, ok = run_step(
            dash,
            "Integration",
            use_rep="X_emb",
            group_by="no_such_column",
            flavor="harmony",
            reference_batch=None,
            max_iters=5,
        )
        assert not ok

    def test_reference_options_follow_group_by(self, dash):
        controls = dash.processor.steps["Integration"].variable_controls
        controls["group_by"].value = "grp"
        assert set(controls["reference_batch"].options) == {"", "A", "B", "C"} or set(
            controls["reference_batch"].options
        ) == {None, "A", "B", "C"}
        controls["group_by"].value = "batch"
        assert "b1" in controls["reference_batch"].options

    def test_reference_options_populated_for_default_group_by(self):
        d = SCLabDashboard(make_adata(60, 20))
        controls = d.processor.steps["Integration"].variable_controls
        assert controls["group_by"].value == "batch"
        assert "b1" in controls["reference_batch"].options

    def test_metadata_change_keeps_group_by(self):
        d = SCLabDashboard(make_adata(60, 20))
        controls = d.processor.steps["Integration"].variable_controls
        assert controls["group_by"].value == "batch"
        n = len(d.broker.exceptions_log)
        # a new categorical column changes the options offered to group* dropdowns
        d.dataset.adata.obs["new_cat"] = pd.Categorical(["u", "v"] * 30)
        d.broker.publish("dset_metadata_change", d.dataset.metadata)
        assert [e for e in d.broker.exceptions_log[n:] if "KeyError" in e] == []
        assert controls["group_by"].value == "batch"


# --------------------------------------------------------------------------
# Doublet detection
# --------------------------------------------------------------------------
class TestDoubletDetection:
    def test_scrublet_adds_score_and_label(self, dash):
        calls = capture(dash, "dset_metadata_change")
        _, ok = run_step(dash, "Doublet Detection", layer="counts", flavor="scrublet")
        assert ok
        obs = dash.dataset.adata.obs
        score, label = obs["scrublet_score"], obs["scrublet_label"]
        assert score.between(0, 1).all() and score.notna().all()
        assert list(label.cat.categories) == ["singlet", "doublet"]
        assert label.notna().all()
        doublets = label == "doublet"
        if doublets.any() and (~doublets).any():
            assert score[doublets].min() >= score[~doublets].max()
        assert any(len(args) > 1 and args[1] == "scrublet_label" for args, _ in calls)

    def test_scrublet_is_deterministic(self, dash):
        obs = dash.dataset.adata.obs
        first = obs["scrublet_score"].copy()
        _, ok = run_step(dash, "Doublet Detection", layer="counts", flavor="scrublet")
        assert ok
        np.testing.assert_allclose(obs["scrublet_score"], first)

    def test_unknown_flavor_fails(self, dash):
        step, ok = run_step(dash, "Doublet Detection", layer="counts", flavor="nope")
        assert not ok
        assert "Unknown flavor: nope" in dash.broker.exceptions_log[-1]

    def test_flavor_options_reflect_installed_packages(self, dash):
        from sclab.tools.doublet_detection import (
            get_available_doublet_detection_methods,
        )

        opts = dash.processor.steps["Doublet Detection"].variable_controls["flavor"]
        assert set(opts.options) == set(get_available_doublet_detection_methods())
        assert "scrublet" in opts.options

    def test_requires_at_least_one_backend(self, dash, monkeypatch):
        import sclab.tools.doublet_detection as dd
        from sclab.examples.processor_steps import DoubletDetection

        monkeypatch.setattr(dd, "any_doublet_detection_available", lambda: False)
        with pytest.raises(ImportError):
            DoubletDetection(dash.processor)

    def test_default_controls_use_counts_layer(self):
        d = SCLabDashboard(make_adata(300, 300))
        step = d.processor.steps["Doublet Detection"]
        assert step.variable_controls["override_layer"].value is False
        layer = step.variable_controls["layer"]
        assert layer.disabled is True
        assert layer.value == d.dataset.counts_layer
        assert step.variable_params["layer"] == "counts"

        step.variable_controls["flavor"].value = "scrublet"
        step.run()
        assert step.run_button.button_style == "success"
        assert "scrublet_score" in d.dataset.adata.obs

    def test_default_ignores_other_layers(self):
        d = SCLabDashboard(make_adata(300, 300))
        adata = d.dataset.adata
        adata.layers["other"] = adata.layers["counts"] * 0
        used = []
        step = d.processor.steps["Doublet Detection"]
        step.variable_controls["flavor"].value = "scrublet"

        import sclab.examples.processor_steps._doublet_detection as mod

        mod_scrublet = mod.scrublet
        mod.scrublet = lambda **kw: used.append(kw["layer"])
        try:
            step.run()
        finally:
            mod.scrublet = mod_scrublet
        assert used == ["counts"]
        # history records both variable controls
        params = d.processor.step_history[-1]["params"]
        assert params["override_layer"] is False
        assert params["layer"] == "counts"  # resolved implicit input

    def test_override_enables_dropdown_and_uses_chosen_layer(self):
        d = SCLabDashboard(make_adata(300, 300))
        adata = d.dataset.adata
        adata.layers["other"] = adata.layers["counts"].copy()
        d.broker.publish("dset_anndata_layers_change", adata.layers)
        step = d.processor.steps["Doublet Detection"]
        controls = step.variable_controls
        controls["flavor"].value = "scrublet"

        controls["override_layer"].value = True
        assert controls["layer"].disabled is False
        controls["layer"].value = "other"

        used = []
        import sclab.examples.processor_steps._doublet_detection as mod

        mod_scrublet = mod.scrublet
        mod.scrublet = lambda **kw: used.append(kw["layer"])
        try:
            step.run()
        finally:
            mod.scrublet = mod_scrublet
        assert used == ["other"]
        params = d.processor.step_history[-1]["params"]
        assert params["override_layer"] is True and params["layer"] == "other"

        controls["override_layer"].value = False
        assert controls["layer"].disabled is True
        assert controls["layer"].value == "counts"

    @pytest.mark.parametrize(
        "flavor, function, seed",
        [
            ("scDblFinder", "scdblfinder", 31415),
            ("scrublet", "scrublet", 0),
            ("doubletdetection", "doubletdetection", 0),
        ],
    )
    def test_uses_library_default_seed(self, flavor, function, seed, monkeypatch):
        import sclab.examples.processor_steps._doublet_detection as mod

        d = SCLabDashboard(make_adata(300, 300))
        step = d.processor.steps["Doublet Detection"]
        used = []
        monkeypatch.setattr(mod, function, lambda **kw: used.append(kw["random_state"]))
        step.run(flavor=flavor)  # flavor need not be installed: the call is faked
        assert step.run_button.button_style == "success"
        assert used == [seed]
        entry = d.processor.step_history[-1]
        assert entry["params"]["random_state"] == seed  # explicit in the history
        assert f"random_state={seed}" in step.to_code(entry["params"])

    def test_layers_change_refresh_keeps_counts_selected(self):
        d = SCLabDashboard(make_adata(300, 300))
        adata = d.dataset.adata
        adata.layers["extra"] = adata.layers["counts"].copy()
        d.broker.publish("dset_anndata_layers_change", adata.layers)
        layer = d.processor.steps["Doublet Detection"].variable_controls["layer"]
        assert "extra" in layer.options
        assert layer.value == "counts"
        assert layer.disabled is True

    def test_normalized_layer_override_fails(self, dash):
        _, ok = run_step(
            dash,
            "Doublet Detection",
            flavor="scrublet",
            override_layer=True,
            layer="counts_log1p",
        )
        assert not ok
        assert "does not look like raw counts" in dash.broker.exceptions_log[-1]


# --------------------------------------------------------------------------
# Gene expression
# --------------------------------------------------------------------------
@pytest.fixture
def gene_dash():
    adata = make_adata(80, 30)
    adata.var["gene_name"] = [f"G{i}" for i in range(adata.n_vars)]
    adata.var["description"] = [
        "nanos homolog" if i == 0 else ("" if i == 1 else f"desc {i}")
        for i in range(adata.n_vars)
    ]
    return SCLabDashboard(adata)


class TestGeneExpression:
    def test_gene_options_annotated_with_info_columns(self, gene_dash):
        opts = (
            gene_dash.processor.steps["Gene Expression"]
            .variable_controls["gene_input"]
            .options
        )
        assert len(opts) == 30
        assert "var_0003 - G2 - desc 2" in opts

    def test_description_text_containing_nan_is_preserved(self, gene_dash):
        opts = (
            gene_dash.processor.steps["Gene Expression"]
            .variable_controls["gene_input"]
            .options
        )
        assert any("nanos homolog" in o for o in opts)

    def test_missing_info_values_render_empty(self):
        from sclab.examples.processor_steps._gene_expression import GeneExpression

        adata = make_adata(10, 3)
        adata.var["gene_name"] = ["G0", np.nan, "nanos"]
        adata.var["symbol"] = pd.Categorical(["S0", None, "S2"])
        v = adata.var_names
        assert GeneExpression._gene_options(adata) == (
            f"{v[0]} - G0 - S0",
            f"{v[1]} -  - ",
            f"{v[2]} - nanos - S2",
        )

    def test_total_vars_change_refreshes_gene_options(self, gene_dash):
        dash = gene_dash
        control = dash.processor.steps["Gene Expression"].variable_controls[
            "gene_input"
        ]
        control.value = "var_0005 - G4 - desc 4"
        n = len(dash.broker.exceptions_log)

        # drop genes, keeping the selected one
        dash.dataset.adata._inplace_subset_var(
            ~dash.dataset.adata.var_names.isin(["var_0001", "var_0002"])
        )
        dash.broker.publish("dset_total_vars_change", dash.dataset.adata.var)
        assert dash.broker.exceptions_log[n:] == []
        assert isinstance(control.options, tuple) and len(control.options) == 28
        assert not any(
            o.startswith(("var_0001 ", "var_0002 ")) for o in control.options
        )
        assert "var_0003 - G2 - desc 2" in control.options
        assert control.value == "var_0005 - G4 - desc 4"

        # drop the selected gene: the value falls back to ""
        dash.dataset.adata._inplace_subset_var(
            ~dash.dataset.adata.var_names.isin(["var_0005"])
        )
        dash.broker.publish("dset_total_vars_change", dash.dataset.adata.var)
        assert dash.broker.exceptions_log[n:] == []
        assert len(control.options) == 27
        assert control.value == ""

    def test_nothing_published_until_gene_and_layer_chosen(self, gene_dash):
        calls = capture(gene_dash, "dplt_plot_figure_request")
        step = gene_dash.processor.steps["Gene Expression"]
        step.send_plot({})
        assert calls == []
        step.variable_controls["gene_input"].value = step.variable_controls[
            "gene_input"
        ].options[3]
        assert calls == []  # layer still unset

    def test_colors_scatter_by_expression(self, gene_dash, monkeypatch):
        calls = capture(gene_dash, "dplt_plot_figure_request")
        step = gene_dash.processor.steps["Gene Expression"]
        messages = []
        monkeypatch.setattr(
            step, "update_output", lambda m, *a, **k: messages.append(m)
        )
        c = step.variable_controls
        c["layer"].value = "counts"
        c["gene_input"].value = c["gene_input"].options[2]
        c["colorscale"].value = "Viridis"

        args, kwargs = calls[-1]
        assert kwargs["colorby"] == "var_0003"
        assert kwargs["color_continuous_scale"] == "Viridis"
        adata = gene_dash.dataset.adata
        expected = np.asarray(adata[:, "var_0003"].layers["counts"].todense()).ravel()
        meta = kwargs["metadata"]
        np.testing.assert_allclose(meta["var_0003"].to_numpy(), expected)
        assert meta.index.equals(adata.obs.index)
        assert messages[-1] == "Showing gene: var_0003"

    def test_selected_cells_do_not_restrict_the_metadata_sent(self, gene_dash):
        # the plotter highlights/filters the selection itself and needs every row
        calls = capture(gene_dash, "dplt_plot_figure_request")
        step = gene_dash.processor.steps["Gene Expression"]
        c = step.variable_controls
        c["layer"].value = "counts"
        gene_dash.dataset.selected_rows = gene_dash.dataset.metadata.index[:7]
        c["gene_input"].value = c["gene_input"].options[0]
        meta = calls[-1][1]["metadata"]
        assert meta.index.equals(gene_dash.dataset.adata.obs.index)
        assert meta["is_selected"].sum() == 7

    def test_time_axis_makes_scatter_vs_metadata(self, gene_dash):
        calls = capture(gene_dash, "dplt_plot_figure_request")
        step = gene_dash.processor.steps["Gene Expression"]
        c = step.variable_controls
        c["layer"].value = "counts"
        c["gene_input"].value = c["gene_input"].options[4]
        c["time_key"].value = "time"
        fig = calls[-1][1]["figure"]
        adata = gene_dash.dataset.adata
        np.testing.assert_allclose(
            np.asarray(fig.data[0].x), adata.obs["time"].to_numpy()
        )
        expected = np.asarray(adata[:, "var_0005"].layers["counts"].todense()).ravel()
        np.testing.assert_allclose(np.asarray(fig.data[0].y), expected)
        assert fig.layout.title.text == "Gene: var_0005, Layer: counts"

    def test_run_button_replots(self, gene_dash):
        calls = capture(gene_dash, "dplt_plot_figure_request")
        step = gene_dash.processor.steps["Gene Expression"]
        c = step.variable_controls
        c["layer"].value = "counts"
        c["gene_input"].value = c["gene_input"].options[2]
        n = len(calls)
        step.run()
        assert step.run_button.button_style == "success"
        assert len(calls) == n + 1

    @pytest.mark.parametrize("time_key", [None, "time"])
    def test_plot_with_active_selection(self, gene_dash, time_key):
        dash = gene_dash
        ctrl = dash.processor.steps["Gene Expression"].variable_controls
        dash.plotter.controls.data_key.value = "Y_array2"
        dash.dataset.selected_rows = dash.dataset.row_names[::2]
        n = len(dash.broker.exceptions_log)

        ctrl["layer"].value = "counts"
        ctrl["time_key"].value = time_key
        ctrl["gene_input"].value = "var_0005 - G4 - desc 4"

        assert dash.broker.exceptions_log[n:] == []
        g = dash.plotter.g
        ids = np.concatenate([np.asarray(t.hovertext) for t in g.data])
        assert set(ids) == set(dash.dataset.row_names)  # all cells, not a subset
        sizes = np.concatenate([np.asarray(t.marker.size) for t in g.data])
        assert set(sizes) == {0.5, 3.0}  # selection highlighted by the plotter


# --------------------------------------------------------------------------
# Differential expression
# --------------------------------------------------------------------------
class TestDifferentialExpressionStep:
    def test_all_groups_vs_rest(self, dash):
        adata = dash.dataset.adata
        step, ok = run_step(
            dash,
            "Differential Expression",
            groupby="grp",
            groups="all",
            reference="rest",
            layer=None,
            name="",
        )
        assert ok
        res = adata.uns["differential_expression"]
        assert list(res["names"].dtype.names) == ["A", "B", "C"]
        assert (
            res["params"]["groupby"] == "grp" and res["params"]["reference"] == "rest"
        )
        assert "pts" in res and "pts_rest" in res
        # every group's top gene is up in that group relative to the rest
        X = adata.X.toarray()
        for g in ["A", "B", "C"]:
            gene = res["names"][g][0]
            j = adata.var_names.get_loc(gene)
            inside = (adata.obs["grp"] == g).to_numpy()
            assert res["scores"][g][0] > 0
            assert X[inside, j].mean() > X[~inside, j].mean()
        # results panel follows the new analysis
        assert step.results.result_selector.value == "differential_expression"
        assert step.results.group_selector.options == ("all", "A", "B", "C")

    def test_unique_keys_for_repeated_and_named_runs(self, dash):
        uns = dash.dataset.adata.uns
        kw = dict(groupby="grp", groups="all", reference="rest", layer=None)
        _, ok = run_step(dash, "Differential Expression", name="", **kw)
        assert ok and "differential_expression_2" in uns
        _, ok = run_step(dash, "Differential Expression", name="x", **kw)
        assert ok and "differential_expression_x" in uns
        _, ok = run_step(dash, "Differential Expression", name="x", **kw)
        assert ok and "differential_expression_x_2" in uns
        step = dash.processor.steps["Differential Expression"]
        assert step.results.result_selector.value == "differential_expression_x_2"
        assert set(step.results.result_selector.options) >= {
            "differential_expression",
            "differential_expression_2",
            "differential_expression_x",
            "differential_expression_x_2",
        }

    def test_specific_groups_against_reference_group(self, dash):
        adata = dash.dataset.adata
        _, ok = run_step(
            dash,
            "Differential Expression",
            groupby="grp",
            groups=("B", "C"),
            reference="A",
            layer="counts_log1p",
            name="ref",
        )
        assert ok
        res = adata.uns["differential_expression_ref"]
        assert list(res["names"].dtype.names) == ["B", "C"]
        assert res["params"]["reference"] == "A"
        assert res["params"]["layer"] == "counts_log1p"

    def test_group_selector_renders_table(self, dash, monkeypatch):
        import itables

        shown = []
        monkeypatch.setattr(itables, "show", lambda df, *a, **k: shown.append(df))
        step = dash.processor.steps["Differential Expression"]
        results = step.results
        results.result_selector.value = "differential_expression_ref"
        assert results.group_selector.options == ("all", "B", "C")
        n = len(shown)
        results.group_selector.value = "B"
        assert len(shown) == n + 1
        res = dash.dataset.adata.uns["differential_expression_ref"]
        assert shown[-1]["names"].tolist() == [str(g) for g in res["names"]["B"]]
        assert "group" not in shown[-1]  # single group
        results.group_selector.value = "all"
        assert set(shown[-1]["group"].astype(str)) == {"B", "C"}
        results.group_selector.value = "all"
        results.sync_results_list()
        assert results.result_selector.value == "differential_expression_ref"
        results.sync_results_list(focus_result="differential_expression")
        assert results.result_selector.value == "differential_expression"

    def test_missing_groupby_fails(self, dash):
        step, ok = run_step(
            dash,
            "Differential Expression",
            groupby=None,
            groups="all",
            reference="rest",
            layer=None,
            name="bad",
        )
        assert not ok
        assert "differential_expression_bad" not in dash.dataset.adata.uns

    def test_empty_groups_means_all(self, dash):
        _, ok = run_step(
            dash,
            "Differential Expression",
            groupby="batch",
            groups=(),
            reference="rest",
            layer=None,
            name="batchde",
        )
        assert ok
        res = dash.dataset.adata.uns["differential_expression_batchde"]
        assert list(res["names"].dtype.names) == ["b1", "b2"]

    def test_group_controls_follow_groupby(self, dash):
        c = dash.processor.steps["Differential Expression"].variable_controls
        c["groupby"].value = "grp"
        assert c["groups"].options == ("A", "B", "C")
        assert c["reference"].options == ("rest", "A", "B", "C")
        assert c["reference"].value == "rest"
        c["reference"].value = "B"
        c["groupby"].value = "batch"  # B not valid for batch -> falls back to rest
        assert c["reference"].options == ("rest", "b1", "b2")
        assert c["reference"].value == "rest"
        c["groupby"].value = None
        assert c["groups"].options == ("",)
        assert c["reference"].value == ""

    def test_var_dataframe_change_event_is_handled(self):
        d = SCLabDashboard(make_adata(60, 20))
        n = len(d.broker.exceptions_log)
        d.broker.publish("dset_var_dataframe_change", d.dataset.adata.var)
        assert d.broker.exceptions_log[n:] == []


class TestRankGenesGroupsDf:
    @pytest.fixture(scope="class")
    def de_adata(self):
        rng = np.random.default_rng(0)
        X = rng.poisson(2.0, size=(90, 12)).astype(float)
        grp = np.repeat(["A", "B", "C"], 30)
        X[grp == "A", 0] += 6  # gene g0 up in A
        X[grp == "B", 1] += 6  # gene g1 up in B
        adata = AnnData(
            X,
            obs=pd.DataFrame(
                {"g": pd.Categorical(grp)}, index=[f"c{i}" for i in range(90)]
            ),
            var=pd.DataFrame(
                {"symbol": [f"S{i}" for i in range(12)]},
                index=[f"g{i}" for i in range(12)],
            ),
        )
        sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(
            adata, "g", method="t-test", pts=True, key_added="ttest"
        )
        sc.tl.rank_genes_groups(adata, "g", method="logreg", key_added="logreg")
        return adata

    def test_all_groups_long_format(self, de_adata):
        df = _rank_genes_groups_df(de_adata, None, key="ttest")
        assert len(df) == 3 * 12
        assert list(df.columns) == [
            "group",
            "names",
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
            "pct_nz_group",
            "pct_nz_reference",
        ]
        assert df["group"].astype(str).unique().tolist() == ["A", "B", "C"]
        # within each group rows keep scanpy's rank order (scores descending)
        for _, sub in df.groupby("group", observed=True):
            assert sub["scores"].is_monotonic_decreasing
        top = df.groupby("group", observed=True).head(1).set_index("group")["names"]
        assert top["A"] == "g0" and top["B"] == "g1"

    def test_single_group_drops_group_column(self, de_adata):
        df = _rank_genes_groups_df(de_adata, "A", key="ttest")
        assert "group" not in df.columns and len(df) == 12
        assert df.iloc[0]["names"] == "g0"
        assert df.iloc[0]["pct_nz_group"] > df.iloc[0]["pct_nz_reference"]

    def test_filters(self, de_adata):
        full = _rank_genes_groups_df(de_adata, "A", key="ttest")
        sig = _rank_genes_groups_df(de_adata, "A", key="ttest", pval_cutoff=0.01)
        assert 0 < len(sig) < len(full) and (sig["pvals_adj"] < 0.01).all()
        up = _rank_genes_groups_df(de_adata, "A", key="ttest", log2fc_min=0.5)
        assert (up["logfoldchanges"] > 0.5).all() and "g0" in set(up["names"])
        down = _rank_genes_groups_df(de_adata, "A", key="ttest", log2fc_max=0)
        assert (down["logfoldchanges"] < 0).all()
        both = _rank_genes_groups_df(
            de_adata, "A", key="ttest", log2fc_min=-1, log2fc_max=1
        )
        assert both["logfoldchanges"].between(-1, 1, inclusive="neither").all()
        # index is reset after filtering
        assert up.index.tolist() == list(range(len(up)))

    def test_gene_symbols_and_multiple_groups(self, de_adata):
        df = _rank_genes_groups_df(
            de_adata, ["A", "C"], key="ttest", gene_symbols="symbol"
        )
        assert set(df["group"].astype(str)) == {"A", "C"}
        assert (df["symbol"] == "S" + df["names"].str[1:]).all()

    def test_logreg_has_only_names_and_scores(self, de_adata):
        df = _rank_genes_groups_df(de_adata, "B", key="logreg", pval_cutoff=0.1)
        assert list(df.columns) == ["names", "scores"]  # filters are ignored
        assert len(df) == 12
