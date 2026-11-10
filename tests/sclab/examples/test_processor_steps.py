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
from sclab.examples.processor_steps._differential_expression import _sheet_name
from sclab.tools.differential_expression import edgepython_is_available

from ..test_background_jobs import FakeLoop
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
        assert step.run_button.disabled
        assert "Run the Preprocess step first" in step.status.value
        with pytest.raises(ValueError, match="Run the Preprocess step first"):
            step.run()
        assert d.processor.step_history == []
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

    def test_cca_with_reference_batch_is_refused(self, dash, shifted_embedding):
        before = set(dash.dataset.adata.obsm)
        hist = len(dash.processor.step_history)
        with pytest.raises(ValueError, match="CCA does not take a reference batch"):
            dash.processor.steps["Integration"].run(
                use_rep="X_emb",
                group_by="batch",
                flavor="cca",
                reference_batch="b1",
                max_iters=5,
            )
        assert set(dash.dataset.adata.obsm) == before
        assert len(dash.processor.step_history) == hist  # nothing recorded

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

    def test_missing_group_column_is_refused(self, dash, shifted_embedding):
        with pytest.raises(ValueError, match="No column 'no_such_column'"):
            dash.processor.steps["Integration"].run(
                use_rep="X_emb",
                group_by="no_such_column",
                flavor="harmony",
                reference_batch=None,
                max_iters=5,
            )

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
        import sclab.examples.processor_steps._doublet_detection as mod
        from sclab.examples.processor_steps import DoubletDetection

        for name in [
            "doubletdetection_is_available",
            "scrublet_is_available",
            "_r_available",
            "_pyscdblfinder_available",
        ]:
            monkeypatch.setattr(mod, name, lambda: False)
        with pytest.raises(ImportError):
            DoubletDetection(dash.processor)

    @staticmethod
    def _scdblfinder_dash(monkeypatch, r: bool, python: bool) -> SCLabDashboard:
        import sclab.examples.processor_steps._doublet_detection as mod

        monkeypatch.setattr(mod, "_r_available", lambda: r)
        monkeypatch.setattr(mod, "_pyscdblfinder_available", lambda: python)
        return SCLabDashboard(make_adata(300, 300))

    @pytest.mark.parametrize("r, python", [(True, False), (False, True)])
    def test_backend_hidden_with_one_backend(self, monkeypatch, r, python):
        d = self._scdblfinder_dash(monkeypatch, r, python)
        step = d.processor.steps["Doublet Detection"]
        c = step.variable_controls
        c["flavor"].value = "scDblFinder"
        assert c["backend"].layout.display == "none"
        backend = "r" if r else "python"
        assert c["backend"].value == backend
        caption = step.tool_label.value
        assert ("R · scDblFinder::scDblFinder" in caption) == r
        assert ("Python · pyscdblfinder" in caption) == python

    def test_backend_dropdown_with_both_backends(self, monkeypatch):
        import threading

        import sclab.examples.processor_steps._doublet_detection as mod

        d = self._scdblfinder_dash(monkeypatch, True, True)
        step = d.processor.steps["Doublet Detection"]
        c = step.variable_controls
        c["flavor"].value = "scrublet"
        assert c["backend"].layout.display == "none"  # scDblFinder only
        assert "scrublet.Scrublet" in step.tool_label.value
        c["flavor"].value = "scDblFinder"
        assert c["backend"].layout.display is None
        assert c["backend"].value == "r"  # what backend="auto" picks
        assert "R · scDblFinder::scDblFinder" in step.tool_label.value
        c["backend"].value = "python"
        assert "Python · pyscdblfinder" in step.tool_label.value

        calls = []

        def fake_scdblfinder(adata, key_added, backend, **_):
            calls.append((backend, threading.current_thread()))
            adata.obs[f"{key_added}_label"] = pd.Categorical(["singlet"] * adata.n_obs)

        monkeypatch.setattr(mod, "scdblfinder", fake_scdblfinder)
        d.broker.jobs.loop = FakeLoop()
        d.broker.jobs.beat_timeout = 0  # no frontend: resume on the timer
        step.run_button.click()
        assert d.broker.jobs.busy  # Python: a background job
        d.broker.jobs.loop.pump(d.broker.jobs)
        assert step.run_button.button_style == "success", step.status.value
        assert calls[-1][0] == "python"
        assert calls[-1][1] is not threading.main_thread()
        entry = d.processor.step_history[-1]
        assert entry["params"]["backend"] == "python"
        assert 'backend="python"' in step.to_code(entry["params"])

        c["backend"].value = "r"
        step.run_button.click()
        assert not d.broker.jobs.busy  # R: synchronous, on the main thread
        assert calls[-1] == ("r", threading.main_thread())
        entry = d.processor.step_history[-1]
        assert entry["status"] == "success" and entry["params"]["backend"] == "r"
        assert "backend=" not in step.to_code(entry["params"])  # the default

        c["flavor"].value = "scrublet"  # no backend recorded for other flavors
        assert "backend" not in step.resolved_params(step._control_params())

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
        assert d.processor.step_history[-1]["status"] == "success"
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
        p = dash.plotter
        ids = p._rows[np.concatenate(p._trace_rows)]
        assert set(ids) == set(dash.dataset.row_names)  # all cells, not a subset
        assert all(np.asarray(t.customdata).size for t in g.data)  # hover: cell ids
        sizes = np.concatenate([np.asarray(t.marker.size) for t in g.data])
        assert set(sizes) == {0.5, 3.0}  # selection highlighted by the plotter


# --------------------------------------------------------------------------
# Differential expression
# --------------------------------------------------------------------------
DE = "Differential Expression"
needs_edgepython = pytest.mark.skipif(
    not edgepython_is_available(), reason="edgepython not installed"
)


def de_params(**params):
    """Every DE control: run(**params) does not depend on earlier tests' values."""
    defaults = dict(
        method="wilcoxon",
        groupby="grp",
        groups=(),
        reference="rest",
        cell_identity=None,
        sample_key=None,
        replicas_per_group=3,
        override_layer=False,
        layer=None,
        name="",
    )
    return {**defaults, **params}


class TestNativeBackendCaptions:
    @pytest.mark.parametrize("backend", ["scanpy", "native"])
    def test_qc_preprocess_pca_say_which_backend_runs(self, monkeypatch, backend):
        from sclab.examples.processor_steps import _pca, _preprocess, _qc

        for mod in (_qc, _preprocess, _pca):
            monkeypatch.setattr(mod, "resolve_backend", lambda _: backend)
        d = SCLabDashboard(make_adata(120, 60))
        for name in ("QC", "Preprocess", "PCA"):
            step = d.processor.steps[name]
            caption = step.tool({})
            assert ("sclab (port of scanpy.pp" in caption) == (backend == "native")


class TestDifferentialExpressionStep:
    def test_tool_caption_follows_method(self):
        d = SCLabDashboard(make_adata(120, 60))
        step = d.processor.steps[DE]
        method = step.variable_controls["method"]
        label = step.tool_label
        children = list(step.controls.children)
        assert children.index(label) == children.index(step.run_button) + 1
        assert label.layout.display is None
        assert "Python · scanpy.tl.rank_genes_groups (wilcoxon)" in label.value
        assert "color:#888" in label.value  # muted
        method.value = "pseudobulk_limma"
        assert "Python · pylimma (limma-voom)" in label.value
        method.value = "pseudobulk_edger"
        assert "Python · edgepython" in label.value

    def test_all_groups_vs_rest(self, dash):
        adata = dash.dataset.adata
        step, ok = run_step(dash, DE, **de_params())
        assert ok
        res = adata.uns["differential_expression"]
        assert set(res) == {"params", "table"}
        table, params = res["table"], res["params"]
        assert list(table["contrast"].unique()) == [
            "A_vs_rest",
            "B_vs_rest",
            "C_vs_rest",
        ]
        assert params["method"] == "wilcoxon" and params["layer"] == "X"
        assert params["reference_group"] == "rest"
        # every group's top gene is up in that group relative to the rest
        X = adata.X.toarray()
        for g in ["A", "B", "C"]:
            top = table[table["contrast"] == f"{g}_vs_rest"].iloc[0]
            j = adata.var_names.get_loc(top["gene"])
            inside = (adata.obs["grp"] == g).to_numpy()
            assert top["score"] > 0
            assert X[inside, j].mean() > X[~inside, j].mean()
        recorded = dash.processor.step_history[-1]["params"]
        assert recorded["layer"] == "X"
        assert recorded["key_added"] == "differential_expression"
        # results panel follows the new analysis
        results = step.results
        assert results.result_selector.value == "differential_expression"
        assert results.contrast_selector.options == (
            "all",
            "A_vs_rest",
            "B_vs_rest",
            "C_vs_rest",
        )
        assert results.identity_selector.layout.display == "none"

    def test_unique_keys_for_repeated_and_named_runs(self, dash):
        uns = dash.dataset.adata.uns
        _, ok = run_step(dash, DE, **de_params())
        assert ok and "differential_expression_2" in uns
        _, ok = run_step(dash, DE, **de_params(name="x"))
        assert ok and "differential_expression_x" in uns
        _, ok = run_step(dash, DE, **de_params(name="x"))
        assert ok and "differential_expression_x_2" in uns
        step = dash.processor.steps[DE]
        assert step.results.result_selector.value == "differential_expression_x_2"
        assert set(step.results.result_selector.options) >= {
            "differential_expression",
            "differential_expression_2",
            "differential_expression_x",
            "differential_expression_x_2",
        }

    def test_specific_groups_against_reference_group(self, dash):
        params = de_params(
            groups=("B", "C"),
            reference="A",
            override_layer=True,
            layer="counts_log1p",
            name="ref",
        )
        _, ok = run_step(dash, DE, **params)
        assert ok
        res = dash.dataset.adata.uns["differential_expression_ref"]
        assert list(res["table"]["contrast"].unique()) == ["B_vs_A", "C_vs_A"]
        assert res["params"]["reference_group"] == "A"
        assert res["params"]["layer"] == "counts_log1p"

    def test_method_switch_sets_the_layer(self, dash):
        step = dash.processor.steps[DE]
        c = step.variable_controls
        c["override_layer"].value = False
        c["method"].value = "pseudobulk_edger"
        assert step.resolved_params(step._control_params())["layer"] == "counts"
        assert c["layer"].value == "counts"
        assert "(counts)" in c["override_layer"].description
        c["method"].value = "t-test"
        assert step.resolved_params(step._control_params())["layer"] == "X"
        assert c["layer"].value is None
        assert "(X)" in c["override_layer"].description
        c["method"].value = "wilcoxon"

    def test_override_checkbox_reveals_the_layer(self, dash):
        step = dash.processor.steps[DE]
        c = step.variable_controls
        assert c["layer"].layout.display == "none"
        c["override_layer"].value = True
        assert c["layer"].layout.display is None
        c["layer"].value = "counts_log1p"
        assert step.resolved_params(step._control_params())["layer"] == "counts_log1p"
        c["method"].value = "pseudobulk_edger"  # the override is kept
        assert c["layer"].value == "counts_log1p"
        c["override_layer"].value = False
        assert c["layer"].layout.display == "none"
        assert c["layer"].value == "counts"
        c["method"].value = "wilcoxon"

    def test_pseudobulk_params_only_for_pseudobulk(self, dash):
        c = dash.processor.steps[DE].variable_controls
        hidden = {
            k
            for k in ("sample_key", "replicas_per_group")
            if c[k].layout.display == "none"
        }
        assert hidden == {"sample_key", "replicas_per_group"}  # wilcoxon
        c["method"].value = "pseudobulk_limma"
        assert c["sample_key"].layout.display is None
        assert c["replicas_per_group"].layout.display is None
        c["sample_key"].value = "batch"  # real samples: no pseudo-replicates
        assert c["replicas_per_group"].layout.display == "none"
        c["sample_key"].value = None
        assert c["replicas_per_group"].layout.display is None
        c["method"].value = "wilcoxon"

    def test_codegen_emits_sclab_tools_de(self, dash):
        code = dash.generate_code()
        assert "sclab.tools.de(" in code and "import sclab.tools" in code
        assert "rank_genes_groups" not in code
        assert 'key_added="differential_expression_ref"' in code
        assert 'condition_group=["B", "C"]' in code
        assert 'reference_group="A"' in code
        assert 'layer="counts_log1p"' in code

    def test_metadata_change_refreshes_identity_and_sample_keys(self, dash):
        c = dash.processor.steps[DE].variable_controls
        c["cell_identity"].value = "batch"
        dash.dataset.adata.obs["donor"] = pd.Categorical(["d1", "d2"] * 200)
        dash.broker.publish("dset_metadata_change", dash.dataset.metadata)
        assert "donor" in c["cell_identity"].options
        assert "donor" in c["sample_key"].options
        assert c["cell_identity"].value == "batch"
        c["cell_identity"].value = None

    def test_missing_groupby_is_refused(self, dash):
        with pytest.raises(ValueError, match="Choose a 'Group by' column"):
            dash.processor.steps[DE].run(**de_params(groupby=None, name="bad"))
        assert "differential_expression_bad" not in dash.dataset.adata.uns

    def test_empty_groups_means_all(self, dash):
        _, ok = run_step(dash, DE, **de_params(groupby="batch", name="batchde"))
        assert ok
        res = dash.dataset.adata.uns["differential_expression_batchde"]
        assert list(res["table"]["contrast"].unique()) == ["b1_vs_rest", "b2_vs_rest"]

    def test_group_controls_follow_groupby(self, dash):
        c = dash.processor.steps[DE].variable_controls
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


@needs_edgepython
def test_de_step_follows_the_seed_setting():
    d = SCLabDashboard(make_adata(240, 120))
    for name in ["QC", "Preprocess"]:
        strict_run_step(d, name)
    d.seed_input.value = "5"
    tables = []
    for name in ("first", "second"):
        strict_run_step(d, DE, **de_params(method="pseudobulk_edger", name=name))
        assert d.processor.step_history[-1]["params"]["random_state"] == 5
        uns = d.dataset.adata.uns[f"differential_expression_{name}"]
        assert uns["params"]["random_state"] == 5
        tables.append(uns["table"])
    pd.testing.assert_frame_equal(*tables)
    assert "random_state=5" in d.generate_code()


@needs_edgepython
def test_pseudobulk_background_run_stores_params_and_table():
    d = SCLabDashboard(make_adata(240, 120))
    for name in ["QC", "Preprocess"]:
        strict_run_step(d, name)
    d.broker.jobs.loop = FakeLoop()
    d.broker.jobs.beat_timeout = 0  # no frontend: resume on the timer
    step = d.processor.steps[DE]
    c = step.variable_controls
    c["method"].value = "pseudobulk_edger"
    c["groupby"].value = "grp"
    c["reference"].value = "A"
    assert not step.run_button.disabled, step.status.value
    step.run_button.click()
    assert d.broker.jobs.busy  # a background job
    d.broker.jobs.loop.pump(d.broker.jobs)
    assert step.run_button.button_style == "success", step.status.value
    res = d.dataset.adata.uns["differential_expression"]
    assert set(res) == {"params", "table"}
    assert res["params"]["method"] == "pseudobulk_edger"
    assert res["params"]["layer"] == "counts"
    assert set(res["table"]["contrast"]) <= {"B_vs_A", "C_vs_A"}
    assert {"gene", "logFC", "pval", "padj", "F"} <= set(res["table"])
    entry = d.processor.step_history[-1]
    assert entry["status"] == "success" and entry["params"]["layer"] == "counts"
    code = d.generate_code()
    assert "sclab.tools.de(" in code
    assert 'method="pseudobulk_edger"' in code
    assert "layer=counts_layer" in code
    assert "sample_key=None" in code and "replicas_per_group=3" in code
    assert step.results.result_selector.value == "differential_expression"


class TestDifferentialExpressionResults:
    @pytest.fixture(scope="class")
    def results(self, dash):
        params = de_params(cell_identity="batch", name="viewer")
        step, ok = run_step(dash, DE, **params)
        assert ok
        return step.results

    def test_identity_picker_only_with_a_cell_identity(self, dash, results):
        results.result_selector.value = "differential_expression_viewer"
        assert results.identity_selector.layout.display is None
        assert results.identity_selector.options == ("all", "b1", "b2")
        results.result_selector.value = "differential_expression"
        assert results.identity_selector.layout.display == "none"
        assert results.identity_selector.options == ("all",)

    def test_pickers_filter_the_table(self, dash, results):
        results.result_selector.value = "differential_expression_viewer"
        full = dash.dataset.adata.uns["differential_expression_viewer"]["table"]
        df = results.filtered_table()
        assert df.columns[0] == "gene" and len(df) == len(full)
        results.contrast_selector.value = "B_vs_rest"
        df = results.filtered_table()
        assert "contrast" not in df and set(df["cell_identity"]) == {"b1", "b2"}
        assert len(df) == (full["contrast"] == "B_vs_rest").sum()
        results.identity_selector.value = "b2"
        df = results.filtered_table()
        assert "cell_identity" not in df
        expected = full[
            (full["contrast"] == "B_vs_rest") & (full["cell_identity"] == "b2")
        ]
        assert df["gene"].tolist() == expected["gene"].tolist()
        results.contrast_selector.value = "all"
        assert set(results.filtered_table()["contrast"]) == set(full["contrast"])
        results.identity_selector.value = "all"

    def test_table_bundles_and_logreg_note(self, dash, results):
        results.result_selector.value = "differential_expression_viewer"
        bundles = results._table()
        text = "".join(str(v) for b in bundles for v in b.values())
        assert "<details><summary>Parameters</summary>" in text
        assert "method: wilcoxon" in text
        assert "no p-values" not in text
        _, ok = run_step(dash, DE, **de_params(method="logreg", name="lr"))
        assert ok
        assert results.result_selector.value == "differential_expression_lr"
        table = dash.dataset.adata.uns["differential_expression_lr"]["table"]
        assert table["pval"].isna().all()
        text = "".join(str(v) for b in results._table() for v in b.values())
        assert "logreg gives no p-values" in text

    def test_excel_export(self, dash, results, monkeypatch):
        import io

        saved = []
        monkeypatch.setattr(
            results.download,
            "save",
            lambda data, name, type: saved.append((data, name)),
        )
        results.result_selector.value = "differential_expression_viewer"
        results.export_button.click()
        ((data, name),) = saved
        assert name == "differential_expression_viewer.xlsx"
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        assert list(sheets) == ["A_vs_rest", "B_vs_rest", "C_vs_rest", "params"]
        a = sheets["A_vs_rest"]
        assert a.columns[0] == "gene" and "cell_identity" in a
        table = dash.dataset.adata.uns["differential_expression_viewer"]["table"]
        assert len(a) == (table["contrast"] == "A_vs_rest").sum()
        params = sheets["params"].set_index("param")["value"]
        assert params["method"] == "wilcoxon" and params["group_key"] == "grp"

    def test_sheet_names_are_valid_and_unique(self):
        taken = {"params"}
        names = [_sheet_name(n, taken) for n in ["a/b:c", "x" * 40, "x" * 40, "PARAMS"]]
        assert names[0] == "a_b_c"
        assert all(len(n) <= 31 for n in names)
        assert len({n.lower() for n in names}) == 4 and names[3] != "PARAMS"

    def test_nothing_built_while_the_results_tab_is_hidden(
        self, dash, results, monkeypatch
    ):
        import itables

        shown = []
        monkeypatch.setattr(
            itables, "to_html_datatable", lambda df, *a, **k: shown.append(df) or ""
        )
        dash.main_content.selected_index = 0
        results.result_selector.value = "differential_expression_viewer"
        results.contrast_selector.value = "A_vs_rest"
        assert shown == []  # Results tab hidden: nothing built
        panel = dash.processor.results_panel
        panel.available_results.value = panel.available_results.options[
            results.namespace
        ]
        dash.main_content.selected_index = 1  # open the Results tab
        assert len(shown) == 1 and results.table_output.outputs
        assert shown[-1].columns[0] == "gene" and "contrast" not in shown[-1]
        results.contrast_selector.value = "B_vs_rest"
        assert len(shown) == 2
        results.result_selector.value = "differential_expression"  # one render
        assert len(shown) == 3
        dash.main_content.selected_index = 0  # hidden: unloaded, not rebuilt
        assert results.table_output.outputs == ()
        results.contrast_selector.value = "C_vs_rest"
        assert len(shown) == 3
        dash.main_content.selected_index = 1
        assert len(shown) == 4  # the change made while hidden is shown
        assert set(shown[-1]["gene"]) == set(
            dash.dataset.adata.uns["differential_expression"]["table"].query(
                "contrast == 'C_vs_rest'"
            )["gene"]
        )
        dash.main_content.selected_index = 0
