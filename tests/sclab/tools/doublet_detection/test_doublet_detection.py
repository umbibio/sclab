import numpy as np
import pandas as pd
import pytest

import sclab.tools.doublet_detection as dd_pkg
from sclab.tools.doublet_detection import (
    any_doublet_detection_available,
    doubletdetection,
    get_available_doublet_detection_methods,
    scdblfinder,
    scrublet,
)
from sclab.tools.doublet_detection import _doubletdetection as dd_mod
from sclab.tools.doublet_detection import _scdblfinder as scdbl_mod
from sclab.tools.doublet_detection import _scrublet as scrub_mod
from sclab.tools.doublet_detection._doubletdetection import (
    doubletdetection_is_available,
)
from sclab.tools.doublet_detection._scdblfinder import scdblfinder_is_available
from sclab.tools.doublet_detection._scrublet import scrublet_is_available

needs_scrublet = pytest.mark.skipif(
    not scrublet_is_available(), reason="scrublet not installed"
)
needs_doubletdetection = pytest.mark.skipif(
    not doubletdetection_is_available(), reason="doubletdetection not installed"
)
needs_scdblfinder = pytest.mark.skipif(
    not scdblfinder_is_available(),
    reason="requires R with scater/scDblFinder/BiocParallel/SingleCellExperiment",
)

QUIET = dict(
    synthetic_doublet_umi_subsampling=1.0,
    use_approx_neighbors=True,
    distance_metric="euclidean",
    get_doublet_neighbor_parents=False,
    min_counts=3,
    min_cells=3,
    min_gene_variability_pctl=85,
    log_transform=False,
    mean_center=True,
    normalize_variance=True,
    n_prin_comps=30,
    svd_solver="arpack",
    verbose=False,
)


class TestAvailability:
    def test_availability_functions_return_bool(self):
        for fn in (
            scrublet_is_available,
            doubletdetection_is_available,
            scdblfinder_is_available,
            any_doublet_detection_available,
        ):
            assert isinstance(fn(), bool)

    def test_method_list_reflects_availability(self, monkeypatch):
        monkeypatch.setattr(dd_pkg, "doubletdetection_is_available", lambda: True)
        monkeypatch.setattr(dd_pkg, "scdblfinder_is_available", lambda: False)
        monkeypatch.setattr(dd_pkg, "scrublet_is_available", lambda: True)

        assert get_available_doublet_detection_methods() == [
            "doubletdetection",
            "scrublet",
        ]
        assert any_doublet_detection_available() is True

    def test_method_list_all_and_none(self, monkeypatch):
        for name in (
            "doubletdetection_is_available",
            "scdblfinder_is_available",
            "scrublet_is_available",
        ):
            monkeypatch.setattr(dd_pkg, name, lambda: True)
        assert get_available_doublet_detection_methods() == [
            "doubletdetection",
            "scDblFinder",
            "scrublet",
        ]

        for name in (
            "doubletdetection_is_available",
            "scdblfinder_is_available",
            "scrublet_is_available",
        ):
            monkeypatch.setattr(dd_pkg, name, lambda: False)
        assert get_available_doublet_detection_methods() == []
        assert any_doublet_detection_available() is False

    def test_find_spec_based_checks(self, monkeypatch):
        monkeypatch.setattr(scrub_mod, "find_spec", lambda name: None)
        monkeypatch.setattr(dd_mod, "find_spec", lambda name: None)
        monkeypatch.setattr(scdbl_mod, "find_spec", lambda name: None)

        assert scrub_mod.scrublet_is_available() is False
        assert dd_mod.doubletdetection_is_available() is False
        assert scdbl_mod.scdblfinder_is_available() is False

    def test_scdblfinder_needs_both_python_and_r(self, monkeypatch):
        monkeypatch.setattr(scdbl_mod, "_python_deps_available", lambda: False)
        monkeypatch.setattr(scdbl_mod, "_r_deps_available", lambda: True)
        assert scdbl_mod.scdblfinder_is_available() is False

        monkeypatch.setattr(scdbl_mod, "_python_deps_available", lambda: True)
        assert scdbl_mod.scdblfinder_is_available() is True


@needs_scrublet
class TestScrublet:
    @pytest.fixture(scope="class")
    def result(self, doublet_data):
        adata = doublet_data[0].copy()
        scrublet(adata, scrub_doublets_kwargs=QUIET)
        return adata

    def test_writes_label_and_score(self, result):
        assert {"scrublet_label", "scrublet_score"} <= set(result.obs.columns)
        assert isinstance(result.obs["scrublet_label"].dtype, pd.CategoricalDtype)
        assert list(result.obs["scrublet_label"].cat.categories) == [
            "singlet",
            "doublet",
        ]
        assert result.obs["scrublet_score"].between(0, 1).all()

    def test_true_doublets_score_higher(self, result, doublet_data):
        is_doublet = doublet_data[1]
        score = result.obs["scrublet_score"].values

        assert score[is_doublet].mean() > 3 * score[~is_doublet].mean()
        # most injected doublets are in the top-scoring cells
        top = np.argsort(score)[-2 * is_doublet.sum() :]
        assert is_doublet[top].sum() >= 0.6 * is_doublet.sum()

    def test_labels_are_consistent_with_scores(self, result):
        score = result.obs["scrublet_score"]
        is_doublet = result.obs["scrublet_label"] == "doublet"

        assert is_doublet.any() and (~is_doublet).any()
        assert score[is_doublet].min() >= score[~is_doublet].max()
        # a modest number of cells is flagged (expected rate 10%)
        assert is_doublet.mean() < 0.2

    def test_deterministic_given_seed(self, result, doublet_data):
        again = doublet_data[0].copy()

        scrublet(again, scrub_doublets_kwargs=QUIET)

        pd.testing.assert_frame_equal(result.obs, again.obs)

    def test_key_added_and_layer(self, doublet_data):
        adata = doublet_data[0].copy()
        adata.layers["counts"] = adata.X.copy()
        adata.X = adata.X * 0  # X must be ignored when a layer is given

        scrublet(adata, layer="counts", key_added="scr", scrub_doublets_kwargs=QUIET)

        assert {"scr_label", "scr_score"} <= set(adata.obs.columns)
        assert "scrublet_label" not in adata.obs
        assert adata.obs["scr_score"].std() > 0

    def test_does_not_touch_other_data(self, doublet_data):
        adata = doublet_data[0].copy()
        X_before = adata.X.copy()

        scrublet(adata, scrub_doublets_kwargs=QUIET)

        assert (adata.X != X_before).nnz == 0
        assert adata.n_obs == X_before.shape[0]

    def test_missing_threshold_labels_everything_singlet(self, adata, monkeypatch):
        import scrublet as scrublet_pkg

        class _Fake:
            def __init__(self, **kwargs):
                self.n = kwargs["counts_matrix"].shape[0]

            def scrub_doublets(self, **kwargs):
                return np.linspace(0, 1, self.n), None

        monkeypatch.setattr(scrublet_pkg, "Scrublet", _Fake)

        scrublet(adata, scrub_doublets_kwargs={})

        assert (adata.obs["scrublet_label"] == "singlet").all()
        assert adata.obs["scrublet_score"].iloc[-1] == 1.0

    def test_parameters_are_forwarded(self, adata, monkeypatch):
        import scrublet as scrublet_pkg

        seen = {}

        class _Fake:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.n = kwargs["counts_matrix"].shape[0]

            def scrub_doublets(self, **kwargs):
                seen["scrub_kwargs"] = kwargs
                return np.zeros(self.n), np.zeros(self.n, bool)

        monkeypatch.setattr(scrublet_pkg, "Scrublet", _Fake)

        scrublet(
            adata,
            sim_doublet_ratio=3.0,
            n_neighbors=7,
            expected_doublet_rate=0.05,
            stdev_doublet_rate=0.01,
            random_state=5,
            scrub_doublets_kwargs={"min_counts": 1},
        )

        assert seen["sim_doublet_ratio"] == 3.0
        assert seen["n_neighbors"] == 7
        assert seen["expected_doublet_rate"] == 0.05
        assert seen["stdev_doublet_rate"] == 0.01
        assert seen["random_state"] == 5
        assert seen["scrub_kwargs"] == {"min_counts": 1}

    def test_import_error_when_not_installed(self, adata, monkeypatch):
        monkeypatch.setattr(scrub_mod, "find_spec", lambda name: None)

        with pytest.raises(ImportError, match="pip install scrublet"):
            scrublet(adata)


@needs_doubletdetection
class TestDoubletDetection:
    KW = dict(n_iters=2, n_components=10, random_state=0)

    @pytest.fixture(scope="class")
    def result(self, doublet_data):
        adata = doublet_data[0].copy()
        doubletdetection(adata, **self.KW)
        return adata

    def test_writes_cluster_label_and_score(self, result):
        obs = result.obs
        assert {
            "doubletdetection_cluster",
            "doubletdetection_label",
            "doubletdetection_score",
        } <= set(obs.columns)
        assert list(obs["doubletdetection_label"].cat.categories) == [
            "singlet",
            "doublet",
        ]
        clusters = obs["doubletdetection_cluster"]
        assert isinstance(clusters.dtype, pd.CategoricalDtype)
        assert all(isinstance(c, str) for c in clusters.cat.categories)
        assert 2 <= clusters.nunique() <= 20
        assert np.isfinite(obs["doubletdetection_score"]).all()

    def test_true_doublets_score_higher(self, result, doublet_data):
        is_doublet = doublet_data[1]
        score = result.obs["doubletdetection_score"].values

        assert np.median(score[is_doublet]) > np.median(score[~is_doublet])
        assert score[is_doublet].mean() > score[~is_doublet].mean() + 1

    def test_labels_flag_the_high_score_cells(self, result):
        score = result.obs["doubletdetection_score"]
        is_doublet = result.obs["doubletdetection_label"] == "doublet"

        if is_doublet.any():
            assert score[is_doublet].mean() > score[~is_doublet].mean()

    def test_key_added_and_layer(self, doublet_data):
        adata = doublet_data[0].copy()
        adata.layers["counts"] = adata.X.copy()
        adata.X = adata.X * 0

        doubletdetection(adata, layer="counts", key_added="dd", **self.KW)

        assert {"dd_cluster", "dd_label", "dd_score"} <= set(adata.obs.columns)
        assert "doubletdetection_label" not in adata.obs
        assert adata.obs["dd_score"].std() > 0


class TestScDblFinder:
    def test_unavailable_raises_helpful_error(self, adata):
        if scdblfinder_is_available():
            pytest.skip("scDblFinder available; error path not reachable")
        pytest.importorskip("rpy2")

        with pytest.raises((ImportError, ModuleNotFoundError)):
            scdblfinder(adata)

    def test_missing_rpy2_message(self, adata, monkeypatch, capsys):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("rpy2"):
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ModuleNotFoundError, match="rpy2 and anndata2ri"):
            scdblfinder(adata)

        assert "pip install rpy2 anndata2ri" in capsys.readouterr().out

    @needs_scdblfinder
    def test_unknown_clusters_column_is_rejected(self, adata):
        with pytest.raises(AssertionError, match="not found in adata.obs"):
            scdblfinder(adata, clusters_col="nope")

    @needs_scdblfinder
    def test_writes_label_and_score(self, adata, doublet_data):
        scdblfinder(adata, verbose=False)

        assert {"scDblFinder_label", "scDblFinder_score"} <= set(adata.obs.columns)
        assert list(adata.obs["scDblFinder_label"].cat.categories) == [
            "singlet",
            "doublet",
        ]
        score = adata.obs["scDblFinder_score"].values
        is_doublet = doublet_data[1]
        assert score[is_doublet].mean() > score[~is_doublet].mean()
        # original counts are untouched (works on a copy)
        assert len(adata.layers) == 0
