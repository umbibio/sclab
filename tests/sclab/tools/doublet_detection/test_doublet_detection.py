import sys
import warnings
from importlib.util import find_spec

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
from sclab.utils import layer_keys

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
        monkeypatch.setattr(scdbl_mod, "_pyscdblfinder_available", lambda: False)
        monkeypatch.setattr(scdbl_mod, "_python_deps_available", lambda: False)
        monkeypatch.setattr(scdbl_mod, "_r_deps_available", lambda: True)
        assert scdbl_mod.scdblfinder_is_available() is False

        monkeypatch.setattr(scdbl_mod, "_python_deps_available", lambda: True)
        assert scdbl_mod.scdblfinder_is_available() is True

    def test_scdblfinder_available_with_python_port_only(self, monkeypatch):
        monkeypatch.setattr(scdbl_mod, "_r_available", lambda: False)
        monkeypatch.setattr(scdbl_mod, "_pyscdblfinder_available", lambda: True)
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

    def test_deterministic_with_threaded_annoy(self):
        # here annoy's default multi-threaded build changed the scores run to run
        import annoy

        from ...test_utils import simple_loop_adata

        adata = simple_loop_adata(n_obs=300, n_vars=300)
        scores = []
        for _ in range(4):
            scrublet(adata, scrub_doublets_kwargs=QUIET)
            scores.append(adata.obs["scrublet_score"].to_numpy())
        for other in scores[1:]:
            np.testing.assert_array_equal(scores[0], other)
        assert annoy.AnnoyIndex.__module__ == "annoy"  # restored

    def test_without_annoy_umap_numba(self, result, doublet_data, monkeypatch):
        # as under Pyodide (scrublet installed with deps=False): exact neighbors
        for name in ("annoy", "umap", "numba", "pynndescent"):
            monkeypatch.setitem(sys.modules, name, None)
        adata = doublet_data[0].copy()
        scrublet(adata, scrub_doublets_kwargs=QUIET)
        exact = adata.obs["scrublet_score"].to_numpy()
        scrublet(adata, scrub_doublets_kwargs=QUIET)
        np.testing.assert_array_equal(exact, adata.obs["scrublet_score"].to_numpy())

        is_doublet = doublet_data[1]
        assert exact[is_doublet].mean() > 3 * exact[~is_doublet].mean()
        approx = result.obs["scrublet_score"].to_numpy()
        assert np.corrcoef(exact, approx)[0, 1] > 0.9

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
        monkeypatch.setitem(sys.modules, "scrublet", None)

        with pytest.raises(ImportError, match="Needs scrublet: pip install"):
            scrublet(adata)

    def test_exact_neighbors_without_annoy(self, adata, monkeypatch):
        import scrublet as scrublet_pkg

        seen = {}

        class _Fake:
            def __init__(self, **kwargs):
                self.n = kwargs["counts_matrix"].shape[0]

            def scrub_doublets(self, **kwargs):
                seen.update(kwargs)
                return np.zeros(self.n), np.zeros(self.n, bool)

        monkeypatch.setattr(scrublet_pkg, "Scrublet", _Fake)
        scrublet(adata, scrub_doublets_kwargs=QUIET)
        assert seen["use_approx_neighbors"] is True  # annoy installed: unchanged

        monkeypatch.setitem(sys.modules, "annoy", None)
        scrublet(adata, scrub_doublets_kwargs=QUIET)
        assert seen["use_approx_neighbors"] is False
        assert QUIET["use_approx_neighbors"] is True  # the caller's dict is untouched


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
            scdblfinder(adata, backend="r")

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
        assert layer_keys(adata) == []


class _Stop(Exception):
    pass


class TestScDblFinderBackend:
    @pytest.fixture
    def calls(self, monkeypatch):
        """Record which backend runs, without running it."""
        seen = []

        def python(*args, **kwargs):
            seen.append("python")
            raise _Stop

        def r():
            seen.append("r")
            raise _Stop

        monkeypatch.setattr(scdbl_mod, "_scdblfinder_python", python)
        monkeypatch.setattr(scdbl_mod, "_try_imports", r)
        return seen

    @pytest.mark.parametrize("r_available, expected", [(True, "r"), (False, "python")])
    def test_auto_prefers_r(self, adata, calls, monkeypatch, r_available, expected):
        monkeypatch.setattr(scdbl_mod, "_r_available", lambda: r_available)
        with pytest.raises(_Stop):
            scdblfinder(adata)
        assert calls == [expected]

    @pytest.mark.parametrize("backend", ["r", "python"])
    def test_explicit_backend(self, adata, calls, monkeypatch, backend):
        monkeypatch.setattr(scdbl_mod, "_r_available", lambda: backend != "r")
        with pytest.raises(_Stop):
            scdblfinder(adata, backend=backend)
        assert calls == [backend]

    def test_unknown_backend(self, adata):
        with pytest.raises(ValueError, match="backend must be"):
            scdblfinder(adata, backend="julia")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(known_doublets_col="kd"), "known_doublets_col"),
            (dict(samples_col="sample"), "samples_col"),
            (dict(clust_cor=5), "clust_cor"),
            (dict(score="weighted"), "score='weighted'"),
        ],
    )
    def test_python_rejects_unported_options(self, adata, kwargs, match):
        adata.obs["kd"] = False
        adata.obs["sample"] = "a"
        with pytest.raises(NotImplementedError, match=match):
            scdblfinder(adata, backend="python", **kwargs)

    def test_python_missing_package(self, adata, monkeypatch):
        monkeypatch.setattr(scdbl_mod, "_pyscdblfinder_available", lambda: False)
        with pytest.raises(ImportError, match=r"sclab\[scdblfinder-py\]"):
            scdblfinder(adata, backend="python")


@pytest.mark.skipif(
    not scdbl_mod._pyscdblfinder_available(), reason="pyscdblfinder not installed"
)
class TestScDblFinderPython:
    @pytest.fixture(scope="class")
    def result(self, doublet_data):
        adata = doublet_data[0].copy()
        scdblfinder(adata, backend="python", clusters_col=True, verbose=False)
        return adata

    def test_same_columns_as_r(self, result):
        obs = result.obs
        assert list(obs.columns) == [
            "scDblFinder_cluster",
            "scDblFinder_label",
            "scDblFinder_score",
        ]
        assert list(obs["scDblFinder_label"].cat.categories) == ["singlet", "doublet"]
        assert all(
            isinstance(c, str) for c in obs["scDblFinder_cluster"].cat.categories
        )
        assert obs["scDblFinder_score"].between(0, 1).all()

    def test_true_doublets_score_higher(self, result, doublet_data):
        score = result.obs["scDblFinder_score"].to_numpy()
        is_doublet = doublet_data[1]
        assert score[is_doublet].mean() > 3 * score[~is_doublet].mean()

    def test_clusters_column(self, doublet_data):
        adata = doublet_data[0].copy()
        adata.obs["leiden"] = pd.Categorical(
            np.repeat(["0", "1", "2", "3"], [200, 200, 150, 25])
        )
        scdblfinder(adata, backend="python", clusters_col="leiden", verbose=False)
        assert (
            adata.obs["scDblFinder_cluster"]
            .astype(str)
            .equals(adata.obs["leiden"].astype(str))
        )


class TestFastcluster:
    @pytest.fixture(autouse=True)
    def _needs_igraph(self):
        pytest.importorskip("igraph")

    def test_recovers_populations(self, doublet_data):
        from sklearn.metrics import adjusted_rand_score

        from sclab.tools.doublet_detection._fastcluster import fastcluster

        X, is_doublet = doublet_data[0].X, doublet_data[1]
        labels = fastcluster(X)
        truth = np.repeat([0, 1, 2], [200, 200, 150])
        assert labels.min() == 1  # 1-based, as R
        assert adjusted_rand_score(truth, labels[~is_doublet]) > 0.9
        np.testing.assert_array_equal(labels, fastcluster(X))  # seeded

    def test_metacells_above_1000_cells(self):
        from sklearn.metrics import adjusted_rand_score

        from sclab.tools.doublet_detection._fastcluster import fastcluster

        rng = np.random.default_rng(1)
        mus = rng.gamma(1, 1, (3, 200)) * (rng.random((3, 200)) < 0.4)
        truth = np.repeat([0, 1, 2], 400)
        X = rng.poisson(mus[truth] * 3).astype(float)
        labels = fastcluster(X, random_state=3)
        assert adjusted_rand_score(truth, labels) > 0.9


@pytest.mark.skipif(not scdbl_mod._r_available(), reason="requires R scDblFinder")
class TestScDblFinderR:
    @pytest.fixture(scope="class")
    def result(self, doublet_data):
        adata = doublet_data[0].copy()
        adata.obs["kd"] = False
        adata.obs.iloc[-5:, 0] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scdblfinder(
                adata,
                backend="r",
                clusters_col=True,  # R's fast clustering
                known_doublets_col="kd",
                dbr_sd=0.02,  # mapped to dbr.sd
                verbose=False,
            )
        return adata, caught

    def test_fixed_arguments_run(self, result):
        adata, _ = result
        assert {"scDblFinder_cluster", "scDblFinder_label", "scDblFinder_score"} <= set(
            adata.obs.columns
        )
        assert adata.obs["scDblFinder_cluster"].nunique() >= 2

    def test_known_doublets_column_survives(self, result):
        adata, _ = result
        assert adata.obs["kd"].sum() == 5
        # with knownDoublets R returns the class as a factor; all cells labelled
        assert adata.obs["scDblFinder_label"].notna().all()

    def test_warning_matches_installed_versions(self, result):
        from rpy2.robjects import r

        def version(package):
            return tuple(int(v) for v in r(f'unlist(packageVersion("{package}"))'))

        sdf, xgb = version("scDblFinder"), version("xgboost")
        broken = (xgb >= (3,) and sdf < (1, 24)) or (xgb >= (3, 4) and sdf < (1, 27, 9))
        fired = any("cannot train its classifier" in str(w.message) for w in result[1])
        assert fired == broken


@pytest.mark.skipif(
    find_spec("rpy2") is None, reason="rpy2 needed to patch rpy2.robjects.r"
)
class TestXgboostMismatchWarning:
    @pytest.fixture
    def versions(self, monkeypatch):
        import rpy2.robjects

        installed = {}
        monkeypatch.setattr(
            rpy2.robjects, "r", lambda code: installed[code.split('"')[1]]
        )
        return installed

    @pytest.mark.parametrize(
        "sdf, xgb, fix",
        [
            ([1, 20, 2], [3, 2, 1, 1], "xgboost 1.7.x"),
            ([1, 16, 0], [3, 0, 0], "xgboost 1.7.x"),
            ([1, 24, 7], [3, 4, 0], "scDblFinder to >= 1.27.9"),
        ],
    )
    def test_warns_on_broken_combinations(self, versions, sdf, xgb, fix):
        versions.update(scDblFinder=sdf, xgboost=xgb)
        with pytest.warns(UserWarning, match="cannot train its classifier") as w:
            scdbl_mod._warn_on_broken_xgboost()
        assert fix in str(w[0].message)
        assert "heuristic" in str(w[0].message)

    @pytest.mark.parametrize(
        "sdf, xgb",
        [
            ([1, 20, 2], [1, 7, 11, 1]),
            ([1, 24, 7], [3, 2, 0]),
            ([1, 27, 12], [3, 4, 1]),
        ],
    )
    def test_silent_on_working_combinations(self, versions, sdf, xgb):
        versions.update(scDblFinder=sdf, xgboost=xgb)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scdbl_mod._warn_on_broken_xgboost()


class TestCheckCounts:
    @pytest.fixture
    def normalized(self, adata):
        adata.layers["counts"] = adata.X.copy()
        adata.X = adata.X.multiply(1 / 3.0).tocsr()  # non-integer values
        return adata

    @needs_scrublet
    def test_scrublet_rejects_normalized_input(self, normalized):
        with pytest.raises(ValueError, match="layer 'X' does not look like raw"):
            scrublet(normalized, scrub_doublets_kwargs=QUIET)
        assert "scrublet_score" not in normalized.obs

    @needs_scrublet
    def test_scrublet_check_can_be_bypassed(self, normalized):
        scrublet(normalized, check_counts=False, scrub_doublets_kwargs=QUIET)
        assert "scrublet_score" in normalized.obs

    @needs_scrublet
    def test_scrublet_checks_the_selected_layer(self, normalized):
        scrublet(normalized, layer="counts", scrub_doublets_kwargs=QUIET)
        normalized.layers["bad"] = normalized.X.copy()
        with pytest.raises(ValueError, match="layer 'bad'"):
            scrublet(normalized, layer="bad", scrub_doublets_kwargs=QUIET)

    @needs_doubletdetection
    def test_doubletdetection_rejects_normalized_input(self, normalized):
        with pytest.raises(ValueError, match="does not look like raw counts"):
            doubletdetection(normalized, n_iters=1, n_components=5)

    @needs_doubletdetection
    def test_doubletdetection_check_can_be_bypassed(self, normalized):
        doubletdetection(normalized, check_counts=False, n_iters=1, n_components=5)
        assert "doubletdetection_score" in normalized.obs

    @needs_scdblfinder
    def test_scdblfinder_rejects_normalized_input(self, normalized):
        with pytest.raises(ValueError, match="does not look like raw counts"):
            scdblfinder(normalized, verbose=False)

    @needs_scdblfinder
    def test_scdblfinder_check_can_be_bypassed(self, normalized):
        scdblfinder(normalized, check_counts=False, verbose=False)
        assert "scDblFinder_score" in normalized.obs
