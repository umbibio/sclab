import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.tools.labeling.sctype import (
    _get_classification_scores_matrix,
    classify_cells,
)

TYPES = ["B", "M", "T"]  # alphabetical: the order of the categorical classes
N_PER = 60
N_GENES = 30


@pytest.fixture
def adata() -> AnnData:
    """Three cell types, each strongly expressing its own block of 4 genes."""
    rng = np.random.default_rng(0)
    labels = np.repeat(TYPES, N_PER)
    X = rng.poisson(1.0, (labels.size, N_GENES)).astype(float)
    for i, t in enumerate(TYPES):
        m = labels == t
        X[m, i * 4 : (i + 1) * 4] += rng.poisson(8, (m.sum(), 4))
    obs = pd.DataFrame(
        {"truth": pd.Categorical(labels, categories=TYPES)},
        index=[f"c{i}" for i in range(labels.size)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)])
    return AnnData(csr_matrix(X), obs=obs, var=var)


def _markers(extra=()):
    rows = []
    for i, t in enumerate(TYPES):  # type i expresses genes g{4i}..g{4i+3}
        for j in range(3):
            rows.append((t, f"g{i * 4 + j}", 1.0))
    rows.extend(extra)
    return pd.DataFrame(rows, columns=["cell_type", "names", "logfoldchanges"])


@pytest.fixture
def markers() -> pd.DataFrame:
    # g0 marks both B and M, so marker specificity varies between genes
    return _markers(extra=[("M", "g0", 1.0)])


def _accuracy(adata, key):
    return (
        adata.obs[key].astype(str).values == adata.obs["truth"].astype(str).values
    ).mean()


class TestClassifyCells:
    def test_per_cell_classification(self, adata, markers):
        classify_cells(adata, markers, "cell_type")

        assert _accuracy(adata, "cell_type_noNA") == 1.0
        assert set(adata.obs["cell_type_noNA"].cat.categories) == set(TYPES)
        # confident calls agree with the best guess
        called = adata.obs["cell_type"].notna()
        assert called.mean() > 0.9
        assert (
            adata.obs.loc[called, "cell_type"].astype(str)
            == adata.obs.loc[called, "cell_type_noNA"].astype(str)
        ).all()

    def test_result_dtype_is_marker_category(self, adata, markers):
        classify_cells(adata, markers, "cell_type")

        assert adata.obs["cell_type"].dtype == markers["cell_type"].dtype
        assert list(adata.obs["cell_type"].cat.categories) == TYPES

    def test_key_added(self, adata, markers):
        classify_cells(adata, markers, "cell_type", key_added="annot")

        assert {"annot", "annot_noNA"} <= set(adata.obs.columns)
        assert "cell_type" not in adata.obs.columns

    def test_save_scores(self, adata, markers):
        classify_cells(adata, markers, "cell_type", save_scores=True)

        scores = adata.obsm["cell_type_scores"]
        assert scores.shape == (adata.n_obs, len(TYPES))
        np.testing.assert_array_equal(
            adata.obs["cell_type_score"].values, scores.max(axis=1)
        )
        best = np.array(TYPES)[scores.argmax(axis=1)]
        np.testing.assert_array_equal(best, adata.obs["cell_type_noNA"].astype(str))

    def test_scores_not_saved_by_default(self, adata, markers):
        classify_cells(adata, markers, "cell_type")

        assert "cell_type_scores" not in adata.obsm
        assert "cell_type_score" not in adata.obs

    def test_high_threshold_yields_unknown(self, adata, markers):
        classify_cells(adata, markers, "cell_type", threshold=1e6)

        assert adata.obs["cell_type"].isna().all()
        assert _accuracy(adata, "cell_type_noNA") == 1.0

    def test_threshold_monotonic(self, adata, markers):
        counts = []
        for thr in [0.0, 1.0, 2.0, 3.0]:
            classify_cells(adata, markers, "cell_type", threshold=thr)
            counts.append(int(adata.obs["cell_type"].notna().sum()))

        assert counts == sorted(counts, reverse=True)
        assert counts[0] == adata.n_obs

    def test_cluster_level_classification(self, adata, markers):
        adata.obs["cluster"] = pd.Categorical(adata.obs["truth"].astype(str))

        classify_cells(adata, markers, "cell_type", cluster_key="cluster")

        assert _accuracy(adata, "cell_type") == 1.0
        assert _accuracy(adata, "cell_type_noNA") == 1.0

    def test_cluster_level_unknown_below_threshold(self, adata, markers, caplog):
        adata.obs["cluster"] = pd.Categorical(adata.obs["truth"].astype(str))

        with caplog.at_level("WARNING"):
            classify_cells(
                adata, markers, "cell_type", cluster_key="cluster", threshold=1e6
            )

        assert adata.obs["cell_type"].isna().all()
        assert _accuracy(adata, "cell_type_noNA") == 1.0
        assert "classified as Unknown" in caplog.text

    def test_mixed_cluster_takes_majority(self, adata, markers):
        # one cluster with 2/3 T and 1/3 B cells is labelled T
        obs = adata.obs
        idx = list(obs.index[obs["truth"] == "T"][:40]) + list(
            obs.index[obs["truth"] == "B"][:20]
        )
        sub = adata[idx].copy()
        sub.obs["cluster"] = pd.Categorical(["only"] * len(idx))

        classify_cells(sub, markers, "cell_type", cluster_key="cluster", threshold=0)

        assert (sub.obs["cell_type"].astype(str) == "T").all()
        assert (sub.obs["cell_type_noNA"].astype(str) == "T").all()

    def test_layer_is_used(self, adata, markers):
        adata.layers["counts"] = adata.X.copy()
        adata.X = csr_matrix(
            np.random.default_rng(1).poisson(1, adata.shape).astype(float)
        )

        classify_cells(adata, markers, "cell_type", layer="counts")

        assert _accuracy(adata, "cell_type_noNA") == 1.0

    def test_dense_matrix(self, adata, markers):
        adata.X = adata.X.toarray()

        classify_cells(adata, markers, "cell_type")

        assert _accuracy(adata, "cell_type_noNA") == 1.0

    def test_negative_markers_lower_the_class_score(self, adata):
        # all B cells strongly express g0..g3. Declaring g1 a *negative* marker for
        # B must lower the B score of B cells relative to the positive-only markers.
        base = _markers()
        neg = _markers(extra=[("B", "g1", -1.0)])
        kw = dict(penalize_non_specific=False, save_scores=True)

        classify_cells(adata, base, "cell_type", **kw)
        s_base = adata.obsm["cell_type_scores"][:, TYPES.index("B")].copy()
        classify_cells(adata, neg, "cell_type", **kw)
        s_neg = adata.obsm["cell_type_scores"][:, TYPES.index("B")]

        is_b = (adata.obs["truth"] == "B").values
        assert s_neg[is_b].mean() < s_base[is_b].mean()
        # cells of other types are barely affected
        assert abs(s_neg[~is_b].mean() - s_base[~is_b].mean()) < abs(
            s_neg[is_b].mean() - s_base[is_b].mean()
        )

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: with penalize_non_specific=True, a marker table whose genes all "
        "appear once (max==min count) gets NaN specificity and is silently dropped; "
        "negative markers are ignored in that case",
    )
    def test_negative_markers_are_not_silently_dropped_by_default(self, adata):
        base = _markers()
        neg = _markers(extra=[("B", "g1", -1.0)])

        classify_cells(adata, base, "cell_type", save_scores=True)
        s_base = adata.obsm["cell_type_scores"][:, TYPES.index("B")].copy()
        classify_cells(adata, neg, "cell_type", save_scores=True)
        s_neg = adata.obsm["cell_type_scores"][:, TYPES.index("B")]

        is_b = (adata.obs["truth"] == "B").values
        assert s_neg[is_b].mean() < s_base[is_b].mean()

    def test_marker_class_key_inferred_from_single_category_column(
        self, adata, markers
    ):
        markers["cell_type"] = markers["cell_type"].astype("category")

        classify_cells(adata, markers)

        assert "cell_type" in adata.obs
        assert _accuracy(adata, "cell_type_noNA") == 1.0

    def test_ambiguous_category_columns_raise(self, adata, markers):
        markers["cell_type"] = markers["cell_type"].astype("category")
        markers["other"] = markers["cell_type"].copy()

        with pytest.raises(AssertionError, match="exactly one column"):
            classify_cells(adata, markers)

    def test_string_marker_column_is_converted_to_category(self, adata, markers):
        assert not markers["cell_type"].dtype.name.startswith("category")

        classify_cells(adata, markers, "cell_type")

        assert markers["cell_type"].dtype.name == "category"

    def test_genes_with_few_expressing_cells_are_ignored(self, adata, markers):
        # a marker gene expressed in <= 5 cells is excluded from scoring
        X = adata.X.toarray()
        X[:, 11] = 0
        X[:3, 11] = 5
        adata.X = csr_matrix(X)
        extra = markers.copy()
        extra.loc[len(extra)] = ("M", "g11", 1.0)  # g11 belongs to the T block

        classify_cells(adata, extra, "cell_type")

        assert _accuracy(adata, "cell_type_noNA") == 1.0

    def test_penalize_non_specific_false(self, adata):
        markers = _markers()  # every marker gene specific to exactly one class

        classify_cells(
            adata, markers, "cell_type", penalize_non_specific=False, save_scores=True
        )

        assert _accuracy(adata, "cell_type_noNA") == 1.0
        assert adata.obs["cell_type"].notna().mean() > 0.9

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: with penalize_non_specific=True and every marker gene unique to "
        "one class, specificity is 0/0=NaN, all genes are dropped and every cell "
        "is scored 0 / unclassified",
    )
    def test_all_specific_markers_with_default_penalty(self, adata):
        classify_cells(adata, _markers(), "cell_type")

        assert adata.obs["cell_type"].notna().mean() > 0.9

    @pytest.fixture
    def with_neighbors(self, adata):
        adata.obsm["X_toy"] = adata.X.toarray()
        sc.pp.neighbors(
            adata,
            n_neighbors=5,
            use_rep="X_toy",
            method="gauss",
            transformer="sklearn",
        )
        return adata

    @pytest.mark.xfail(
        strict=True,
        reason="BUG: sctype calls pool_neighbors(..., layer=layer) but pool_neighbors "
        "has no `layer` parameter -> TypeError whenever neighbors_key is given",
    )
    def test_neighbor_pooling(self, with_neighbors, markers):
        classify_cells(with_neighbors, markers, "cell_type", neighbors_key="neighbors")

        assert _accuracy(with_neighbors, "cell_type_noNA") > 0.9

    def test_neighbor_pooling_downstream_logic(
        self, with_neighbors, markers, monkeypatch
    ):
        # Bypass the `layer=` TypeError above to exercise the rest of the pooled
        # path: pooled scores scale with the neighbourhood size for thresholding.
        from sclab.preprocess import pool_neighbors as real_pool
        from sclab.tools.labeling import sctype as sctype_mod

        def pool(adata, layer=None, **kw):
            return real_pool(adata, key=layer, **kw)

        monkeypatch.setattr(sctype_mod, "pool_neighbors", pool)

        classify_cells(
            with_neighbors,
            markers,
            "cell_type",
            neighbors_key="neighbors",
            save_scores=True,
        )

        assert _accuracy(with_neighbors, "cell_type_noNA") > 0.9
        assert with_neighbors.obsm["cell_type_scores"].shape == (180, 3)


class TestScoreMatrix:
    def test_shape_and_class_ordering(self, adata, markers):
        m = markers.copy()
        m["cell_type"] = pd.Categorical(m["cell_type"], categories=TYPES)

        Xc = _get_classification_scores_matrix(
            adata, m.query("logfoldchanges > 0"), "cell_type"
        )

        assert Xc.shape == (adata.n_obs, 3)
        truth = adata.obs["truth"].astype(str).values
        for i, t in enumerate(TYPES):
            own = Xc[truth == t, TYPES.index(t)].mean()
            other = np.delete(Xc[truth == t], TYPES.index(t), axis=1).mean()
            assert own > other

    def test_unknown_class_gets_zero_score(self, adata):
        rows = _markers()
        rows.loc[len(rows)] = ("Unknown", "g20", 1.0)
        rows["cell_type"] = pd.Categorical(
            rows["cell_type"], categories=TYPES + ["Unknown"]
        )  # "Unknown" sorts last

        Xc = _get_classification_scores_matrix(
            adata,
            rows.query("logfoldchanges > 0"),
            "cell_type",
            penalize_non_specific=False,
        )

        assert (Xc[:, -1] == 0).all()
