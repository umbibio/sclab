import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import stats
from scipy.sparse import csr_matrix, issparse

from sclab.tools.labeling.sctype import (
    _get_classification_scores_matrix,
    classify_cells,
)

TYPES = ["B", "M", "T"]  # alphabetical: the order of the categorical classes
N_PER = 60
N_GENES = 30


@pytest.fixture
def adata() -> AnnData:
    """Three cell types, each strongly expressing its own block of 4 genes
    (log-normalized values)."""
    rng = np.random.default_rng(0)
    labels = np.repeat(TYPES, N_PER)
    X = rng.poisson(1.0, (labels.size, N_GENES)).astype(float)
    for i, t in enumerate(TYPES):
        m = labels == t
        X[m, i * 4 : (i + 1) * 4] += rng.poisson(8, (m.sum(), 4))
    X = np.log1p(X)  # sctype expects log-normalized expression
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
        # a cluster with 2/3 T and 1/3 B cells is labelled T. Genes are z-scored
        # over all cells, so the mixed cluster must be scored within the full data
        # (z-scores over the cluster alone sum to zero and carry no signal).
        obs = adata.obs
        mixed = list(obs.index[obs["truth"] == "T"][:40]) + list(
            obs.index[obs["truth"] == "B"][:20]
        )
        adata.obs["cluster"] = pd.Categorical(
            np.where(obs.index.isin(mixed), "mixed", "rest")
        )

        classify_cells(adata, markers, "cell_type", cluster_key="cluster", threshold=0)

        in_mixed = adata.obs["cluster"] == "mixed"
        assert (adata.obs.loc[in_mixed, "cell_type"].astype(str) == "T").all()
        assert (adata.obs.loc[in_mixed, "cell_type_noNA"].astype(str) == "T").all()

    def test_layer_is_used(self, adata, markers):
        adata.layers["norm"] = adata.X.copy()
        adata.X = csr_matrix(
            np.log1p(np.random.default_rng(1).poisson(1, adata.shape).astype(float))
        )

        classify_cells(adata, markers, "cell_type", layer="norm")

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
        X[:3, 11] = np.log1p(5)
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

    def test_neighbor_pooling(self, with_neighbors, markers):
        classify_cells(with_neighbors, markers, "cell_type", neighbors_key="neighbors")

        assert _accuracy(with_neighbors, "cell_type_noNA") > 0.9

    def test_neighbor_pooling_scores(self, with_neighbors, markers):
        # pooled scores scale with the neighbourhood size for thresholding
        classify_cells(
            with_neighbors,
            markers,
            "cell_type",
            neighbors_key="neighbors",
            save_scores=True,
        )

        assert _accuracy(with_neighbors, "cell_type_noNA") > 0.9
        assert with_neighbors.obsm["cell_type_scores"].shape == (180, 3)

    def test_neighbor_pooling_with_layer(self, with_neighbors, markers):
        with_neighbors.layers["norm"] = with_neighbors.X.copy()
        with_neighbors.X = csr_matrix(with_neighbors.X.shape)

        classify_cells(
            with_neighbors,
            markers,
            "cell_type",
            layer="norm",
            neighbors_key="neighbors",
        )

        assert _accuracy(with_neighbors, "cell_type_noNA") > 0.9

    def test_raw_counts_raise(self, adata, markers):
        adata.X = csr_matrix(np.round(np.expm1(adata.X.toarray())))

        with pytest.raises(ValueError, match="adata.X looks like raw counts"):
            classify_cells(adata, markers, "cell_type")

    def test_raw_counts_layer_raises(self, adata, markers):
        adata.layers["counts"] = csr_matrix(np.round(np.expm1(adata.X.toarray())))

        with pytest.raises(ValueError, match="layer 'counts' looks like raw counts"):
            classify_cells(adata, markers, "cell_type", layer="counts")

    def test_invalid_specificity_raises(self, adata, markers):
        with pytest.raises(ValueError, match="specificity"):
            classify_cells(adata, markers, "cell_type", specificity="nope")

    def test_observed_range_specificity_classifies(self, adata, markers):
        classify_cells(adata, markers, "cell_type", specificity="observed_range")

        assert _accuracy(adata, "cell_type_noNA") == 1.0


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


def _weighted_scores(adata, markers, weights, key="cell_type"):
    """Independent re-implementation of the ScType score with explicit weights.

    `weights` maps gene -> specificity weight; genes with weight <= 0 or expressed
    in <= 5 cells are dropped. All markers are assumed positive.
    """
    X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    expressed = (X > X.min()).sum(axis=0) > 5
    marker_genes = set(markers["names"])
    genes = [
        g
        for g, e in zip(adata.var_names, expressed)
        if e and weights.get(g, 0.0) > 0 and g in marker_genes
    ]
    cols = [list(adata.var_names).index(g) for g in genes]
    Z = stats.zscore(X[:, cols], axis=0) * np.array([weights[g] for g in genes])
    classes = markers[key].cat.categories
    Xc = np.zeros((adata.n_obs, len(classes)))
    for c, cls in enumerate(classes):
        listed = markers.loc[markers[key] == cls, "names"]
        listed = set(listed[listed.isin(adata.var_names[expressed])])
        if len(listed) == 0:
            continue
        Xc[:, c] = Z[:, [g in listed for g in genes]].sum(axis=1) / np.sqrt(len(listed))
    return Xc


def _small_dataset(n_cells=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.log1p(rng.poisson(3.0, (n_cells, 5)).astype(float))
    return AnnData(
        X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )


def _four_type_markers():
    # g0: 1 type (w=1), g1: 2 types (w=2/3), g2: 4 types (w=0), g3: 1 type (w=1)
    rows = [
        ("T1", "g0"),
        ("T1", "g1"),
        ("T2", "g1"),
        ("T1", "g2"),
        ("T2", "g2"),
        ("T3", "g2"),
        ("T4", "g2"),
        ("T3", "g3"),
    ]
    df = pd.DataFrame(rows, columns=["cell_type", "names"])
    df["logfoldchanges"] = 1.0
    df["cell_type"] = pd.Categorical(df["cell_type"])
    return df


class TestSpecificity:
    weights = {"g0": 1.0, "g1": 2 / 3, "g3": 1.0}  # g2 (in all 4 types) -> 0, dropped

    def test_published_weights(self):
        adata, markers = _small_dataset(), _four_type_markers()

        Xc = _get_classification_scores_matrix(adata, markers, "cell_type")

        np.testing.assert_allclose(Xc, _weighted_scores(adata, markers, self.weights))

    def test_weights_do_not_depend_on_the_dataset(self):
        # g0 is expressed in only 5 cells of the subset, so it is filtered out there.
        # The weight of g1 must stay 2/3 (the legacy formula made it jump to 1 as
        # the observed max count changed).
        adata, markers = _small_dataset(), _four_type_markers()
        X = adata.X.copy()
        X[:10, 0] = 0
        adata.X = X
        sub = adata[:15].copy()
        assert (sub.X[:, 0] > 0).sum() == 5

        Xc = _get_classification_scores_matrix(sub, markers, "cell_type")

        np.testing.assert_allclose(Xc, _weighted_scores(sub, markers, self.weights))

    def test_single_class_table_is_not_penalized(self):
        adata = _small_dataset()
        markers = _four_type_markers().query("cell_type == 'T1'").copy()
        markers["cell_type"] = markers["cell_type"].cat.remove_unused_categories()

        Xc = _get_classification_scores_matrix(adata, markers, "cell_type")

        weights = {"g0": 1.0, "g1": 1.0, "g2": 1.0}
        np.testing.assert_allclose(Xc, _weighted_scores(adata, markers, weights))

    def test_observed_range_reproduces_legacy_formula(self, adata, markers):
        pos = markers.copy()
        pos["cell_type"] = pd.Categorical(pos["cell_type"], categories=TYPES)
        # legacy: x counts, min-max rescaled over the marker genes (min=1 < max=2)
        x = pos.groupby("names")["cell_type"].count()
        legacy = 1.0 - (x - x.min()) / (x.max() - x.min())
        legacy = legacy[legacy > 0].to_dict()
        assert x.min() < x.max()

        Xc = _get_classification_scores_matrix(
            adata, pos, "cell_type", specificity="observed_range"
        )

        np.testing.assert_allclose(Xc, _weighted_scores(adata, pos, legacy))
        # ...and it differs from the published weighting (g0: dropped vs 0.5)
        Xp = _get_classification_scores_matrix(adata, pos, "cell_type")
        assert not np.allclose(Xc, Xp)

    def test_observed_range_with_equal_counts_is_not_nan(self, adata):
        pos = _markers()
        pos["cell_type"] = pd.Categorical(pos["cell_type"], categories=TYPES)

        Xc = _get_classification_scores_matrix(
            adata, pos, "cell_type", specificity="observed_range"
        )

        assert np.isfinite(Xc).all()
        assert (Xc != 0).any()

    def test_penalize_false_gives_unit_weights(self):
        adata, markers = _small_dataset(), _four_type_markers()

        Xc = _get_classification_scores_matrix(
            adata, markers, "cell_type", penalize_non_specific=False
        )

        weights = {"g0": 1.0, "g1": 1.0, "g2": 1.0, "g3": 1.0}
        np.testing.assert_allclose(Xc, _weighted_scores(adata, markers, weights))
