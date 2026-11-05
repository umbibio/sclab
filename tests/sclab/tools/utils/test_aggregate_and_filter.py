import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.tools.utils import aggregate_and_filter
from sclab.tools.utils._aggregate_and_filter import (
    _get_groups_to_drop,
    _groups,
    _get_layer,
    _get_replica_idxs,
    _including,
)


@pytest.fixture(autouse=True)
def _seed():
    import random

    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def adata() -> AnnData:
    rng = np.random.default_rng(0)
    n = 240
    batch = np.repeat(["A", "B", "C"], [100, 100, 40])
    ct = np.tile(["x", "y"], n // 2)
    X = rng.poisson(3, (n, 6)).astype(float)
    obs = pd.DataFrame({"batch": batch, "ct": ct}, index=[f"c{i}" for i in range(n)])
    ad = AnnData(csr_matrix(X), obs=obs, var=pd.DataFrame(index=list("abcdef")))
    ad.layers["counts"] = csr_matrix(X * 2)
    return ad


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


class TestAggregation:
    def test_sums_are_conserved_per_group(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=3, min_cells_per_group=1
        )

        X = _dense(adata.X)
        for g in ["A", "B", "C"]:
            rows = res.obs.index[res.obs["batch"] == g]
            expected = X[(adata.obs["batch"] == g).values].sum(axis=0)
            np.testing.assert_array_equal(_dense(res[rows].X).sum(axis=0), expected)
        np.testing.assert_array_equal(_dense(res.X).sum(axis=0), X.sum(axis=0))

    def test_replicas_partition_cells(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=4, min_cells_per_group=1
        )

        assert res.n_obs == 12
        assert res.obs["replica_size"].sum() == adata.n_obs
        sizes = res.obs.loc[res.obs["batch"] == "A", "replica_size"]
        assert sizes.tolist() == [25, 25, 25, 25]
        assert res.obs["replica"].dtype == "category"
        assert set(res.obs["replica"]) == {"1", "2", "3", "4"}

    def test_obs_index_naming(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=2, min_cells_per_group=1
        )

        assert list(res.obs_names) == [
            "A_rep1",
            "A_rep2",
            "B_rep1",
            "B_rep2",
            "C_rep1",
            "C_rep2",
        ]
        assert list(res.var_names) == list(adata.var_names)

    def test_uneven_split_distributes_remainder(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=3, min_cells_per_group=1
        )

        sizes = res.obs.loc[res.obs["batch"] == "C", "replica_size"].tolist()
        assert sorted(sizes, reverse=True) == [14, 13, 13]

    def test_dense_and_sparse_agree(self, adata):
        dense = AnnData(adata.X.toarray(), obs=adata.obs.copy(), var=adata.var.copy())

        r1 = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=1
        )
        r2 = aggregate_and_filter(
            dense, "batch", replicas_per_group=1, min_cells_per_group=1
        )

        np.testing.assert_array_equal(r1.X, r2.X)
        pd.testing.assert_frame_equal(r1.var, r2.var)

    def test_layer_selection(self, adata):
        res = aggregate_and_filter(
            adata,
            "batch",
            layer="counts",
            replicas_per_group=1,
            min_cells_per_group=1,
        )
        base = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=1
        )

        np.testing.assert_array_equal(res.X, 2 * base.X)
        np.testing.assert_array_equal(res.var["tot_expr_A"], 2 * base.var["tot_expr_A"])

    def test_layer_X_means_adata_X(self, adata):
        a = aggregate_and_filter(
            adata, "batch", layer="X", replicas_per_group=1, min_cells_per_group=1
        )
        b = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=1
        )

        np.testing.assert_array_equal(a.X, b.X)

    def test_does_not_modify_input(self, adata):
        obs_before = adata.obs.copy()
        X_before = adata.X.copy()

        aggregate_and_filter(adata, "batch", min_cells_per_group=1)

        pd.testing.assert_frame_equal(adata.obs, obs_before)
        assert (adata.X != X_before).nnz == 0
        assert not isinstance(
            adata.obs["batch"].dtype, pd.CategoricalDtype
        )  # not converted in place

    def test_bootstrap_resamples_with_replacement(self, adata):
        res = aggregate_and_filter(
            adata,
            "batch",
            replicas_per_group=3,
            min_cells_per_group=1,
            bootstrap_sampling=True,
        )

        assert (
            res.obs["replica_size"].groupby(res.obs["batch"], observed=True).first()
            == [100, 100, 40]
        ).all()
        assert res.obs["replica_size"].sum() == adata.n_obs * 3
        # resampled sums differ from the plain per-group sum
        plain = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=1
        )
        boot_a = _dense(res[res.obs["batch"] == "A"].X)
        assert not np.array_equal(
            boot_a[0], _dense(plain[plain.obs["batch"] == "A"].X)[0]
        )

    def test_group_dtype_is_preserved(self, adata):
        adata.obs["batch"] = pd.Categorical(
            adata.obs["batch"], categories=["C", "B", "A"]
        )

        res = aggregate_and_filter(adata, "batch", min_cells_per_group=1)

        assert list(res.obs["batch"].cat.categories) == ["C", "B", "A"]


class TestFiltering:
    def test_small_groups_are_dropped(self, adata, capsys):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=2, min_cells_per_group=50
        )

        assert set(res.obs["batch"]) == {"A", "B"}
        assert "Dropping the following groups" in capsys.readouterr().out
        assert not any("C_" in c for c in res.var.columns)

    def test_rest_statistics_include_dropped_cells(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=50
        )

        X = _dense(adata.X)
        # "not A" counts B and the (dropped) C cells as well
        np.testing.assert_array_equal(
            res.var["tot_expr_notA"].values,
            X[(adata.obs["batch"] != "A").values].sum(axis=0),
        )

    def test_use_cells_subsets_before_grouping(self, adata):
        res = aggregate_and_filter(
            adata,
            "batch",
            use_cells={"ct": ["x"]},
            replicas_per_group=1,
            min_cells_per_group=1,
        )

        X = _dense(adata.X)
        sel = (adata.obs["batch"] == "A") & (adata.obs["ct"] == "x")
        np.testing.assert_array_equal(
            _dense(res[["A_rep1"]].X)[0], X[sel.values].sum(axis=0)
        )
        assert res.obs["replica_size"].sum() == (adata.obs["ct"] == "x").sum()

    def test_use_cells_multiple_keys_intersect(self, adata):
        res = aggregate_and_filter(
            adata,
            "ct",
            use_cells={"batch": ["A", "B"], "ct": ["y"]},
            replicas_per_group=1,
            min_cells_per_group=1,
        )

        assert res.n_obs == 1
        assert res.obs["replica_size"].iloc[0] == 100

    def test_cells_with_missing_identity_are_ignored(self, adata):
        adata.obs["ct"] = pd.Categorical(adata.obs["ct"])
        adata.obs.loc[adata.obs_names[:10], "ct"] = np.nan

        res = aggregate_and_filter(
            adata,
            "batch",
            "ct",
            replicas_per_group=1,
            min_cells_per_group=1,
        )

        assert res.obs["replica_size"].sum() == adata.n_obs - 10


class TestCellIdentity:
    def test_crossed_groups(self, adata):
        res = aggregate_and_filter(
            adata, "batch", "ct", replicas_per_group=1, min_cells_per_group=10
        )

        assert res.n_obs == 6
        assert "A_x_rep1" in res.obs_names
        assert res.obs.loc["A_x_rep1", "ct"] == "x"
        assert res.obs.loc["C_y_rep1", "replica_size"] == 20
        for col in ["pct_expr_A_x", "pct_expr_notA_x", "tot_expr_C_y"]:
            assert col in res.var.columns

    def test_rest_statistics_stay_within_identity(self, adata):
        res = aggregate_and_filter(
            adata, "batch", "ct", replicas_per_group=1, min_cells_per_group=10
        )

        X = _dense(adata.X)
        rest_x = ((adata.obs["batch"] != "A") & (adata.obs["ct"] == "x")).values
        np.testing.assert_allclose(res.var["pct_expr_notA_x"], (X[rest_x] > 0).mean(0))
        np.testing.assert_array_equal(res.var["tot_expr_notA_x"], X[rest_x].sum(0))

    def test_min_cells_applies_to_crossed_groups(self, adata):
        res = aggregate_and_filter(
            adata, "batch", "ct", replicas_per_group=1, min_cells_per_group=30
        )

        # C has only 20 cells per cell type
        assert set(res.obs["batch"]) == {"A", "B"}


class TestStats:
    def test_pct_num_tot_columns(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=1
        )

        X = _dense(adata.X)
        a = (adata.obs["batch"] == "A").values
        np.testing.assert_array_equal(res.var["tot_expr_A"], X[a].sum(axis=0))
        np.testing.assert_array_equal(res.var["num_expr_A"], (X[a] > 0).sum(axis=0))
        np.testing.assert_allclose(res.var["pct_expr_A"], (X[a] > 0).mean(axis=0))
        np.testing.assert_allclose(res.var["pct_expr_notA"], (X[~a] > 0).mean(axis=0))
        np.testing.assert_array_equal(
            res.var["tot_expr_A"] + res.var["tot_expr_notA"], X.sum(axis=0)
        )
        assert res.var.shape[1] == 3 * 2 * 3  # 3 stats x (group, rest) x 3 groups

    def test_make_stats_false_gives_empty_var(self, adata):
        res = aggregate_and_filter(
            adata, "batch", min_cells_per_group=1, make_stats=False
        )

        assert res.var.shape[1] == 0
        assert list(res.var_names) == list(adata.var_names)


class TestDummies:
    def test_dummy_columns(self, adata):
        res = aggregate_and_filter(
            adata, "batch", replicas_per_group=1, min_cells_per_group=1
        )

        assert res.obs.loc["A_rep1", ["batch_A", "batch_B", "batch_C"]].tolist() == [
            "A",
            "notB",
            "notC",
        ]
        assert res.obs.loc["B_rep1", ["batch_A", "batch_B", "batch_C"]].tolist() == [
            "notA",
            "B",
            "notC",
        ]

    def test_make_dummies_false(self, adata):
        res = aggregate_and_filter(
            adata, "batch", min_cells_per_group=1, make_dummies=False
        )

        assert not any(c.startswith("batch_") for c in res.obs.columns)


class TestHelpers:
    def test_including(self):
        assert _including(("A", "x"), [])
        assert _including(("A",), [])
        assert not _including(("A", "x"), [("A", "x")])
        assert not _including(("A", "x"), ["A"])
        assert not _including(("A",), [("A",)])
        assert not _including(("A", float("nan")), [])

    def test_get_groups_to_drop_handles_scalar_and_tuple_keys(self, adata, capsys):
        single = _get_groups_to_drop(adata, ["batch"], 50)
        assert "C" in single and ("C",) in single

        double = _get_groups_to_drop(adata, ["batch", "ct"], 30)
        assert ("C", "x") in double and ("C", "y") in double
        assert ("A", "x") not in double

    def test_get_replica_idxs_partitions(self, adata):
        sub = adata[adata.obs["batch"] == "C"]

        parts = _get_replica_idxs(sub, 3, bootstrap_sampling=False)

        assert len(parts) == 3
        assert sorted(np.concatenate(parts)) == sorted(sub.obs_names)

    def test_get_replica_idxs_bootstrap_shape(self, adata):
        sub = adata[adata.obs["batch"] == "C"]

        parts = _get_replica_idxs(sub, 3, bootstrap_sampling=True)

        assert parts.shape == (3, 40)
        assert set(parts.ravel()) <= set(sub.obs_names)

    def test_get_layer(self, adata):
        assert _get_layer(adata, None) is adata.X
        assert _get_layer(adata, "X") is adata.X
        assert _get_layer(adata, "counts") is adata.layers["counts"]
        dense = _get_layer(adata, "counts", dense=True)
        assert isinstance(dense, np.ndarray)
        np.testing.assert_array_equal(dense, adata.layers["counts"].toarray())


class TestSampleKey:
    def test_one_pseudobulk_per_sample(self, adata):
        adata.obs["sample"] = np.where(np.arange(adata.n_obs) % 2, "s1", "s2")

        res = aggregate_and_filter(
            adata,
            "batch",
            sample_key="sample",
            replicas_per_group=4,
            min_cells_per_group=1,
        )

        assert list(res.obs_names) == ["A_s1", "A_s2", "B_s1", "B_s2", "C_s1", "C_s2"]
        assert res.obs.loc["B_s2", "replica"] == "s2"
        X = _dense(adata.X)
        cells = ((adata.obs["batch"] == "B") & (adata.obs["sample"] == "s2")).values
        np.testing.assert_array_equal(_dense(res[["B_s2"]].X)[0], X[cells].sum(0))
        assert res.obs.loc["B_s2", "replica_size"] == cells.sum()

    def test_nested_samples_with_identity(self, adata):
        adata.obs["sample"] = adata.obs["batch"] + "1"

        res = aggregate_and_filter(
            adata, "batch", "ct", sample_key="sample", min_cells_per_group=1
        )

        assert "A_x_A1" in res.obs_names and res.n_obs == 6


class TestRandomState:
    def test_seeded_replicas_are_reproducible(self, adata):
        a = aggregate_and_filter(adata, "batch", min_cells_per_group=1, random_state=1)
        import random

        random.seed(123)  # global state is not used
        b = aggregate_and_filter(adata, "batch", min_cells_per_group=1, random_state=1)
        c = aggregate_and_filter(adata, "batch", min_cells_per_group=1, random_state=2)

        np.testing.assert_array_equal(a.X, b.X)
        assert not np.array_equal(a.X, c.X)

    def test_seeded_bootstrap_is_reproducible(self, adata):
        kw = dict(min_cells_per_group=1, bootstrap_sampling=True, random_state=1)
        np.testing.assert_array_equal(
            aggregate_and_filter(adata, "batch", **kw).X,
            aggregate_and_filter(adata, "batch", **kw).X,
        )


class TestPandas4Groupby:
    def test_single_key_groups_are_scalar(self, adata):
        keys = list(_groups(adata, ["batch"]))
        assert keys == ["A", "B", "C"]
        assert all(isinstance(k, tuple) for k in _groups(adata, ["batch", "ct"]))

    def test_no_tuple_key_warning(self, adata):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*keys of `groups`.*")
            aggregate_and_filter(adata, "batch", min_cells_per_group=1)
            aggregate_and_filter(adata, "batch", "ct", min_cells_per_group=1)
