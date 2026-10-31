import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.dataset import SCLabDataset
from sclab.dataset._exceptions import InvalidRowSubset


def make_adata(n_obs=12, n_vars=5, seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(2.0, size=(n_obs, n_vars)).astype(float))
    obs = pd.DataFrame(
        {
            "num": np.arange(n_obs, dtype=float),
            "cat": pd.Categorical(["a", "b", "c"] * (n_obs // 3)),
            "few_str": ["x", "y"] * (n_obs // 2),
            "many_str": [f"s{i}" for i in range(n_obs)],
            "flag": [True, False] * (n_obs // 2),
        },
        index=[f"cell{i}" for i in range(n_obs)],
    )
    obsm = {"X_emb": rng.normal(size=(n_obs, 3))}
    return AnnData(
        X, obs=obs, obsm=obsm, var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    )


@pytest.fixture
def ds() -> SCLabDataset:
    return SCLabDataset(make_adata())


def record(ds: SCLabDataset, event: str) -> list:
    calls = []
    original = ds.broker.publish

    def spy(ev, *a, **k):
        if ev == event:
            calls.append((a, k))
        return original(ev, *a, **k)

    ds.broker.publish = spy
    return calls


class TestConstruction:
    def test_requires_anndata(self):
        with pytest.raises(TypeError, match="AnnData"):
            SCLabDataset(pd.DataFrame())

    def test_copy_true_isolates_input(self):
        a = make_adata()
        d = SCLabDataset(a)
        assert d.adata is not a
        assert "counts" not in a.layers  # original not modified
        d.adata.obs["new"] = 1
        assert "new" not in a.obs

    def test_copy_false_shares_object(self):
        a = make_adata()
        d = SCLabDataset(a, copy=False)
        assert d.adata is a and "counts" in a.layers

    def test_counts_layer_created_from_X_and_not_overwritten(self):
        a = make_adata()
        d = SCLabDataset(a)
        assert (d.adata.layers["counts"] != a.X).nnz == 0
        # modifying X afterwards must not alter the stored counts
        d.adata.X = d.adata.X * 2
        assert (d.adata.layers["counts"] != d.adata.X).nnz > 0

        b = make_adata()
        b.layers["counts"] = b.X * 7
        d2 = SCLabDataset(b)
        assert (d2.adata.layers["counts"] != b.layers["counts"]).nnz == 0

    def test_custom_counts_layer_name_and_dataset_name(self):
        d = SCLabDataset(make_adata(), name="mine", counts_layer="raw")
        assert (
            d.name == "mine"
            and "raw" in d.adata.layers
            and "counts" not in d.adata.layers
        )

    def test_shared_broker_registers_events(self):
        from sclab.event import EventBroker

        broker = EventBroker()
        d = SCLabDataset(make_adata(), broker=broker)
        assert d.broker is broker
        assert set(SCLabDataset.events) <= broker.available_events


class TestMetadata:
    def test_column_filtering_and_is_selected(self, ds):
        md = ds.metadata
        assert "many_str" not in md.columns  # > 10 unique strings dropped
        assert isinstance(md["few_str"].dtype, pd.CategoricalDtype)  # <= 10 -> category
        assert set(md.columns) == {"num", "cat", "few_str", "flag", "is_selected"}
        assert md["is_selected"].isna().all()
        assert md["is_selected"].dtype == "boolean"
        # source obs is not modified by the conversion
        assert ds.adata.obs["few_str"].dtype == object
        assert "many_str" in ds.adata.obs

    def test_is_selected_reflects_selection(self, ds):
        ds.select_rows(pd.Index(["cell1", "cell3"]))
        md = ds.metadata
        assert md["is_selected"].dtype == bool
        assert md.index[md["is_selected"]].tolist() == ["cell1", "cell3"]

    def test_row_names(self, ds):
        assert ds.row_names.tolist() == [f"cell{i}" for i in range(12)]


class TestDataDict:
    def test_data_dict_contents(self, ds):
        dd = ds.data_dict
        assert list(dd) == ["metadata", "X_emb"]
        assert dd["metadata"].columns.tolist() == ["num"]
        assert all(pd.api.types.is_numeric_dtype(t) for t in dd["metadata"].dtypes)
        emb = dd["X_emb"]
        assert emb.columns.tolist() == ["X_EMB 1", "X_EMB 2", "X_EMB 3"]
        assert emb.index.equals(ds.metadata.index)
        np.testing.assert_array_equal(emb.values, ds.adata.obsm["X_emb"])

    def test_update_reflects_new_obsm(self, ds):
        ds.adata.obsm["X_new"] = np.zeros((12, 2))
        assert "X_new" in ds.data_dict
        assert ds.data_dict["X_new"].shape == (12, 2)

    @pytest.mark.parametrize(
        "value,exc,msg",
        [
            (np.zeros((5, 2)), AssertionError, "same length"),
            (np.zeros((12, 2, 2)), AssertionError, "1D or 2D"),
            ("nope", TypeError, "DataFrame or numpy"),
            (pd.DataFrame({"a": range(12)}), AssertionError, "same index"),
        ],
    )
    def test_validate_data_errors(self, ds, value, exc, msg):
        with pytest.raises(exc, match=msg):
            ds._validate_data("k", value)

    def test_validate_data_variants(self, ds):
        none = ds._validate_data("k", None)
        assert none.shape == (12, 0) and none.index.equals(ds.metadata.index)
        one_d = ds._validate_data("k", np.arange(12.0))
        assert one_d.shape == (12, 1) and one_d.columns.tolist() == ["K 1"]
        df = pd.DataFrame({"a": range(12)}, index=ds.metadata.index)
        assert ds._validate_data("k", df) is df

    def test_validate_data_dict_errors(self, ds):
        with pytest.raises(AssertionError, match="dictionary"):
            ds._validate_data_dict([1])
        with pytest.raises(AssertionError, match="keys must be strings"):
            ds._validate_data_dict({1: np.zeros((12, 1))})
        a = pd.DataFrame({"a": range(12)}, index=ds.metadata.index)
        b = a.iloc[::-1]
        with pytest.raises(AssertionError, match="same index"):
            ds._validate_data_dict({"a": a, "b": b})

    def test_data_dict_is_read_only(self, ds):
        with pytest.raises(AttributeError):
            ds.data_dict = {"emb2": np.zeros((12, 2))}
        assert list(ds.data_dict) == ["metadata", "X_emb"]

    def test_metadata_is_read_only(self, ds):
        new = ds.adata.obs.copy()
        new["extra"] = 1
        with pytest.raises(AttributeError):
            ds.metadata = new
        assert "extra" not in ds.metadata
        assert "extra" not in ds.adata.obs

    def test_update_data_dict_publishes_current_data_dict(self, ds):
        calls = record(ds, "dset_data_dict_change")
        n_exc = len(ds.broker.exceptions_log)
        ds.adata.obsm["X_new"] = np.zeros((12, 2))

        ds.update_data_dict()

        assert len(calls) == 1
        (args, kwargs) = calls[0]
        assert kwargs == {}
        (published,) = args
        assert list(published) == ["metadata", "X_emb", "X_new"]
        assert published["X_new"].shape == (12, 2)
        assert ds.broker.exceptions_log[n_exc:] == []


class TestDataKeySelection:
    def test_default_data_is_empty_frame(self, ds):
        assert ds.data.shape == (12, 0)

    def test_select_and_reset(self, ds):
        calls = record(ds, "dset_data_key_selection_change")
        ds.select_data_key("X_emb")
        assert ds.data.shape == (12, 3)
        assert len(calls) == 1 and calls[0][0][0].shape == (12, 3)
        ds.reset_data_key()
        assert ds.data.shape == (12, 0) and len(calls) == 2

    def test_unknown_key(self, ds):
        with pytest.raises(ValueError, match="not found"):
            ds.select_data_key("nope")
        assert ds._selected_data_key is None

    def test_ctrl_callback(self, ds):
        ds.ctrl_data_key_change_callback("X_emb")
        assert ds._selected_data_key == "X_emb"
        ds.ctrl_data_key_change_callback(None)
        assert ds._selected_data_key is None


class TestRowSelection:
    def test_default_selection_is_empty(self, ds):
        assert len(ds.selected_rows) == 0 and ds.selected_rows.name == "selected_rows"
        assert not ds.selected_rows_mask.any()

    def test_select_union_and_deselect(self, ds):
        ds.select_rows(pd.Index(["cell0", "cell1"]))
        ds.select_rows(pd.Index(["cell1", "cell2"]))
        assert sorted(ds.selected_rows) == ["cell0", "cell1", "cell2"]
        assert ds.selected_rows_mask.sum() == 3
        assert ds.selected_rows_mask[:3].all()
        ds.deselect_rows(pd.Index(["cell0"]))
        assert sorted(ds.selected_rows) == ["cell1", "cell2"]

    def test_selected_slices(self, ds):
        ds.select_data_key("X_emb")
        ds.select_rows(pd.Index(["cell4", "cell5"]))
        assert ds.selected_rows_data.shape == (2, 3)
        assert ds.selected_rows_metadata.index.tolist() == ["cell4", "cell5"]
        np.testing.assert_array_equal(
            ds.selected_rows_data.values, ds.adata.obsm["X_emb"][[4, 5]]
        )

    def test_invalid_rows_rejected(self, ds):
        with pytest.raises(AssertionError, match="invalid values"):
            ds.select_rows(pd.Index(["nope"]))
        with pytest.raises(AssertionError, match="invalid values"):
            ds.deselect_rows(pd.Index(["nope"]))
        with pytest.raises(AssertionError, match="pandas Index"):
            ds.select_rows(["cell0"])

    def test_clear_and_events(self, ds):
        calls = record(ds, "dset_selected_rows_change")
        ds.select_rows(pd.Index(["cell0"]))
        ds.clear_selected_rows()
        assert ds._selected_rows is None
        assert ds.metadata["is_selected"].isna().all()
        assert len(calls) == 2 and calls[1][0][0] is None

    def test_setter_casts_to_index_dtype(self, ds):
        ds.selected_rows = pd.Index(["cell2"], dtype=object)
        assert ds.selected_rows.dtype == ds.metadata.index.dtype

    def test_plotter_and_clear_callbacks(self, ds):
        ds.dplt_selected_points_change_callback(pd.Index(["cell7"]))
        assert ds.selected_rows.tolist() == ["cell7"]
        ds.dspr_clear_selection_click_callback()
        assert ds._selected_rows is None


class TestFilterRows:
    def test_filter_subsets_all_aligned_data(self, ds):
        calls = record(ds, "dset_total_rows_change")
        keep = pd.Index(["cell1", "cell5", "cell9"])
        ds.filter_rows(keep)
        assert ds.adata.obs_names.tolist() == keep.tolist()
        assert ds.adata.layers["counts"].shape[0] == 3
        assert ds.adata.obsm["X_emb"].shape == (3, 3)
        assert ds.data_dict["X_emb"].index.equals(keep)
        assert len(calls) == 1 and calls[0][0][0].shape[0] == 3

    def test_accepts_plain_sequence(self, ds):
        ds.filter_rows(["cell0", "cell2"])
        assert ds.adata.n_obs == 2

    def test_invalid_subset_raises_and_leaves_data(self, ds):
        with pytest.raises(InvalidRowSubset):
            ds.filter_rows(pd.Index(["cell0", "ghost"]))
        assert ds.adata.n_obs == 12

    def test_preserves_requested_row_order(self, ds):
        ds.filter_rows(pd.Index(["cell9", "cell1"]))
        assert ds.adata.obs_names.tolist() == ["cell9", "cell1"]


class TestApplyLabel:
    def test_creates_categorical_column(self, ds):
        calls = record(ds, "dset_metadata_change")
        ds.apply_label(pd.Index(["cell0", "cell1"]), "lab", "grpA")
        col = ds.adata.obs["lab"]
        assert isinstance(col.dtype, pd.CategoricalDtype)
        assert col.cat.categories.tolist() == ["grpA"]
        assert col.loc[["cell0", "cell1"]].tolist() == ["grpA", "grpA"]
        assert col.drop(["cell0", "cell1"]).isna().all()
        assert calls[-1][0][1] == "lab"

    def test_adds_new_labels_to_existing_column(self, ds):
        ds.apply_label(pd.Index(["cell0"]), "lab", "A")
        ds.apply_label(pd.Index(["cell1"]), "lab", "B")
        ds.apply_label(pd.Index(["cell2"]), "lab", "A")
        col = ds.adata.obs["lab"]
        assert col.cat.categories.tolist() == ["A", "B"]
        assert col.loc[["cell0", "cell1", "cell2"]].tolist() == ["A", "B", "A"]

    def test_empty_label_clears_values(self, ds):
        ds.apply_label(pd.Index(["cell0", "cell1"]), "lab", "A")
        ds.apply_label(pd.Index(["cell0"]), "lab", "")
        col = ds.adata.obs["lab"]
        assert pd.isna(col.loc["cell0"]) and col.loc["cell1"] == "A"

    def test_label_appears_in_metadata(self, ds):
        ds.apply_label(pd.Index(["cell0"]), "lab", "A")
        assert "lab" in ds.metadata.columns
