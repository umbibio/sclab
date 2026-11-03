import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import anndata
from anndata import AnnData
from ipywidgets import (
    Checkbox,
    Dropdown,
    FloatRangeSlider,
    IntRangeSlider,
    SelectMultiple,
)
from scipy.sparse import csr_matrix

import sclab
from sclab import SCLabDashboard
from sclab.dataset.processor import Processor


def make_adata(n_obs=40, n_vars=6, batch=True) -> AnnData:
    rng = np.random.default_rng(1)
    X = csr_matrix(rng.poisson(2.0, size=(n_obs, n_vars)).astype(float))
    obs = pd.DataFrame(
        {
            "num_f": np.linspace(0, 10, n_obs),
            "num_i": np.arange(n_obs),
            "cat": pd.Categorical(["a", "b", "c", "d"] * (n_obs // 4)),
            "flag": [True, False] * (n_obs // 2),
        },
        index=[f"c{i:02d}" for i in range(n_obs)],
    )
    if batch:
        obs["batch"] = pd.Categorical(["b1", "b2"] * (n_obs // 2))
    return AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]))


@pytest.fixture
def dash() -> SCLabDashboard:
    return SCLabDashboard(make_adata())


@pytest.fixture
def proc(dash) -> Processor:
    return dash.processor


# --------------------------------------------------------------------------
# construction / batch handling
# --------------------------------------------------------------------------
class TestConstruction:
    def test_batch_key_autodetected(self, proc):
        assert proc.batch_key == "batch"
        assert proc.batch_values == ["b1", "b2"]

    def test_no_batch_column(self):
        p = SCLabDashboard(make_adata(batch=False)).processor
        assert p.batch_key is None and p.batch_values is None
        assert p.make_selectbatch_drowpdown() == {}
        assert p.make_groupbybatch_checkbox() == {}

    def test_explicit_batch_key(self):
        d = SCLabDashboard(make_adata(batch=False), batch_key="cat")
        assert d.processor.batch_key == "cat"
        assert d.processor.batch_values == ["a", "b", "c", "d"]

    def test_batch_controls(self, proc):
        sel = proc.make_selectbatch_drowpdown(description="Reference Batch")
        assert list(sel) == ["reference_batch"]
        assert isinstance(sel["reference_batch"], Dropdown)
        assert set(sel["reference_batch"].options) == {"", "b1", "b2"}  # "" -> None
        assert sel["reference_batch"].value is None
        chk = proc.make_groupbybatch_checkbox()
        assert (
            isinstance(chk["group_by_batch"], Checkbox) and chk["group_by_batch"].value
        )

    def test_builtin_steps_registered(self, proc):
        assert {"filter_rows", "apply_label", "QC", "Preprocess", "PCA"} <= set(
            proc.steps
        )
        assert "Processing" in proc.step_groups


# --------------------------------------------------------------------------
# step registration
# --------------------------------------------------------------------------
class TestAddSteps:
    def test_add_step_func_and_run(self, proc):
        seen = {}

        def my_func(a=0, b=1):
            seen.update(a=a, b=b)

        proc.add_step_func(
            "My Step", my_func, fixed_params={"b": 5}, variable_controls={}
        )
        assert "My Step" in proc.steps
        proc.steps["My Step"].run(a=3)
        assert seen == {"a": 3, "b": 5}
        entry = proc.step_history[-1]
        assert entry["step"] == "My Step" and entry["params"] == {"b": 5, "a": 3}
        assert entry["status"] == "success" and entry["error"] is None
        assert entry["shape_before"] == entry["shape_after"] == [40, 6]

    def test_duplicate_step_rejected(self, proc):
        proc.add_step_func("Dup", lambda: None)
        with pytest.raises(AssertionError, match="already exists"):
            proc.add_step_func("Dup", lambda: None)

    def test_add_steps_instantiates_classes_and_creates_group(self, proc):
        from sclab.dataset.processor.step import ProcessorStepBase

        class Mine(ProcessorStepBase):
            name = "mine"
            description = "Mine Step"

            def __init__(self, parent):
                super().__init__(parent, fixed_params={}, variable_controls={})

            def function(self):
                pass

        proc.add_steps(Mine, step_group_name="Custom")
        assert isinstance(proc.steps["Mine Step"], Mine)
        assert "Custom" in proc.step_groups
        assert "Mine Step" in proc.step_groups["Custom"].titles

        with pytest.raises(AssertionError, match="already exists"):
            proc.add_steps(Mine(proc), step_group_name="Custom")

    def test_add_steps_rejects_non_step_class(self, proc):
        with pytest.raises(AssertionError, match="must be a subclass"):
            proc.add_steps(dict)

    def test_step_added_to_existing_group_is_merged(self, proc):
        from sclab.dataset.processor.step import ProcessorStepBase

        class Extra(ProcessorStepBase):
            name = "extra"
            description = "Extra Step"

            def __init__(self, parent):
                super().__init__(parent, fixed_params={}, variable_controls={})

            def function(self):
                pass

        before = set(proc.step_groups["Processing"].titles)
        proc.add_steps([Extra])
        after = set(proc.step_groups["Processing"].titles)
        assert after == before | {"Extra Step"}


# --------------------------------------------------------------------------
# step run behaviour (success / failure bookkeeping)
# --------------------------------------------------------------------------
class TestStepRun:
    def test_failure_is_logged_not_raised_and_recorded_as_failed(self, proc, dash):
        def boom():
            raise RuntimeError("kaboom")

        proc.add_step_func("Boom", boom)
        events = []
        original = dash.broker.publish

        def spy(ev, *a, **k):
            events.append((ev, k))
            return original(ev, *a, **k)

        dash.broker.publish = spy
        proc.steps["Boom"].run()
        (entry,) = proc.step_history
        assert entry["step"] == "Boom" and entry["status"] == "failed"
        assert entry["error"] == "RuntimeError: kaboom"
        assert "kaboom" in dash.broker.exceptions_log[-1]
        ended = [k for ev, k in events if ev.endswith("_ended")]
        assert ended and ended[-1]["status"] == "failed"
        assert isinstance(ended[-1]["error"], RuntimeError)


# --------------------------------------------------------------------------
# step history
# --------------------------------------------------------------------------
def label_step(proc, rows, column="lab", label="x"):
    proc.steps["apply_label"].run(index=pd.Index(rows), column=column, label=label)


def steps_and_params(proc) -> list[tuple]:
    return [(e["step"], e["params"]) for e in proc.step_history]


class TestStepHistory:
    def test_pd_index_params_are_stored_as_lists(self, proc):
        label_step(proc, ["c00", "c01"])
        (entry,) = proc.step_history
        assert entry["step"] == "apply_label"
        assert entry["params"] == {
            "index": ["c00", "c01"],
            "column": "lab",
            "label": "x",
        }
        json.dumps(proc.step_history)  # serialisable

    def test_hash_is_deterministic_and_history_sensitive(self, proc):
        h0 = proc.step_history_hash
        label_step(proc, ["c00"])
        h1 = proc.step_history_hash
        label_step(proc, ["c01"], label="y")
        h2 = proc.step_history_hash
        assert len({h0, h1, h2}) == 3 and len(h1) == 64
        assert proc._get_step_history_hash(proc.step_history[:1]) == h1

    @pytest.mark.parametrize("fmt", ["json", "pickle"])
    def test_save_load_apply_roundtrip(self, tmp_path, fmt):
        d1 = SCLabDashboard(make_adata())
        label_step(d1.processor, ["c00", "c01", "c02"], "lab", "A")
        d1.processor.steps["filter_rows"].run(index=pd.Index(["c00", "c01", "c05"]))
        path = tmp_path / f"hist.{fmt}"
        d1.processor.save_step_history(path)

        d2 = SCLabDashboard(make_adata())
        d2.processor.load_step_history(path)
        d2.processor.apply_step_history()

        assert steps_and_params(d2.processor) == steps_and_params(d1.processor)
        assert d2.dataset.adata.obs_names.tolist() == ["c00", "c01", "c05"]
        assert d2.dataset.adata.obs["lab"].loc[["c00", "c01"]].tolist() == ["A", "A"]
        assert d2.processor.step_history_hash == d1.processor.step_history_hash

    def test_apply_only_runs_new_steps(self, tmp_path):
        d1 = SCLabDashboard(make_adata())
        label_step(d1.processor, ["c00"], "lab", "A")
        label_step(d1.processor, ["c01"], "lab", "B")
        p = tmp_path / "h.json"
        d1.processor.save_step_history(p)

        d2 = SCLabDashboard(make_adata())
        label_step(d2.processor, ["c00"], "lab", "A")  # same first step already applied
        d2.processor.load_step_history(p)
        d2.processor.apply_step_history()
        assert len(d2.processor.step_history) == 2
        col = d2.dataset.adata.obs["lab"]
        assert col.loc["c00"] == "A" and col.loc["c01"] == "B"

    def test_apply_on_matching_prefix_with_tuple_params_from_json(self, tmp_path):
        def make_dash():
            d = SCLabDashboard(make_adata())
            d.processor.add_step_func("Dims", lambda dims: None)
            d.processor.steps["Dims"].run(dims=(1, 10))  # JSON loads this as a list
            return d

        d1 = make_dash()
        label_step(d1.processor, ["c01"], "lab", "B")
        p = tmp_path / "h.json"
        d1.processor.save_step_history(p)

        d2 = make_dash()
        d2.processor.load_step_history(p)
        d2.processor.apply_step_history()
        assert len(d2.processor.step_history) == 2
        assert d2.dataset.adata.obs["lab"].loc["c01"] == "B"

    def test_apply_detects_mismatch_and_missing_history(self, tmp_path):
        d1 = SCLabDashboard(make_adata())
        label_step(d1.processor, ["c00"], "lab", "A")
        p = tmp_path / "h.json"
        d1.processor.save_step_history(p)

        d2 = SCLabDashboard(make_adata())
        with pytest.raises(AssertionError, match="No step history loaded"):
            d2.processor.apply_step_history()
        label_step(d2.processor, ["c00"], "other", "Z")  # diverging history
        d2.processor.load_step_history(p)
        with pytest.raises(AssertionError, match="mismatch"):
            d2.processor.apply_step_history()

        d3 = SCLabDashboard(make_adata())
        label_step(d3.processor, ["c00"], "lab", "A")
        label_step(d3.processor, ["c01"], "lab", "A")
        d3.processor.load_step_history(p)  # loaded history is shorter
        with pytest.raises(AssertionError, match="mismatch"):
            d3.processor.apply_step_history()

    def test_format_inference_and_errors(self, proc, tmp_path):
        label_step(proc, ["c00"])
        proc.save_step_history(tmp_path / "a.json")
        proc.save_step_history(tmp_path / "b.pickle", format="pickle")
        assert json.loads((tmp_path / "a.json").read_text()) == proc.step_history
        assert pickle.loads((tmp_path / "b.pickle").read_bytes()) == proc.step_history
        with pytest.raises(ValueError, match="Unsupported format"):
            proc.save_step_history(tmp_path / "c.txt")
        with pytest.raises(ValueError, match="Unsupported format"):
            proc.load_step_history(tmp_path / "c.txt")

    def test_print_step_history(self, proc, capsys):
        label_step(proc, ["c00", "c01"], "lab", "A")
        proc.print_step_history()
        out = capsys.readouterr().out
        assert "apply_label(index=(c00, c01), column=lab, label=A)" in out
        proc.print_step_history(with_hash=True)
        out = capsys.readouterr().out
        assert proc.step_history_hash[:8] in out

    def test_entry_schema(self, proc):
        label_step(proc, ["c00"])
        entry = proc.step_history[-1]
        assert set(entry) == {
            "step",
            "params",
            "status",
            "error",
            "time",
            "shape_before",
            "shape_after",
            "versions",
        }
        assert datetime.fromisoformat(entry["time"]).tzinfo is not None
        assert set(entry["versions"]) == {"sclab", "scanpy", "anndata"}
        assert entry["versions"]["sclab"] == sclab.__version__

    def test_shapes_and_filled_in_defaults(self, proc):
        from sclab.dataset.processor.step import ProcessorStepBase

        class Implicit(ProcessorStepBase):
            name = "implicit"
            description = "Implicit"

            def __init__(self, parent):
                super().__init__(parent, fixed_params={}, variable_controls={})

            def resolved_params(self, params):
                return {**super().resolved_params(params), "batch_key": "batch"}

            def function(self, n: int, dims: tuple = (1, 10)):
                self.parent.dataset.filter_rows(self.parent.dataset.row_names[:n])

        proc.add_steps(Implicit)
        proc.steps["Implicit"].run(n=30)
        entry = proc.step_history[-1]
        assert entry["params"] == {"n": 30, "dims": [1, 10], "batch_key": "batch"}
        assert entry["shape_before"] == [40, 6] and entry["shape_after"] == [30, 6]

    def test_replay_passes_only_accepted_params(self, tmp_path):
        def make_dash():
            d = SCLabDashboard(make_adata())
            d.processor.add_step_func(
                "Take",
                lambda n: d.processor.filter_rows(d.dataset.row_names[:n].tolist()),
            )
            return d

        d1 = make_dash()
        d1.processor.append_to_step_history("Take", {"n": 7, "implicit": "x"})
        p = tmp_path / "h.json"
        d1.processor.save_step_history(p)

        d2 = make_dash()
        d2.processor.load_step_history(p)
        d2.processor.apply_step_history()  # would raise TypeError on `implicit`
        assert d2.dataset.adata.n_obs == 7

    def test_failed_entries_are_skipped_on_replay(self, tmp_path):
        d1 = SCLabDashboard(make_adata())
        d1.processor.add_step_func("Boom", lambda: 1 / 0)
        d1.processor.steps["Boom"].run()
        label_step(d1.processor, ["c00"], "lab", "A")
        assert [e["status"] for e in d1.processor.step_history] == [
            "failed",
            "success",
        ]
        p = tmp_path / "h.json"
        d1.processor.save_step_history(p)

        d2 = SCLabDashboard(make_adata())
        d2.processor.load_step_history(p)
        d2.processor.apply_step_history()
        assert steps_and_params(d2.processor) == steps_and_params(d1.processor)[1:]
        assert d2.processor.step_history_hash == d1.processor.step_history_hash

    def test_old_format_file_is_accepted(self, tmp_path):
        old = [
            {
                "step_description": "apply_label",
                "params": {"index": ["c00", "c01"], "column": "lab", "label": "A"},
            },
            {"step_description": "filter_rows", "params": {"index": ["c00", "c01"]}},
        ]
        p = tmp_path / "old.json"
        p.write_text(json.dumps(old))

        d = SCLabDashboard(make_adata())
        d.processor.load_step_history(p)
        d.processor.apply_step_history()
        assert d.dataset.adata.obs_names.tolist() == ["c00", "c01"]
        assert [e["step"] for e in d.processor.step_history] == [
            "apply_label",
            "filter_rows",
        ]

    def test_history_survives_h5ad_round_trip(self, tmp_path):
        d1 = SCLabDashboard(make_adata())
        rows = [f"c{i:02d}" for i in range(36)]
        long_rows = rows * 2  # longer than the uns inline limit
        label_step(d1.processor, rows[:2], "lab", "A")
        d1.processor.append_to_step_history("Fake", {"index": long_rows})
        d1.processor.steps["filter_rows"].run(index=pd.Index(rows))

        stored = json.loads(d1.dataset.adata.uns["sclab"]["history"])
        assert stored[0] == d1.processor.step_history[0]  # short lists inline
        assert stored[1]["params"]["index"] == {
            "__summary__": "list",
            "len": 72,
            "sha256": stored[1]["params"]["index"]["sha256"],
        }
        assert stored[2]["params"]["index"] == rows  # 36 <= limit

        path = tmp_path / "a.h5ad"
        d1.dataset.adata.write_h5ad(path)
        adata = anndata.read_h5ad(path)
        assert json.loads(adata.uns["sclab"]["history"]) == stored

        d2 = SCLabDashboard(adata)  # picks the history up and shows it
        assert d2.processor.step_history == stored
        assert d2.processor.step_history_hash == d1.processor.step_history_hash
        assert "Fake" in d2.processor.history_table.value

        # a full history file matches the summarized prefix; summaries can't replay
        p = tmp_path / "h.json"
        d1.processor.save_step_history(p)
        d2.processor.load_step_history(p)
        d2.processor.apply_step_history()  # nothing new: no error
        d3 = SCLabDashboard(make_adata())
        d3.processor.add_step_func("Fake", lambda index: None)
        d3.processor._loaded_step_history = stored
        with pytest.raises(ValueError, match="summarized"):
            d3.processor.apply_step_history()

    def test_history_table_lists_steps(self, dash, proc):
        assert "No steps" in proc.history_table.value
        assert proc.history_panel in dash.main_content.children
        label_step(proc, ["c00"], "lab", "A")
        proc.add_step_func("Boom", lambda: 1 / 0)
        proc.steps["Boom"].run()
        table = proc.history_table.value
        assert table.count("<tr") == 3  # header + 2 steps
        assert "apply_label" in table and "label=A" in table
        assert "failed: ZeroDivisionError" in table and "40 x 6" in table


# --------------------------------------------------------------------------
# selection controls
# --------------------------------------------------------------------------
class TestSelectionControls:
    def test_control_types_follow_dtypes(self, proc):
        c = proc.selection_controls_dict
        assert set(c) == {"num_f", "num_i", "cat", "flag", "batch"}
        assert isinstance(c["num_f"], FloatRangeSlider)
        assert isinstance(c["num_i"], IntRangeSlider)
        assert isinstance(c["cat"], SelectMultiple)
        assert list(c["cat"].options) == ["a", "b", "c", "d"]
        assert isinstance(c["flag"], SelectMultiple)
        assert list(c["flag"].options) == [True, False]
        assert c["num_i"].min == 0 and c["num_i"].max == 39
        # initially no selection: range sliders collapsed, selectors empty
        assert c["num_i"].value == (0, 0) and c["cat"].value == ()

    def test_selection_values_property(self, proc):
        proc.selection_controls_dict["cat"].value = ("a",)
        assert proc.selection_values["cat"] == ("a",)

    def test_categorical_selection_selects_matching_rows(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a", "c")
        sel = dash.dataset.selected_rows
        assert sel.tolist() == [f"c{i:02d}" for i in range(40) if i % 4 in (0, 2)]

    def test_numeric_range_is_inclusive(self, proc, dash):
        proc.selection_controls_dict["num_i"].value = (5, 9)
        assert dash.dataset.selected_rows.tolist() == [
            f"c{i:02d}" for i in range(5, 10)
        ]

    def test_bool_selection(self, proc, dash):
        proc.selection_controls_dict["flag"].value = (True,)
        assert len(dash.dataset.selected_rows) == 20
        assert dash.dataset.metadata.loc[dash.dataset.selected_rows, "flag"].all()

    def test_multiple_controls_intersect(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a", "b")
        proc.selection_controls_dict["num_i"].value = (0, 5)
        assert dash.dataset.selected_rows.tolist() == [
            f"c{i:02d}" for i in (0, 1, 4, 5)
        ]

    def test_disjoint_controls_give_empty_selection(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a",)  # i % 4 == 0
        proc.selection_controls_dict["num_i"].value = (1, 3)  # i in 1..3
        assert len(dash.dataset.selected_rows) == 0
        assert (
            dash.dataset._selected_rows is not None
        )  # empty selection != no selection

    def test_no_active_control_means_no_selection(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a",)
        proc.selection_controls_dict["cat"].value = ()
        assert dash.dataset._selected_rows is None

    def test_empty_later_subset_empties_selection(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a",)
        assert len(dash.dataset.selected_rows) == 10
        # num_f takes values k * 10/39 = 0, 0.256, ...: nothing lies in (0.05, 0.2)
        proc.selection_controls_dict["num_f"].value = (0.05, 0.2)
        assert len(dash.dataset.selected_rows) == 0

    def test_empty_first_subset_empties_selection(self, proc, dash):
        # num_f precedes cat in control order: an empty earlier result must stay empty
        proc.selection_controls_dict["num_f"].value = (0.05, 0.2)
        proc.selection_controls_dict["cat"].value = ("a",)
        assert len(dash.dataset.selected_rows) == 0
        assert dash.dataset._selected_rows is not None

    def test_clear_selection_resets_controls(self, proc, dash):
        c = proc.selection_controls_dict
        c["cat"].value = ("a",)
        c["num_i"].value = (3, 8)
        proc.dspr_clear_selection_click_callback()
        assert c["cat"].value == () and c["num_i"].value == (0, 0)
        assert dash.dataset._selected_rows is None

    def test_plotter_selection_resets_controls(self, proc):
        c = proc.selection_controls_dict
        c["cat"].value = ("b",)
        proc.dplt_selected_points_change_callback(pd.Index(["c00"]))
        assert c["cat"].value == ()

    def test_keep_selected_filters_rows_and_records_history(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a",)
        proc.dspr_keep_selected_click_callback()
        assert dash.dataset.adata.n_obs == 10
        assert set(dash.dataset.adata.obs["cat"]) == {"a"}
        assert proc.step_history[-1]["step"] == "filter_rows"
        # slider bounds refreshed after the row change
        assert proc.selection_controls_dict["num_i"].max == 36

    def test_drop_selected_removes_rows(self, proc, dash):
        proc.selection_controls_dict["cat"].value = ("a",)
        proc.dspr_drop_selected_click_callback()
        assert dash.dataset.adata.n_obs == 30
        assert "a" not in set(dash.dataset.adata.obs["cat"])

    def test_apply_label_via_controls(self, proc, dash):
        proc.selection_controls_dict["num_i"].value = (0, 3)
        lab = proc.selection_labeling_controls_dict
        lab["create_new_key_checkbox"].value = True
        lab["new_medatadata_key"].value = "mylabel"
        lab["new_label"].value = "first"
        assert lab["apply_button"].disabled is False
        proc.dspr_apply_label_click_callback()
        col = dash.dataset.adata.obs["mylabel"]
        assert col.loc[["c00", "c01", "c02", "c03"]].tolist() == ["first"] * 4
        assert col.drop(["c00", "c01", "c02", "c03"]).isna().all()
        assert "mylabel" in lab["existing_metadata_key"].options

    def test_apply_label_button_disabled_without_key(self, proc):
        lab = proc.selection_labeling_controls_dict
        assert lab["apply_button"].disabled is True
        lab["create_new_key_checkbox"].value = True
        lab["new_medatadata_key"].value = "k"
        assert lab["apply_button"].disabled is False
        lab["new_medatadata_key"].value = ""
        assert lab["apply_button"].disabled is True

    def test_new_obs_column_gets_a_control(self, proc, dash):
        dash.dataset.adata.obs["extra"] = np.arange(40) * 0.5
        proc._make_selection_controls()
        assert isinstance(proc.selection_controls_dict["extra"], FloatRangeSlider)

    def test_datetime_columns_get_no_control(self, proc, dash):
        dash.dataset.adata.obs["dt"] = pd.date_range("2020", periods=40)
        # datetimes are filtered out of metadata, so they never get a control
        proc._make_selection_controls()
        assert "dt" not in proc.selection_controls_dict

    def test_removed_obs_column_removes_control(self, proc, dash):
        del dash.dataset.adata.obs["num_i"]
        proc._make_selection_controls()
        assert "num_i" not in proc.selection_controls_dict
        assert all(c.description != "num_i" for c in proc.selection_controls_list)

    def test_removed_column_keeps_other_filters(self, proc, dash):
        c = proc.selection_controls_dict
        c["num_i"].value = (5, 9)
        c["cat"].value = ("a",)
        n = len(dash.broker.exceptions_log)
        proc.remove_column_selection_control("num_i")
        assert "num_i" not in c
        assert dash.dataset.selected_rows.tolist() == [
            f"c{i:02d}" for i in range(0, 40, 4)
        ]
        assert dash.broker.exceptions_log[n:] == []

    def test_removed_only_active_column_clears_selection(self, proc, dash):
        proc.selection_controls_dict["num_i"].value = (5, 9)
        n = len(dash.broker.exceptions_log)
        del dash.dataset.adata.obs["num_i"]
        proc._make_selection_controls()
        assert dash.dataset._selected_rows is None
        assert dash.broker.exceptions_log[n:] == []

    def test_active_filters_survive_unrelated_metadata_change(self, proc, dash):
        c = proc.selection_controls_dict
        c["cat"].value = ("a", "b")
        c["num_i"].value = (2, 20)
        c["num_f"].value = (1.0, 8.0)
        expected = dash.dataset.selected_rows.copy()
        n = len(dash.broker.exceptions_log)
        dash.dataset.adata.obs["extra"] = np.arange(40) * 0.5
        proc._make_selection_controls()
        assert c["cat"].value == ("a", "b")
        assert c["num_i"].value == (2, 20)
        assert c["num_f"].value == (1.0, 8.0)
        assert dash.dataset.selected_rows.equals(expected)
        assert dash.broker.exceptions_log[n:] == []

    def test_inactive_controls_stay_inactive(self, proc, dash):
        c = proc.selection_controls_dict
        dash.dataset.adata.obs["num_i"] = np.arange(40) + 100
        proc._make_selection_controls()
        assert c["cat"].value == () and c["flag"].value == ()
        assert c["num_i"].value == (100, 100)
        assert c["num_f"].value == (c["num_f"].min, c["num_f"].min)
        assert dash.dataset._selected_rows is None

    def test_clipped_numeric_range_recomputes_selection(self, proc, dash):
        proc.selection_controls_dict["num_i"].value = (5, 30)
        n = len(dash.broker.exceptions_log)
        dash.dataset.adata.obs["num_i"] = np.arange(40).clip(max=20)
        proc._make_selection_controls()
        assert proc.selection_controls_dict["num_i"].value == (5, 20)
        # values >= 5 after clipping: rows 5..39 (rows 20..39 all equal 20)
        assert dash.dataset.selected_rows.tolist() == [
            f"c{i:02d}" for i in range(5, 40)
        ]
        assert dash.broker.exceptions_log[n:] == []

    def test_removed_category_is_dropped_from_filter(self, proc, dash):
        c = proc.selection_controls_dict
        c["cat"].value = ("a", "b")
        n = len(dash.broker.exceptions_log)
        obs = dash.dataset.adata.obs
        obs["cat"] = pd.Categorical(
            obs["cat"].astype(str).replace({"b": "c"}), categories=["a", "c", "d"]
        )
        proc._make_selection_controls()
        assert list(c["cat"].options) == ["a", "c", "d"]
        assert c["cat"].value == ("a",)
        assert dash.dataset.selected_rows.tolist() == [
            f"c{i:02d}" for i in range(0, 40, 4)
        ]
        assert dash.broker.exceptions_log[n:] == []

    def test_all_selected_categories_removed_clears_selection(self, proc, dash):
        c = proc.selection_controls_dict
        c["cat"].value = ("b",)
        dash.dataset.adata.obs["cat"] = pd.Categorical(["a", "c"] * 20)
        proc._make_selection_controls()
        assert c["cat"].value == ()
        assert dash.dataset._selected_rows is None


# --------------------------------------------------------------------------
# callbacks that refresh step controls
# --------------------------------------------------------------------------
class TestControlRefreshCallbacks:
    def test_metadata_change_refreshes_group_dropdowns(self, proc, dash):
        control = proc.steps["Differential Expression"].variable_controls["groupby"]
        assert "new_cat" not in control.options
        dash.dataset.adata.obs["new_cat"] = pd.Categorical(["u", "v"] * 20)
        proc.dset_metadata_change_callback()
        assert "new_cat" in control.options
        labelkey = proc.selection_labeling_controls_dict["existing_metadata_key"]
        assert "new_cat" in labelkey.options

    def test_metadata_change_refreshes_axis_dropdowns_and_keeps_valid_value(
        self, proc, dash
    ):
        control = proc.steps["Gene Expression"].variable_controls["time_key"]
        control.value = "num_f"
        dash.dataset.adata.obs["extra_num"] = np.arange(40.0)
        proc.dset_metadata_change_callback()
        assert "extra_num" in control.options
        assert control.value == "num_f"
        del dash.dataset.adata.obs["num_f"]
        proc.dset_metadata_change_callback()
        assert "num_f" not in control.options and control.value is None

    def test_layers_change_refreshes_layer_dropdowns(self, proc, dash):
        control = proc.steps["Differential Expression"].variable_controls["layer"]
        dash.dataset.adata.layers["fancy"] = dash.dataset.adata.X.copy()
        proc.dset_anndata_layers_change_callback()
        assert "fancy" in control.options

    def test_data_dict_change_selects_x_pca_by_default(self, proc, dash):
        control = proc.steps["Integration"].variable_controls["use_rep"]
        dash.dataset.adata.obsm["X_pca"] = np.zeros((40, 2))
        proc.dset_data_dict_change_callback()
        assert control.value == "X_pca"
        assert {"counts", "X_pca"} <= set(control.options)

    def test_second_data_dict_change_keeps_neighbors_use_rep(self, proc, dash):
        dash.dataset.adata.obsm["X_pca"] = np.zeros((40, 4))
        n = len(dash.broker.exceptions_log)
        dash.broker.publish("dset_data_dict_change", dash.dataset.data_dict)
        # a new representation (e.g. produced by Integration) changes the options
        dash.dataset.adata.obsm["X_pca_harmony"] = np.zeros((40, 4))
        dash.broker.publish("dset_data_dict_change", dash.dataset.data_dict)
        control = proc.steps["Neighbors"].variable_controls["use_rep"]
        assert [e for e in dash.broker.exceptions_log[n:] if "KeyError" in e] == []
        assert control.value == "X_pca"


# --------------------------------------------------------------------------
# category dropdown keeps its value when metadata changes
# --------------------------------------------------------------------------
class TestCategoryDropdownKeepsValue:
    @pytest.mark.parametrize("picked", ["cat", "flag", "batch"])
    def test_value_survives_new_column(self, proc, dash, picked):
        dropdown = proc.visible_category_dropdown
        dropdown.value = picked
        dash.dataset.adata.obs["new_cat"] = pd.Categorical(["u", "v"] * 20)
        dash.broker.publish("dset_metadata_change", dash.dataset.metadata, "new_cat")
        assert "new_cat" in dropdown.options
        assert dropdown.value == picked
        assert dropdown.options[dropdown.index] == picked

    def test_frontend_gets_a_single_update(self, proc, dash):
        dropdown = proc.visible_category_dropdown
        dropdown.value = "batch"
        sent = []
        dropdown.send_state = lambda key=None: sent.append(key)
        dash.dataset.adata.obs["new_cat"] = pd.Categorical(["u", "v"] * 20)
        dash.broker.publish("dset_metadata_change", dash.dataset.metadata, "new_cat")
        assert len(sent) == 1  # no transient reset to the first option

    def test_removed_category_falls_back_to_first_option(self, proc, dash):
        dropdown = proc.visible_category_dropdown
        dropdown.value = "batch"
        del dash.dataset.adata.obs["batch"]
        dash.broker.publish("dset_metadata_change", dash.dataset.metadata, None)
        assert dropdown.value == "cat"
