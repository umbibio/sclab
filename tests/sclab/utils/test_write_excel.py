"""Tests for sclab.utils.write_excel.

No Excel engine (xlsxwriter/openpyxl) is installed in the test environment and
``write_excel`` needs the xlsxwriter API (``add_format``, ``set_column``,
``conditional_format``...). The orchestration is therefore tested against a
recording fake of ``pd.ExcelWriter``; one end-to-end test runs only when
xlsxwriter is importable.
"""

import os

import numpy as np
import pandas as pd
import pytest

from sclab.utils import _write_excel as we
from sclab.utils import write_excel


# --------------------------------------------------------------------------
# Recording fakes
# --------------------------------------------------------------------------
class FakeWorksheet:
    def __init__(self):
        self.columns = {}  # idx -> (width, fmt)
        self.conditional = []  # (first_row, col, last_row, col, options)
        self.autofilter_args = None
        self.writes = []  # (row, col, value, fmt)

    def set_column(self, first, last, width, fmt):
        assert first == last
        self.columns[first] = (width, fmt)

    def conditional_format(self, r0, c0, r1, c1, options):
        self.conditional.append((r0, c0, r1, c1, options))

    def autofilter(self, *args):
        self.autofilter_args = args

    def write(self, row, col, value, fmt=None):
        self.writes.append((row, col, value, fmt))


class FakeWorkbook:
    def add_format(self, props=None):
        return dict(props or {})


class FakeWriter:
    last = None

    def __init__(self, path, *a, **k):
        self.path = path
        self.book = FakeWorkbook()
        self.sheets = {}
        self.frames = {}
        FakeWriter.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_excel(monkeypatch):
    def to_excel(
        self, writer, sheet_name="Sheet1", index=True, startrow=0, header=True
    ):
        writer.frames[sheet_name] = self.copy()
        writer.sheets[sheet_name] = FakeWorksheet()

    monkeypatch.setattr(pd, "ExcelWriter", FakeWriter)
    monkeypatch.setattr(pd.DataFrame, "to_excel", to_excel)
    FakeWriter.last = None
    return FakeWriter


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "name": ["alpha", "b", "gamma-long-name"],
            "count": [3, 1, 2],
            "ratio": [0.25, 0.5, 0.125],
            "tiny": [1e-8, 2e-9, 3e-7],
            "flag": [True, False, True],
        },
        index=pd.Index(["g1", "g2", "g3"], name="gene"),
    )


# --------------------------------------------------------------------------
# _guess_column_format / width
# --------------------------------------------------------------------------
class TestGuessFormat:
    @pytest.mark.parametrize(
        "values,expected",
        [
            (pd.Series([True, False]), "boolean"),
            (pd.Series([1, 2, 3]), "nodecimal"),
            (pd.Series([1.0, 2.0, 3.0]), "nodecimal"),
            (pd.Series([0.5, 0.25]), "number"),
            (pd.Series([1e7 + 0.5, 2.0]), "scientific"),
            (pd.Series([1e-5, 0.5]), "scientific"),
            (pd.Series([0.5, np.nan]), "number"),
            (pd.Series(["a", "b"]), "text"),
        ],
    )
    def test_guess(self, values, expected):
        assert we._guess_column_format(values) == expected

    def test_boundaries_are_strict(self):
        # 1e-3 is not below the 1e-3 threshold; 1e6 is not above the 1e6 one
        assert we._guess_column_format(pd.Series([1e-3, 0.5])) == "number"
        assert we._guess_column_format(pd.Series([2e6 + 0.5, 1.5])) == "scientific"

    def test_categorical_is_text_and_has_width(self):
        s = pd.Series(pd.Categorical(["aa", "bbbb"]), name="c")
        fmt = we._guess_column_format(s)
        assert fmt == "text"
        assert we._guess_column_width(s, fmt) >= 10

    def test_datetime_column_width(self):
        s = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]), name="when")
        fmt = we._guess_column_format(s)
        assert fmt == "text"
        assert we._guess_column_width(s, fmt) >= 10


class TestGuessWidth:
    def test_rounding_and_clamps(self):
        # width = ceil(len * 1.4), clamped to [10, 75]
        s = pd.Series(["x" * 20], name="c")
        assert we._guess_column_width(s, "text") == 28
        s = pd.Series(["x" * 500], name="c")
        assert we._guess_column_width(s, "text") == 75
        s = pd.Series(["x"], name="c")
        assert we._guess_column_width(s, "text") == 10

    def test_text_uses_longer_of_header_and_values(self):
        s = pd.Series(["ab"], name="a_very_long_column_name")
        assert we._guess_column_width(s, "text") == int(np.ceil(len(s.name) * 1.4))

    def test_numeric_uses_header_only(self):
        s = pd.Series([123456789.0], name="x" * 30)
        for fmt in ["number", "nodecimal", "scientific", "boolean", "number_BlueRed"]:
            assert we._guess_column_width(s, fmt) == 42

    def test_percentage_min_width(self):
        assert we._guess_column_width(pd.Series([0.1], name="p"), "percentage") == 14

    def test_empty_series_uses_header(self):
        s = pd.Series([], dtype=float, name="x" * 20)
        assert we._guess_column_width(s, "text") == 28

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            we._guess_column_width(pd.Series([1.0], name="a"), "wat")

    def test_round_width(self):
        assert we._round_width(5, 6, 10, 75) == 10
        assert we._round_width(10, 6, 10, 75) == 14
        assert we._round_width(100, 6, 10, 75) == 75


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize(
        "fmt",
        [
            "text",
            "bold",
            "number",
            "scientific",
            "percentage",
            "nodecimal",
            "boolean",
            "number_BlueRed",
            "number_GreenBlue|-1,0,1",
            "number_Red_desc",
            "number_Blue|0,5",
        ],
    )
    def test_valid_formats(self, fmt):
        we._validate_column_formats({"c": fmt})

    @pytest.mark.parametrize("fmt", ["bogus", "Number", "number_Pink", "3color", ""])
    def test_invalid_formats(self, fmt):
        with pytest.raises(AssertionError, match="Unknown format"):
            we._validate_column_formats({"c": fmt})

    def test_formats_must_be_dict(self):
        with pytest.raises(AssertionError):
            we._validate_column_formats(["text"])

    def test_widths(self):
        we._validate_column_widths({"a": 10, "b": 12.5})
        for bad in [0, -3, "10", None]:
            with pytest.raises(AssertionError, match="Invalid column width"):
                we._validate_column_widths({"a": bad})
        with pytest.raises(AssertionError):
            we._validate_column_widths([10])

    def test_color_lookup(self):
        assert we._color("Blue") == "#B3CDE3"
        assert we._color("White") == "#F7F7F7"
        with pytest.raises(ValueError, match="Invalid color"):
            we._color("Pink")

    def test_is_writable(self, tmp_path):
        assert we._is_writable(tmp_path / "new.xlsx")  # does not exist yet
        p = tmp_path / "f.xlsx"
        p.write_bytes(b"x")
        assert we._is_writable(p)
        if os.geteuid() != 0:
            p.chmod(0o400)
            try:
                assert we._is_writable(p) is False
            finally:
                p.chmod(0o600)


# --------------------------------------------------------------------------
# conditional format builders
# --------------------------------------------------------------------------
class TestConditionalFormats:
    def test_three_color_default_is_symmetric_about_zero(self):
        s = pd.Series([-2.0, 0.5, 4.0, np.inf, np.nan])
        f = we._make_3_color_scale_conditional_format(s, scale=1.5)
        assert f["type"] == "3_color_scale"
        assert f["max_value"] == pytest.approx(6.0)  # max(|q0|,|q1|)*1.5, inf dropped
        assert f["min_value"] == pytest.approx(-6.0)
        assert f["mid_value"] == 0
        assert (f["min_color"], f["mid_color"], f["max_color"]) == (
            "#B3CDE3",
            "#F7F7F7",
            "#FFB3B3",
        )

    def test_three_color_partial_overrides(self):
        s = pd.Series([1.0, 3.0, 5.0])
        f = we._make_3_color_scale_conditional_format(
            s, min_color="Green", max_color="#123456", min_value=0
        )
        assert f["min_value"] == 0
        assert f["mid_value"] == 3.0  # midpoint of data
        assert f["max_value"] == 5.0
        assert f["min_color"] == "#B3E3B3"
        assert f["max_color"] == "#123456"

    def test_two_color_quantile_defaults_and_overrides(self):
        s = pd.Series(np.arange(101, dtype=float))
        f = we._make_2_color_scale_conditional_format(s)
        assert f["type"] == "2_color_scale"
        assert f["min_value"] == pytest.approx(5.0)
        assert f["max_value"] == pytest.approx(95.0)
        assert f["min_color"] == "#F7F7F7" and f["max_color"] == "#B3E3B3"
        f = we._make_2_color_scale_conditional_format(
            s, min_color="White", max_color="Red", min_value=1, max_value=2
        )
        assert (f["min_value"], f["max_value"]) == (1, 2)
        assert f["max_color"] == "#FFB3B3"

    def test_percentage_and_cell_formats(self):
        p = we._make_percentage_conditional_format()
        assert p["type"] == "data_bar" and (p["min_value"], p["max_value"]) == (0, 1)
        c = we._make_cell_color_conditional_format("==", "TRUE", {"x": 1})
        assert c == {
            "type": "cell",
            "criteria": "==",
            "value": "TRUE",
            "format": {"x": 1},
        }

    def test_write_table_header(self):
        ws = FakeWorksheet()
        we._write_table_header(["a", "b", "c"], ws, {"bold": True})
        assert ws.writes == [
            (0, 0, "a", {"bold": True}),
            (0, 1, "b", {"bold": True}),
            (0, 2, "c", {"bold": True}),
        ]

    def test_formats_dict_has_defaults(self):
        fm = we._get_formats_dict(FakeWorkbook())
        assert fm["number"] == {"num_format": "0.00"}
        assert fm["percentage"] == {"num_format": "0.00%"}
        assert fm["cellGreen"] == {"bg_color": "#B3E3B3"}
        assert fm["header_format"]["bold"] is True
        # unknown keys fall back to a plain format instead of KeyError
        assert fm["number_BlueRed"] == {}

    def test_print_column_info(self, capsys):
        we._print_column_info(0, "S", "col", "text", 12, 2)
        we._print_column_info(1, "S", "col2", "number", 10, 2)
        out = capsys.readouterr().out
        assert "Sheet Name: S" in out and "col2" in out
        assert out.count("-" * 55) == 4


# --------------------------------------------------------------------------
# write_excel orchestration (fake engine)
# --------------------------------------------------------------------------
class TestWriteExcel:
    def test_single_frame_goes_to_sheet1_with_index_reset(
        self, fake_excel, df, tmp_path
    ):
        write_excel(df, tmp_path / "o.xlsx")
        w = fake_excel.last
        assert list(w.sheets) == ["Sheet1"]
        out = w.frames["Sheet1"]
        assert out.columns.tolist() == [
            "gene",
            "name",
            "count",
            "ratio",
            "tiny",
            "flag",
        ]
        assert out["gene"].tolist() == ["g1", "g2", "g3"]
        # caller's frame is not mutated
        assert df.index.name == "gene" and "gene" not in df.columns

    def test_dict_of_frames_gives_one_sheet_each(self, fake_excel, df, tmp_path):
        write_excel({"A": df, "B": df.iloc[:1]}, tmp_path / "o.xlsx")
        w = fake_excel.last
        assert list(w.sheets) == ["A", "B"]
        assert len(w.frames["B"]) == 1

    def test_guessed_formats_and_widths(self, fake_excel, df, tmp_path):
        write_excel(df, tmp_path / "o.xlsx")
        ws = fake_excel.last.sheets["Sheet1"]
        fmts = {i: ws.columns[i][1] for i in ws.columns}
        # columns: gene, name, count, ratio, tiny, flag
        assert fmts[0] == {"font_name": "Calibri", "font_size": 11}  # text
        assert fmts[2] == {"num_format": "0"}  # nodecimal
        assert fmts[3] == {"num_format": "0.00"}  # number
        assert fmts[4] == {"num_format": "0.00E+00"}  # scientific
        assert fmts[5]["font_name"] == "Monospace"  # boolean
        assert ws.columns[1][0] == int(np.ceil(len("gamma-long-name") * 1.4))
        # boolean column gets a cell-colour rule over rows 0..n
        bool_rules = [c for c in ws.conditional if c[1] == 5]
        assert len(bool_rules) == 1
        assert (
            bool_rules[0][4]["type"] == "cell" and bool_rules[0][4]["value"] == "TRUE"
        )
        assert (bool_rules[0][0], bool_rules[0][2]) == (0, 3)

    def test_datetime_and_all_nan_object_columns(self, fake_excel, tmp_path):
        d = pd.DataFrame(
            {
                "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "empty": pd.Series([np.nan, np.nan], dtype=object),
                "mixed": ["a", 12345678901234567890],
            }
        )
        write_excel(d, tmp_path / "o.xlsx")
        ws = fake_excel.last.sheets["Sheet1"]
        assert ws.columns[1][0] == int(np.ceil(len("2020-01-01") * 1.4))
        assert ws.columns[2][0] == 10
        assert ws.columns[3][0] == int(np.ceil(len("12345678901234567890") * 1.4))

    def test_header_and_autofilter(self, fake_excel, df, tmp_path):
        write_excel(df, tmp_path / "o.xlsx")
        ws = fake_excel.last.sheets["Sheet1"]
        assert [w[2] for w in ws.writes] == [
            "gene",
            "name",
            "count",
            "ratio",
            "tiny",
            "flag",
        ]
        assert ws.autofilter_args == (0, 0, 2, 5)

        write_excel(df, tmp_path / "o.xlsx", autofilter=False)
        assert fake_excel.last.sheets["Sheet1"].autofilter_args is None

    def test_explicit_formats_and_widths_override_guessing(
        self, fake_excel, df, tmp_path
    ):
        write_excel(
            df,
            tmp_path / "o.xlsx",
            column_formats={"ratio": "percentage", "count": "bold"},
            column_widths={"ratio": 33, "name": 7},
        )
        ws = fake_excel.last.sheets["Sheet1"]
        assert ws.columns[3] == (33, {"num_format": "0.00%"})
        assert ws.columns[2][1] == {"bold": True}
        assert ws.columns[1][0] == 7
        # percentage -> data bar rule on that column
        assert any(
            c[1] == 3 and c[4]["type"] == "data_bar"
            for c in fake_excel.last.sheets["Sheet1"].conditional
        )

    def test_guess_disabled_uses_defaults(self, fake_excel, df, tmp_path):
        write_excel(df, tmp_path / "o.xlsx", guess_formats=False, guess_widths=False)
        ws = fake_excel.last.sheets["Sheet1"]
        assert all(w == 10 for w, _ in ws.columns.values())
        assert all(f == {} for _, f in ws.columns.values())  # "noformat"
        assert ws.conditional == []

    def test_color_scale_formats_parse_bounds(self, fake_excel, df, tmp_path):
        write_excel(
            df,
            tmp_path / "o.xlsx",
            column_formats={
                "ratio": "number_BlueRed|-1,0,1",
                "tiny": "number_Green_desc|0,2",
            },
        )
        ws = fake_excel.last.sheets["Sheet1"]
        r = next(c[4] for c in ws.conditional if c[1] == 3)
        assert r["type"] == "3_color_scale"
        assert (r["min_value"], r["mid_value"], r["max_value"]) == (-1, 0, 1)
        assert r["min_color"] == we._color("Blue") and r["max_color"] == we._color(
            "Red"
        )
        t = next(c[4] for c in ws.conditional if c[1] == 4)
        assert t["type"] == "2_color_scale"
        assert (t["min_value"], t["max_value"]) == (0, 2)
        # _desc: white at the low end, colour at the high end
        assert t["min_color"] == we._color("White") and t["max_color"] == we._color(
            "Green"
        )

    def test_sortby_orders_rows_and_ignores_unknown_columns(
        self, fake_excel, df, tmp_path
    ):
        write_excel(df, tmp_path / "o.xlsx", sortby="count")
        assert fake_excel.last.frames["Sheet1"]["count"].tolist() == [1, 2, 3]
        write_excel(df, tmp_path / "o.xlsx", sortby=["missing"])
        assert fake_excel.last.frames["Sheet1"]["count"].tolist() == [3, 1, 2]
        write_excel(
            df, tmp_path / "o.xlsx", sortby=["missing", "count"], sort_ascending=True
        )
        assert fake_excel.last.frames["Sheet1"]["count"].tolist() == [1, 2, 3]

    def test_sort_descending(self, fake_excel, df, tmp_path):
        write_excel(df, tmp_path / "o.xlsx", sortby="count", sort_ascending=False)
        assert fake_excel.last.frames["Sheet1"]["count"].tolist() == [3, 2, 1]

    def test_sort_mixed_directions(self, fake_excel, tmp_path):
        d = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]})
        write_excel(
            d, tmp_path / "o.xlsx", sortby=["a", "b"], sort_ascending=[True, False]
        )
        out = fake_excel.last.frames["Sheet1"]
        assert out[["a", "b"]].values.tolist() == [[1, 2], [1, 1], [2, 2], [2, 1]]

    def test_missing_sort_column_keeps_flags_aligned(self, fake_excel, tmp_path):
        d = pd.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 1, 2]})
        write_excel(
            d,
            tmp_path / "o.xlsx",
            sortby=["missing", "a", "b"],
            sort_ascending=[True, False, True],
        )
        out = fake_excel.last.frames["Sheet1"]
        assert out[["a", "b"]].values.tolist() == [[2, 1], [2, 2], [1, 1], [1, 2]]

    def test_sort_ascending_length_mismatch_raises(self, fake_excel, df, tmp_path):
        with pytest.raises(ValueError, match="sort_ascending must be a bool"):
            write_excel(
                df, tmp_path / "o.xlsx", sortby=["count"], sort_ascending=[True, False]
            )

    def test_verbose_prints_table(self, fake_excel, df, tmp_path, capsys):
        write_excel(df, tmp_path / "o.xlsx", verbose=True)
        out = capsys.readouterr().out
        assert "Sheet Name: Sheet1" in out and "scientific" in out

    def test_invalid_arguments_raise_before_writing(self, fake_excel, df, tmp_path):
        with pytest.raises(AssertionError, match="Unknown format"):
            write_excel(df, tmp_path / "o.xlsx", column_formats={"count": "nope"})
        with pytest.raises(AssertionError, match="Invalid column width"):
            write_excel(df, tmp_path / "o.xlsx", column_widths={"count": -1})
        assert fake_excel.last is None  # no writer was opened

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can write read-only files")
    def test_unwritable_file_prints_message_and_returns(
        self, fake_excel, df, tmp_path, capsys
    ):
        p = tmp_path / "locked.xlsx"
        p.write_bytes(b"x")
        p.chmod(0o400)
        try:
            assert write_excel(df, p) is None
        finally:
            p.chmod(0o600)
        assert "not writable" in capsys.readouterr().out
        assert fake_excel.last is None


def test_write_excel_end_to_end_roundtrip(df, tmp_path):
    pytest.importorskip("xlsxwriter")
    pytest.importorskip("openpyxl")
    p = tmp_path / "o.xlsx"
    write_excel({"S": df}, p, sortby="count")
    back = pd.read_excel(p, sheet_name="S")
    assert back["gene"].tolist() == ["g2", "g3", "g1"]
    assert back["count"].tolist() == [1, 2, 3]
