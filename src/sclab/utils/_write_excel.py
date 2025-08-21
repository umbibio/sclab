import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

_column_format_type = Literal[
    "text", "bold", "number", "scientific", "percentage", "nodecimal", "boolean"
]


def write_excel(
    dataframes: pd.DataFrame | dict[str, pd.DataFrame],
    file_path: str | Path,
    sortby: str | list[str] | None = None,
    sort_ascending: bool | list[bool] = None,
    autofilter: bool = True,
    guess_formats: bool = True,
    guess_widths: bool = True,
    column_formats: dict[str, _column_format_type] | None = None,
    column_widths: dict[str, int] | None = None,
    verbose: bool = False,
) -> None:
    """
    Write pandas DataFrame(s) to an Excel file with enhanced formatting and styling.

    This function provides advanced Excel writing capabilities including automatic format detection,
    column width adjustment, conditional formatting, and table headers with autofilter.

    Parameters
    ----------
    dataframes : pandas.DataFrame or dict[str, pandas.DataFrame]
        Single DataFrame or dictionary of DataFrames where keys are sheet names.
    file_path : str or pathlib.Path
        Path where the Excel file will be saved.
    sortby : str or None, optional
        Column name to sort the data by. If specified, applies 3-color scale conditional formatting
        to the sorted column. Default is None.
    sort_ascending : bool, optional
        Sort order when sortby is specified. Default is True.
    autofilter : bool, optional
        Whether to add filter buttons to column headers. Default is True.
    guess_formats : bool, optional
        Whether to automatically detect and apply appropriate formats based on column data types.
        Default is True.
    guess_widths : bool, optional
        Whether to automatically estimate and apply column widths based on column name/contents.
        Default is True.
    column_formats : dict[str, str] or None, optional
        Custom format specifications for columns. Keys are column names, values are format types:
        "text", "bold", "number", "scientific", "percentage", "nodecimal", or "boolean".
        Default is None.
    column_widths : dict[str, int] or None, optional
        Custom width specifications for columns. Keys are column names, values are widths in
        Excel units. Default is None.
    verbose : bool, optional
        Whether to print detailed information about column formatting. Default is False.

    Returns
    -------
    None

    Notes
    -----
    - If a single DataFrame is provided, it will be written to 'Sheet1'
    - Percentage columns get conditional formatting based on their values
    - The function validates file writability and format specifications before proceeding
    - Column formats and widths are either explicitly specified or automatically detected
    """
    if not isinstance(sortby, list):
        sortby = [sortby]

    if sort_ascending is None:
        sort_ascending = [True]

    if not isinstance(sort_ascending, list):
        sort_ascending = [sort_ascending]

    if column_formats is None:
        column_formats = {}
    else:
        _validate_column_formats(column_formats)

    if column_widths is None:
        column_widths = {}
    else:
        _validate_column_widths(column_widths)

    if not isinstance(dataframes, dict):
        dataframes = {"Sheet1": dataframes}

    file_path = Path(file_path)

    if not _is_writable(file_path):
        print(f"File {file_path} is not writable. Please make sure it's not in use.")
        return

    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, df in dataframes.items():
            df = df.reset_index()

            _sortby = [c for c in sortby if c in df.columns]
            _sort_ascending = [a for a in sort_ascending if a]
            if not _sort_ascending:
                _sort_ascending = True
            if _sortby:
                df = df.sort_values(by=_sortby, ascending=_sort_ascending)

            df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=1, header=False
            )
            max_row, max_col = df.shape

            workbook = writer.book
            formats = _get_formats_dict(workbook)

            worksheet = writer.sheets[sheet_name]

            for i, col in enumerate(df.columns):
                if col in column_formats:
                    column_format = column_formats[col]
                elif guess_formats:
                    column_format = _guess_column_format(df[col])
                else:
                    column_format = "noformat"

                if col in column_widths:
                    column_width = column_widths[col]
                elif guess_widths:
                    try:
                        column_width = _guess_column_width(df[col], column_format)
                    except ValueError:
                        column_width = 10
                else:
                    column_width = 10

                fmt = formats[column_format]
                worksheet.set_column(i, i, column_width, fmt)

                if verbose:
                    _print_column_info(
                        i, sheet_name, col, column_format, column_width, max_col
                    )

                if column_format == "percentage":
                    fmt = _make_percentage_conditional_format()
                    worksheet.conditional_format(0, i, max_row, i, fmt)

                if column_format == "boolean":
                    fmt = _make_cell_color_conditional_format(
                        "==", "TRUE", formats["cellGreen"]
                    )
                    worksheet.conditional_format(0, i, max_row, i, fmt)

                # if col in sortby:
                #     fmt = _make_3_color_scale_conditional_format(df[col])
                #     worksheet.conditional_format(0, i, max_row, i, fmt)

                if m := re.match(
                    r"^number_(Blue|Green|Red)(Blue|Green|Red)\|?([\-\+\d.]+)?,?([\-\+\d.]+)?,?([\-\+\d.]+)?$",
                    column_format,
                ):
                    min_color = m.group(1)
                    max_color = m.group(2)

                    if m.group(3):
                        min_value = float(m.group(3))
                    else:
                        min_value = None

                    if m.group(4):
                        mid_value = float(m.group(4))
                    else:
                        mid_value = None

                    if m.group(5):
                        max_value = float(m.group(5))
                    else:
                        max_value = None

                    fmt = _make_3_color_scale_conditional_format(
                        df[col],
                        min_color=min_color,
                        mid_color="White",
                        max_color=max_color,
                        min_value=min_value,
                        mid_value=mid_value,
                        max_value=max_value,
                    )
                    worksheet.conditional_format(0, i, max_row, i, fmt)

                if m := re.match(
                    r"^number_(Blue|Green|Red)(_desc)?\|?([\-\+\d.]+)?,?([\-\+\d.]+)?$",
                    column_format,
                ):
                    desc = m.group(2) == "_desc"
                    if desc:
                        min_color = "White"
                        max_color = m.group(1)
                    else:
                        min_color = m.group(1)
                        max_color = "White"

                    if m.group(3):
                        min_value = float(m.group(3))
                    else:
                        min_value = None

                    if m.group(4):
                        max_value = float(m.group(4))
                    else:
                        max_value = None

                    fmt = _make_2_color_scale_conditional_format(
                        df[col],
                        min_color=min_color,
                        max_color=max_color,
                        min_value=min_value,
                        max_value=max_value,
                    )
                    # print(col, df[col].dtype, df[col].iloc[:5].to_list(), fmt)
                    worksheet.conditional_format(0, i, max_row, i, fmt)

                if column_format == "3color":
                    fmt = _make_3_color_scale_conditional_format(df[col])
                    worksheet.conditional_format(0, i, max_row, i, fmt)

            _write_table_header(df.columns, worksheet, formats["header_format"])

            if autofilter:
                worksheet.autofilter(0, 0, max_row - 1, max_col - 1)


def _color(color: str):
    match color:
        case "Blue":
            return "#B3CDE3"
        case "Green":
            return "#B3E3B3"
        case "Red":
            return "#FFB3B3"
        case "White":
            return "#F7F7F7"
        case _:
            raise ValueError(f"Invalid color: {color}")


def _validate_column_formats(column_formats: dict[str, str]):
    assert isinstance(column_formats, dict), "column_formats must be a dict"
    for col, fmt in column_formats.items():
        if re.match(
            r"^number_(Blue|Green|Red)(Blue|Green|Red)?(_desc)?\|?([\-\+\d.]+)?,?([\-\+\d.]+)?,?([\-\+\d.]+)?$",
            fmt,
        ):
            continue
        else:
            assert fmt in (
                "text",
                "bold",
                "number",
                "scientific",
                "percentage",
                "nodecimal",
                "boolean",
            ), f"Unknown format: {fmt}"


def _validate_column_widths(column_widths: dict[str, int]):
    assert isinstance(column_widths, dict), "column_widths must be a dict"
    for col, width in column_widths.items():
        assert isinstance(width, int | float), f"Invalid column width: {width}"
        assert width > 0, f"Invalid column width: {width}"


def _is_writable(file_path: str | Path) -> None:
    file_path = Path(file_path)

    if file_path.exists() and file_path.is_file():
        try:
            with open(file_path, "a"):
                return True
        except PermissionError:
            return False

    return True


def _get_formats_dict(workbook):
    return defaultdict(
        lambda: workbook.add_format(),
        **{
            "noformat": workbook.add_format(),
            "header_format": workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top",
                    "fg_color": "#FCFCFC",
                    "border": 1,
                }
            ),
            "text": workbook.add_format({"font_name": "Calibri", "font_size": 11}),
            "bold": workbook.add_format({"bold": True}),
            "number": workbook.add_format({"num_format": "0.00"}),
            "scientific": workbook.add_format({"num_format": "0.00E+00"}),
            "percentage": workbook.add_format({"num_format": "0.00%"}),
            "nodecimal": workbook.add_format({"num_format": "0"}),
            "boolean": workbook.add_format(
                {
                    "bold": True,
                    "valign": "center",
                    "align": "center",
                    "font_name": "Monospace",
                    "font_size": 11,
                }
            ),
            **{
                f"cell{color}": workbook.add_format(
                    {
                        "bg_color": _color(color),
                    }
                )
                for color in ["Blue", "Green", "Red", "White"]
            },
        },
    )


def _guess_column_format(column: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(column):
        return "boolean"

    if pd.api.types.is_numeric_dtype(column):
        if pd.api.types.is_float_dtype(column):
            eps = np.finfo(column.dtype).eps

            if (column % 1 == 0).all():
                return "nodecimal"

            if np.log10(column.abs() + eps).max() > 6:
                return "scientific"

            if np.log10(column.abs() + eps).min() < -3:
                return "scientific"

            return "number"

        return "nodecimal"

    return "text"


def _guess_column_width(
    column: pd.Series,
    column_format: str,
    step: int = 6,
    min_width: int = 10,
    max_width: int = 75,
) -> int:
    # widths should be multiple of 5, minimum 10 and maximum 75
    colname = str(column.name)
    colname_len = len(colname)
    if len(column) == 0:
        return _round_width(colname_len, step, min_width, max_width)

    match column_format:
        case "text":
            max_len = int(column.str.len().max())
            return _round_width(max(max_len, colname_len), step, min_width, max_width)

        case "number" | "nodecimal" | "scientific" | "boolean":
            return _round_width(colname_len, step, min_width, max_width)

        case x if re.match("^number_.*", x):
            return _round_width(colname_len, step, min_width, max_width)

        case "percentage":
            return _round_width(max(min_width, colname_len), step, min_width, max_width)

        case _:
            raise ValueError(f"Unknown format: {column_format}")


def _round_width(value: float, step: int, min_width: int, max_width: int) -> int:
    out = math.ceil(value * 1.4)
    return int(max(min_width, min(max_width, out)))


def _print_column_info(i, sheet_name, col, column_format, column_width, max_col):
    if i == 0:
        print()
        print("-" * 55)
        print(f"| Sheet Name: {sheet_name:<39} |")
        print("-" * 55)
        print(f"| {'Column':<20} | {'Format':<20} | {'Width':>5} |")
        print("-" * 55)
    print(f"| {col[:20]:>20} | {column_format:>20} | {column_width:>5} |")
    if i == max_col - 1:
        print("-" * 55)


def _make_percentage_conditional_format():
    return {
        "type": "data_bar",
        "bar_color": "#76C1E1",  # Light blue for the bar color
        "min_type": "num",  # Define minimum as a numeric value
        "min_value": 0,  # Minimum percentage (0%)
        "max_type": "num",  # Define maximum as a numeric value
        "max_value": 1,  # Maximum percentage (100%)
    }


def _make_cell_color_conditional_format(criteria, value, fmt):
    return {
        "type": "cell",
        "criteria": criteria,
        "value": value,
        "format": fmt,
    }


def _make_3_color_scale_conditional_format(
    series: pd.Series,
    scale: float = 1.5,
    min_color: str = "Blue",  # soft blue
    mid_color: str = "White",  # off-white
    max_color: str = "Red",  # soft red
    min_value: int | float | None = None,
    mid_value: int | float | None = None,
    max_value: int | float | None = None,
):
    values = series.loc[np.isfinite(series)]

    if min_value is None and mid_value is None and max_value is None:
        bound = values.quantile([0, 1]).abs().max() * scale
        min_value = -bound
        mid_value = 0
        max_value = bound

    if min_value is None:
        min_value = values.min()

    if max_value is None:
        max_value = values.max()

    if mid_value is None:
        mid_value = (values.max() + values.min()) / 2

    if not min_color.startswith("#"):
        min_color = _color(min_color)

    if not mid_color.startswith("#"):
        mid_color = _color(mid_color)

    if not max_color.startswith("#"):
        max_color = _color(max_color)

    return {
        "type": "3_color_scale",
        "min_type": "num",  # Can be "num", "percent", or "percentile"
        "min_value": min_value,
        "min_color": min_color,
        "mid_type": "num",
        "mid_value": mid_value,
        "mid_color": mid_color,
        "max_type": "num",
        "max_value": max_value,
        "max_color": max_color,
    }


def _make_2_color_scale_conditional_format(
    series: pd.Series,
    min_color: str = "#F7F7F7",
    max_color: str = "#B3E3B3",
    min_value: int | float | None = None,
    max_value: int | float | None = None,
):
    values = series.loc[np.isfinite(series)]

    if min_value is None:
        min_value = values.quantile(0.05)

    if max_value is None:
        max_value = values.quantile(0.95)

    if not min_color.startswith("#"):
        min_color = _color(min_color)

    if not max_color.startswith("#"):
        max_color = _color(max_color)

    return {
        "type": "2_color_scale",
        "min_type": "num",
        "min_value": min_value,
        "min_color": min_color,
        "max_type": "num",
        "max_value": max_value,
        "max_color": max_color,
    }


def _write_table_header(column_names: Sequence[str], worksheet, header_format):
    # Write the column headers with the defined format.
    for col_num, value in enumerate(column_names):
        worksheet.write(0, col_num, value, header_format)
