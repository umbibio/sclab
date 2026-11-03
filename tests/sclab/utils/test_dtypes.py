import pandas as pd

from sclab.utils import text_columns, text_or_category_columns


def _df():
    return pd.DataFrame(
        {
            "default_str": ["a", "b"],
            "object_str": pd.Series(["a", "b"], dtype=object),
            "cat": pd.Categorical(["u", "v"]),
            "num": [1, 2],
            "flag": [True, False],
        }
    )


def test_text_columns_cover_object_and_str_dtypes():
    assert text_columns(_df()) == ["default_str", "object_str"]


def test_text_or_category_columns_keep_column_order():
    assert text_or_category_columns(_df()) == ["default_str", "object_str", "cat"]
