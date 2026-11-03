import pandas as pd
from pandas.api.types import is_string_dtype


def text_columns(df: pd.DataFrame) -> list:
    """Columns holding text: ``object`` dtype (pandas 2) or ``str`` dtype (pandas 3)."""
    return [c for c, dt in df.dtypes.items() if is_string_dtype(dt)]


def text_or_category_columns(df: pd.DataFrame) -> list:
    """Text and categorical columns of ``df``, in column order."""
    return [
        c
        for c, dt in df.dtypes.items()
        if isinstance(dt, pd.CategoricalDtype) or is_string_dtype(dt)
    ]
