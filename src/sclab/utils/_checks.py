import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import issparse


def looks_like_raw_counts(X: ArrayLike) -> bool:
    """Return True if all values of ``X`` are non-negative integers.

    Works for dense arrays and any scipy sparse matrix or array. For sparse
    input only the stored values are inspected.
    """
    data = X.data if issparse(X) else np.asarray(X)
    if data.size == 0:
        return True
    return bool(np.all(data >= 0) and np.all(np.mod(data, 1) == 0))


def check_raw_counts(X: ArrayLike, name: str = "data", hint: str = "") -> None:
    """Raise ``ValueError`` if ``X`` does not look like raw counts.

    Parameters
    ----------
    X : array-like or sparse
        Matrix to check.
    name : str, optional
        How to refer to ``X`` in the error message, e.g. ``"layer 'X'"``.
    hint : str, optional
        Extra guidance appended to the error message.
    """
    if not looks_like_raw_counts(X):
        message = (
            f"{name} does not look like raw counts (expected non-negative integers)."
        )
        if hint:
            message += f" {hint}"
        raise ValueError(message)
