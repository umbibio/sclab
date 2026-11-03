"""Selection provenance of the Keep/Drop/Label steps: replay and code emission.

The dashboard's ``filter_rows`` (Keep/Drop) and ``apply_label`` (Label) steps
record how the cells were selected instead of their obs names. The selection
is the intersection of the active filter controls and the plot selection::

    {
        "filters": {                          # active filter controls, ANDed
            "batch": {"isin": ["b1"]},        # categorical / bool SelectMultiple
            "total_counts": {"between": [500, 5000]},  # range slider, inclusive
        },
        "shape": {                            # box or lasso drawn on the plot
            "id": "sel1",                     # key in adata.uns["sclab"]["selections"]
            "type": "box",                    # or "lasso" with "polygon": [[x, y], ...]
            "basis": "X_umap",                # obsm key; absent: dims are obs columns
            "dims": [0, 1],                   # plotted obsm columns (or obs column names)
            "x": [1.2, 3.4], "y": [-2.0, 0.5],
        },
        "mask": "sel2",                       # instead of filters/shape, see below
        "n": 123,                             # selected cells, checked on replay
        "drop": True,                         # filter_rows only: Drop instead of Keep
    }

Absent keys mean "not used". When filters and shape do not reproduce the
selection exactly (cells picked by code, a plot selection on log, rank or
rotated axes, several lassos), the whole selection is recorded as a boolean
``mask`` instead. Shapes and masks are stored h5ad-safe in
``adata.uns["sclab"]["selections"][id]`` as ``{"type", "basis"?, "dims", "x"?,
"y"?}``, ``{"type": "lasso", "basis"?, "dims", "polygon": (n, 2) array}`` or
``{"type": "mask", "mask": bool array}``. Shape params also carry the geometry
inline, so a saved history replays without the uns entry; a mask is only in uns.

Pre-0.5 entries with an ``index`` list of obs names replay as before.

``filter_rows_code`` and ``apply_label_code`` render an entry's params as
Python that needs ``import numpy as np``, ``import pandas as pd`` and
``import sclab``, and acts on a variable ``adata``.
"""

import json

import numpy as np
from anndata import AnnData

from ...tools._select import select_box, select_polygon
from ._history import UNS_KEY

SELECTIONS_KEY = "selections"


def stored_selections(adata: AnnData) -> dict:
    return adata.uns.setdefault(UNS_KEY, {}).setdefault(SELECTIONS_KEY, {})


def store_selection(adata: AnnData, payload: dict, key: str | None = None) -> str:
    """Store a shape or mask in ``adata.uns`` under ``key`` or a new id."""
    selections = stored_selections(adata)
    i = len(selections) + 1
    while key is None and f"sel{i}" in selections:
        i += 1
    key = key or f"sel{i}"
    selections[key] = {
        k: np.asarray(v) if isinstance(v, list | tuple | np.ndarray) else v
        for k, v in payload.items()
        if k != "id"
    }
    return key


def shape_mask(adata: AnnData, shape: dict) -> np.ndarray:
    basis, dims = shape.get("basis"), list(shape["dims"])
    if shape["type"] == "box":
        return select_box(adata, basis, x=shape.get("x"), y=shape.get("y"), dims=dims)
    return select_polygon(adata, basis, shape["polygon"], dims=dims)


def selection_mask(
    adata: AnnData,
    filters: dict | None = None,
    shape: dict | None = None,
    mask: str | None = None,
) -> np.ndarray:
    """Boolean mask of the cells a recorded selection picks in ``adata``."""
    selected = np.ones(adata.n_obs, dtype=bool)
    for column, f in (filters or {}).items():
        series = adata.obs[column]
        if "between" in f:
            series = series.between(*f["between"])
        else:
            series = series.isin(f["isin"])
        selected &= series.to_numpy(dtype=bool, na_value=False)
    if shape:
        selected &= shape_mask(adata, shape)
    if mask:
        if mask not in (selections := stored_selections(adata)):
            raise KeyError(
                f"selection {mask!r} not in adata.uns['sclab']['selections']"
            )
        selected &= np.asarray(selections[mask]["mask"], dtype=bool)
    return selected


def _lit(value) -> str:
    """Python literal; strings in double quotes."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return ", ".join(map(_lit, value)).join("[]")
    return repr(value)


def _uns(key: str, field: str) -> str:
    return f'adata.uns["{UNS_KEY}"]["{SELECTIONS_KEY}"][{_lit(key)}][{_lit(field)}]'


def _shape_call(shape: dict) -> str:
    args = [_lit(shape.get("basis"))]
    if shape["type"] == "box":
        func = "select_box"
        args += [
            f"{k}=({_lit(shape[k][0])}, {_lit(shape[k][1])})"
            for k in "xy"
            if k in shape
        ]
    else:
        func = "select_polygon"
        args.append(_uns(shape["id"], "polygon"))
    if list(shape["dims"]) != [0, 1]:
        args.append(f"dims=({', '.join(map(_lit, shape['dims']))})")
    return f"sclab.tools.{func}(adata, {', '.join(args)})"


def _mask_lines(params: dict) -> list[str]:
    terms = []
    for column, f in (params.get("filters") or {}).items():
        if "between" in f:
            lo, hi = f["between"]
            terms.append(f"adata.obs[{_lit(column)}].between({_lit(lo)}, {_lit(hi)})")
        else:
            terms.append(f"adata.obs[{_lit(column)}].isin({_lit(list(f['isin']))})")
    if params.get("shape"):
        terms.append(_shape_call(params["shape"]))
    if params.get("mask"):
        terms.append(_uns(params["mask"], "mask"))  # last: `&=` must not modify uns
    return [f"mask = {terms[0]}"] + [f"mask &= {t}" for t in terms[1:]]


def _legacy(step: str) -> str:
    return f"# {step}: cells recorded as a list of obs names (pre-0.5); no code"


def filter_rows_code(params: dict) -> str:
    """Code for a ``filter_rows`` (Keep/Drop) history entry."""
    if "index" in params:
        return _legacy("filter_rows")
    keep = "~mask" if params.get("drop") else "mask"
    return "\n".join([*_mask_lines(params), f"adata = adata[{keep}].copy()"])


def apply_label_code(params: dict) -> str:
    """Code for an ``apply_label`` (Label) history entry; mirrors ``apply_label``."""
    if "index" in params:
        return _legacy("apply_label")
    column, label = _lit(params["column"]), _lit(params["label"])
    lines = [
        *_mask_lines(params),
        f"if {column} not in adata.obs:",
        f"    adata.obs[{column}] = pd.Categorical("
        f"[None] * adata.n_obs, categories=[{label}])",
    ]
    if params["label"]:
        lines += [
            f"elif {label} not in adata.obs[{column}].cat.categories:",
            f"    adata.obs[{column}] = adata.obs[{column}].cat.add_categories({label})",
        ]
    value = label if params["label"] else "np.nan"
    lines.append(f"adata.obs.loc[mask, {column}] = {value}")
    return "\n".join(lines)
