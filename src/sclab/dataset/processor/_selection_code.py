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
``import sclab``, and acts on a variable ``adata``. Box bounds and polygons of
up to ``MAX_INLINE_VERTICES`` vertices are inlined; longer lasso polygons and
masks live only in ``adata.uns`` of the dashboard, so the code reads them from
the adata the dashboard saved (``SELECTIONS_FILE``, to be edited by the user).
"""

import json

import numpy as np
from anndata import AnnData

from ...tools._select import select_box, select_polygon
from ._history import UNS_KEY

SELECTIONS_KEY = "selections"
SELECTIONS_FILE = "dashboard.h5ad"  # placeholder path in the emitted code
MAX_INLINE_VERTICES = 20  # longer lasso polygons are read from the saved adata


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


def _saved_selection(key: str, field: str) -> str:
    """Read a stored polygon / mask from the adata the dashboard saved."""
    return (
        f"sclab._io.read_adata({_lit(SELECTIONS_FILE)}).uns[{_lit(UNS_KEY)}]"
        f"[{_lit(SELECTIONS_KEY)}][{_lit(key)}][{_lit(field)}]"
    )


def _shape_lines(shape: dict) -> tuple[list[str], str]:
    """Setup lines (polygon) and the expression selecting the shape's cells."""
    from ._codegen import ADATA, Name, call  # circular import

    dims = {} if list(shape["dims"]) == [0, 1] else {"dims": tuple(shape["dims"])}
    basis = shape.get("basis")
    if shape["type"] == "box":
        bounds = {k: tuple(shape[k]) for k in "xy" if k in shape}
        return [], call("sclab.tools.select_box", ADATA, basis, **bounds, **dims)
    polygon = shape["polygon"]
    if len(polygon) <= MAX_INLINE_VERTICES:  # the geometry is in the params
        lines = ["polygon = [", *(f"    {_lit(list(v))}," for v in polygon), "]"]
    else:
        lines = [
            f"# lasso with {len(polygon)} vertices: its polygon is read from the adata",
            "# the dashboard saved (adata.write_h5ad); edit the path to where you put it",
            f"polygon = {_saved_selection(shape['id'], 'polygon')}",
        ]
    term = call("sclab.tools.select_polygon", ADATA, basis, Name("polygon"), **dims)
    return lines, term


def _mask_lines(params: dict) -> list[str]:
    setup, terms = [], []
    for column, f in (params.get("filters") or {}).items():
        if "between" in f:
            lo, hi = f["between"]
            terms.append(f"adata.obs[{_lit(column)}].between({_lit(lo)}, {_lit(hi)})")
        else:
            terms.append(f"adata.obs[{_lit(column)}].isin({_lit(list(f['isin']))})")
    if params.get("shape"):
        setup, term = _shape_lines(params["shape"])
        terms.append(term)
    if params.get("mask"):
        setup.append(
            "# boolean mask of cells picked in the dashboard: read from the saved adata"
        )
        terms.append(_saved_selection(params["mask"], "mask"))
    return [*setup, f"mask = {terms[0]}"] + [f"mask &= {t}" for t in terms[1:]]


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
    empty = params["label"] in ("", None)  # False is a label (bool columns)
    obs = f"adata.obs[{column}]"
    lines = [
        *_mask_lines(params),
        f"if {column} not in adata.obs:",
        f"    adata.obs[{column}] = pd.Categorical("
        f"[None] * adata.n_obs, categories=[{label}])",
    ]
    if empty:  # a numpy bool column needs NA support to be cleared
        lines += [f"elif {obs}.dtype == bool:", f'    {obs} = {obs}.astype("boolean")']
    else:  # bool columns take True / False as is
        lines += [
            f"elif isinstance({obs}.dtype, pd.CategoricalDtype) "
            f"and {label} not in {obs}.cat.categories:",
            f"    {obs} = {obs}.cat.add_categories({label})",
        ]
    value = "np.nan" if empty else label
    lines.append(f"adata.obs.loc[mask, {column}] = {value}")
    return "\n".join(lines)
