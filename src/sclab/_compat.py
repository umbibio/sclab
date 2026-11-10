"""Platform flags and the registry of optional dependencies.

The step reasons ("Needs scanpy: ...") and sclab's ``ImportError`` messages
are built from :data:`FEATURES`, as is :func:`optional_dependencies`.
"""

import sys
from importlib.util import find_spec
from typing import NamedTuple

IS_WASM = sys.platform == "emscripten"  # Pyodide: no threads, no native extensions


class Feature(NamedTuple):
    used_by: str  # what needs it, for the user
    modules: tuple[str, ...]  # import names, checked with find_spec
    extra: str | None  # the sclab extra installing them, or None
    pip: tuple[str, ...]  # distribution names, for pip without an extra / micropip


FEATURES: dict[str, Feature] = {
    "scanpy": Feature(
        "QC, Preprocess, PCA, Neighbors, UMAP, Cluster; wilcoxon, t-test, logreg DE",
        ("scanpy",),
        "scanpy",
        ("scanpy",),
    ),
    "bbknn": Feature("Neighbors with Group By Batch", ("bbknn",), "bbknn", ("bbknn",)),
    "scanorama": Feature(
        "Integration flavor scanorama",
        ("scanorama", "scanpy"),
        None,
        ("scanorama", "scanpy"),
    ),
    "scrublet": Feature(
        "Doublet Detection: scrublet", ("scrublet",), "doublets", ("scrublet",)
    ),
    "doubletdetection": Feature(
        "Doublet Detection: doubletdetection",
        ("doubletdetection",),
        "doublets",
        ("doubletdetection",),
    ),
    "scDblFinder": Feature(
        "Doublet Detection: scDblFinder (also needs R with scDblFinder, scater)",
        ("rpy2", "anndata2ri"),
        "r",
        ("rpy2", "anndata2ri"),
    ),
    "scDblFinder-py": Feature(
        "Doublet Detection: scDblFinder without R (backend='python')",
        ("pyscdblfinder", "igraph"),
        "scdblfinder-py",
        ("pyscdblfinder", "igraph"),
    ),
    "pylimma": Feature("pseudobulk_limma DE", ("pylimma",), "de", ("pylimma",)),
    "edgepython": Feature(
        "pseudobulk_edger DE; edgeR filtering in pseudobulk_limma",
        ("edgepython",),
        "de-edger",
        ("edgepython",),
    ),
    "jscatter": Feature(
        "jscatter 2D plot backend", ("jscatter",), "jscatter", ("jupyter-scatter",)
    ),
    "lazy": Feature(
        "Lazy loading of large datasets", ("xarray", "dask"), "lazy", ("xarray", "dask")
    ),
    "R": Feature(
        "R methods: edgeR and limma (R), ALRA imputation (also need R packages)",
        ("rpy2", "anndata2ri"),
        "r",
        ("rpy2", "anndata2ri"),
    ),
}


def install_command(feature: str) -> str:
    """How to install ``feature``'s packages: micropip under Pyodide, else pip."""
    f = FEATURES[feature]
    if IS_WASM and f.extra == "r":
        return "not available under Pyodide (no R)"
    if IS_WASM:
        return f"await micropip.install({list(f.pip)!r})"
    if f.extra is not None:
        return f"pip install 'sclab[{f.extra}]'"
    return f"pip install {' '.join(f.pip)}"


def _missing_modules(feature: str) -> list[str]:
    return [m for m in FEATURES[feature].modules if find_spec(m) is None]


def requirement(feature: str) -> str:
    """``"<packages>: <install command>"``, e.g. for "pseudobulk_limma needs ..."."""
    return f"{', '.join(FEATURES[feature].modules)}: {install_command(feature)}"


def missing(feature: str) -> str | None:
    """A step's :meth:`validate` reason if ``feature``'s packages are not installed."""
    if not _missing_modules(feature):
        return None
    return f"Needs {requirement(feature)}"


def require(feature: str):
    """Raise ``ImportError`` with the :func:`missing` reason, if any."""
    if reason := missing(feature):
        raise ImportError(reason)


def optional_dependencies(print_table: bool = True):
    """The optional features of sclab, their packages, and whether they are installed.

    Availability is checked with :func:`importlib.util.find_spec`: nothing is
    imported (R packages are not checked).

    Parameters
    ----------
    print_table : bool, optional
        Print the table. Default is True.

    Returns
    -------
    pandas.DataFrame
        One row per feature: ``feature``, ``used_by``, ``packages``,
        ``available``, ``missing`` and ``install``.
    """
    import pandas as pd

    rows = []
    for name, f in FEATURES.items():
        missing_modules = _missing_modules(name)
        rows.append(
            dict(
                feature=name,
                used_by=f.used_by,
                packages=", ".join(f.modules),
                available=not missing_modules,
                missing=", ".join(missing_modules),
                install=install_command(name),
            )
        )
    df = pd.DataFrame(rows)
    if print_table:  # left-aligned, the long text last
        cols = ["feature", "available", "install", "used_by"]
        text = df[cols].astype(str)
        widths = [max(len(c), text[c].str.len().max()) for c in cols]
        for row in [cols, *text.itertuples(index=False)]:
            print("  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())
    return df
