import warnings
from importlib.util import find_spec
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from sclab._citations import cites
from sclab.utils import check_raw_counts, layer_keys

PYTHON_DEPENDENCIES = ["rpy2", "anndata2ri"]
R_DEPENDENCIES = ["scater", "scDblFinder", "BiocParallel", "SingleCellExperiment"]


def scdblfinder_is_available() -> bool:
    """True if either backend (R or pure Python) can run."""
    return _r_available() or _pyscdblfinder_available()


def _r_available() -> bool:
    return _python_deps_available() and _r_deps_available()


def _pyscdblfinder_available() -> bool:
    return all(find_spec(dep) is not None for dep in ("pyscdblfinder", "igraph"))


@cites("scdblfinder")
def scdblfinder(
    adata: AnnData,
    layer: str = "X",
    key_added: str = "scDblFinder",
    check_counts: bool = True,
    clusters_col: str | bool | None = None,
    samples_col: str | None = None,
    clust_cor: np.ndarray | int | None = None,
    artificial_doublets: int | None = None,
    known_doublets_col: str | None = None,
    known_use: Literal["discard", "positive"] = "discard",
    dbr: float | None = None,
    dbr_sd: float | None = None,
    nfeatures: int = 1352,
    dims: int = 20,
    k: int | None = None,
    remove_unidentifiable: bool = True,
    include_pcs: int = 19,
    prop_random=0,
    prop_markers=0,
    aggregate_features: bool = False,
    score: Literal["xgb", "weighted", "ratio"] = "xgb",
    processing: str = "default",
    metric: str = "logloss",
    nrounds: float = 0.25,
    max_depth: int = 4,
    iter: int = 3,
    training_features: list[str] | None = None,
    unident_th: float | None = None,
    multi_sample_mode: Literal[
        "split", "singleModel", "singleModelSplitThres", "asOne"
    ] = "split",
    threshold: bool = True,
    verbose: bool = True,
    random_state: int = 31415,
    backend: Literal["auto", "python", "r"] = "auto",
):
    """Detect doublet cells using scDblFinder.

    Adds ``{key_added}_label`` and ``{key_added}_score`` (and
    ``{key_added}_cluster`` when ``clusters_col`` is given) to
    ``adata.obs``. Remaining parameters are forwarded to
    ``scDblFinder::scDblFinder`` (R) or ``pyscdblfinder.sc_dbl_finder``
    (Python). The input is copied; ``adata`` is only modified by adding
    the ``adata.obs`` columns.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer to use as the count matrix. Use ``"X"`` for ``adata.X``.
        Default is ``"X"``.
    key_added : str, optional
        Prefix for the columns added to ``adata.obs``. Default is
        ``"scDblFinder"``.
    check_counts : bool, optional
        If True, raise ``ValueError`` when the selected matrix does not
        look like raw counts (non-negative integers). Set to False to
        bypass the check, e.g. for non-integer corrected counts. Default
        is True.
    clusters_col : str or bool, optional
        Column of ``adata.obs`` with cluster labels, or True for
        scDblFinder's fast clustering (``clusters=TRUE``). Default is None
        (random artificial doublets).
    samples_col : str, optional
        Column of ``adata.obs`` with sample labels (R backend only).
    known_doublets_col : str, optional
        Boolean column of ``adata.obs`` marking known doublets (R backend
        only).
    backend : {"auto", "python", "r"}, optional
        ``"r"`` runs the Bioconductor package through rpy2. ``"python"``
        runs pyscdblfinder, a Python port (``pip install
        'sclab[scdblfinder-py]'``); it handles a single sample and raises
        ``NotImplementedError`` for ``samples_col``, ``known_doublets_col``
        and the other options it does not port. ``"auto"`` (default) uses
        R when R and scDblFinder are installed, and Python otherwise.

    References
    ----------
    .. [1] Germain P-L, Lun A, Garcia Meixide C, Macnair W, Robinson MD.
       Doublet identification in single-cell sequencing data using
       scDblFinder. F1000Research 2021, 10:979.
       https://doi.org/10.12688/f1000research.73600.2
    .. [2] pyscdblfinder 0.3.2, a Python port of scDblFinder,
       https://github.com/omicverse/py-scDblFinder, GPL-3.
    """
    if backend not in ("auto", "python", "r"):
        raise ValueError(f"backend must be 'auto', 'python' or 'r', not {backend!r}")

    if check_counts:
        X = adata.X if layer == "X" else adata.layers[layer]
        check_raw_counts(
            X,
            name=f"layer '{layer}'",
            hint=(
                "Doublet detection needs raw UMI counts; use layer='counts' "
                "(or check_counts=False for non-integer corrected counts)."
            ),
        )

    for col in (clusters_col, samples_col, known_doublets_col):
        if isinstance(col, str):  # clusters_col=True asks for fast clustering
            assert col in adata.obs.columns, f"Column {col} not found in adata.obs"

    if backend == "auto":
        backend = "r" if _r_available() else "python"

    if backend == "python":
        return _scdblfinder_python(
            adata,
            layer=layer,
            key_added=key_added,
            clusters_col=clusters_col,
            unsupported={
                "samples_col (multi-sample mode)": samples_col is not None,
                "known_doublets_col": known_doublets_col is not None,
                "clust_cor": clust_cor is not None,
                "prop_markers": prop_markers != 0,
                f"score={score!r}": score != "xgb",
                "training_features": training_features is not None,
                "remove_unidentifiable=False": not remove_unidentifiable,
            },
            artificial_doublets=artificial_doublets,
            dbr=dbr,
            dbr_sd=dbr_sd,
            n_features=nfeatures,
            dims=dims,
            k=k,
            include_pcs=include_pcs,
            prop_random=prop_random,
            aggregate_features=aggregate_features,
            processing=processing,
            metric=metric,
            nrounds=nrounds,
            max_depth=max_depth,
            iter=iter,
            unident_th=unident_th,
            threshold=threshold,
            verbose=verbose,
            random_state=random_state,
        )

    _try_imports()
    _warn_on_broken_xgboost()

    import anndata2ri
    from rpy2.robjects import ListVector, StrVector
    from rpy2.robjects import r as R
    from rpy2.robjects.conversion import localconverter

    adata_ = adata.copy()
    if layer == "X":
        X = adata_.X
    else:
        X = adata_.layers[layer]

    for layer in layer_keys(adata_):
        adata_.layers.pop(layer, None)

    adata_.layers["counts"] = X.astype(float)

    # keep obs columns used for doublet detection
    keep = [clusters_col, samples_col, known_doublets_col]
    cols = [col for col in adata_.obs.columns if col in keep]
    adata_.obs = adata_.obs[cols]

    # empty unnecessary metadata
    adata_.var = adata_.var[[]]

    # empty unnecessary uns
    adata_.uns = {}

    sc_dbl_finder_params_dict = {
        "clusters": clusters_col,
        "samples": samples_col,
        "clustCor": clust_cor,
        "artificialDoublets": artificial_doublets,
        "knownDoublets": known_doublets_col,
        "knownUse": known_use,
        "dbr": dbr,
        "dbr.sd": dbr_sd,
        "nfeatures": nfeatures,
        "dims": dims,
        "k": k,
        "removeUnidentifiable": remove_unidentifiable,
        "includePCs": include_pcs,
        "propRandom": prop_random,
        "propMarkers": prop_markers,
        "aggregateFeatures": aggregate_features,
        "score": score,
        "processing": processing,
        "metric": metric,
        "nrounds": nrounds,
        "max_depth": max_depth,
        "iter": iter,
        "trainingFeatures": StrVector(training_features) if training_features else None,
        "unident.th": unident_th,
        "multiSampleMode": multi_sample_mode,
        "threshold": threshold,
        "verbose": verbose,
    }

    sc_dbl_finder_params = ListVector(
        {k: v for k, v in sc_dbl_finder_params_dict.items() if v is not None}
    )

    with localconverter(anndata2ri.converter):
        R.assign("sce", adata_)
        R.assign("scDblFinderParams", sc_dbl_finder_params)
        R.assign("random_state", random_state)

    R(
        """
        suppressPackageStartupMessages({
            library(scater)
            library(scDblFinder)
            library(BiocParallel)
        })

        set.seed(random_state)
        sce <- do.call(scDblFinder, c(list(sce), scDblFinderParams))

        NULL
        """
    )

    with localconverter(anndata2ri.converter):
        if clusters_col:
            series = pd.Categorical(np.array(R("sce$scDblFinder.cluster"), dtype=int))
            categories: pd.Index = series.categories
            series = series.rename_categories(categories.astype(str))
            adata.obs[f"{key_added}_cluster"] = series

        # a factor (not character) when knownDoublets is given
        vector = R("as.character(sce$scDblFinder.class)")
        adata.obs[f"{key_added}_label"] = pd.Categorical(vector, ["singlet", "doublet"])
        adata.obs[f"{key_added}_score"] = R("sce$scDblFinder.score")


def _scdblfinder_python(
    adata: AnnData,
    layer: str,
    key_added: str,
    clusters_col: str | bool | None,
    unsupported: dict[str, bool],
    random_state: int,
    verbose: bool,
    **kwargs,
):
    """Run pyscdblfinder, the Python port of scDblFinder (single sample).

    References
    ----------
    .. [1] Germain P-L, Lun A, Garcia Meixide C, Macnair W, Robinson MD.
       Doublet identification in single-cell sequencing data using
       scDblFinder. F1000Research 2021, 10:979.
       https://doi.org/10.12688/f1000research.73600.2
    .. [2] pyscdblfinder 0.3.2, https://github.com/omicverse/py-scDblFinder,
       GPL-3.
    """
    if bad := [name for name, used in unsupported.items() if used]:
        raise NotImplementedError(
            f"The Python scDblFinder backend does not support {', '.join(bad)}; "
            "use backend='r' (requires R with scDblFinder)."
        )
    if not _pyscdblfinder_available():
        raise ImportError(
            "scdblfinder needs either R with scDblFinder (backend='r') or the "
            "Python port: pip install 'sclab[scdblfinder-py]'"
        )

    from pyscdblfinder import sc_dbl_finder
    from scipy import sparse

    from ._fastcluster import fastcluster

    X = adata.X if layer == "X" else adata.layers[layer]
    if clusters_col is True:
        clusters = fastcluster(
            X,
            ndims=kwargs["dims"],
            nfeatures=kwargs["n_features"],
            random_state=random_state,
        )
    elif clusters_col:
        clusters = adata.obs[clusters_col].to_numpy()
    else:
        clusters = None
    if clusters is not None and len(pd.unique(clusters)) < 2:
        raise ValueError("scDblFinder needs at least two clusters")

    if kwargs["dbr"] is None:  # R's default since 1.19.6; pyscdblfinder uses 1%
        kwargs["dbr_per1k"] = 0.008

    result = sc_dbl_finder(
        sparse.csr_matrix(X, dtype=np.float64).T.tocsr(),  # genes x cells
        clusters=clusters,
        random_state=random_state,
        verbose=verbose,
        progress=verbose,
        **kwargs,
    )
    real = result.real_cells()

    if clusters_col:
        series = pd.Categorical(clusters)
        series = series.rename_categories(series.categories.astype(str))
        adata.obs[f"{key_added}_cluster"] = series
    adata.obs[f"{key_added}_label"] = pd.Categorical(
        real["class"].to_numpy(), ["singlet", "doublet"]
    )
    adata.obs[f"{key_added}_score"] = real["score"].to_numpy(dtype=float)


def _warn_on_broken_xgboost() -> None:
    """Warn when scDblFinder cannot train its classifier with the R xgboost.

    scDblFinder < 1.24 calls the xgboost 1.x API (``xgb.cv(data=<matrix>)``),
    which xgboost >= 3 rejects; scDblFinder < 1.27.9 also fails with xgboost
    >= 3.4 (NEWS 1.27.9: "fixed issue with xgboost >=3.4"). scDblFinder
    catches the training error and keeps its starting score, silently before
    1.26. Only warns; the computation is not changed.
    """
    from rpy2.robjects import r

    def version(package: str) -> tuple[int, ...]:
        return tuple(int(v) for v in r(f'unlist(packageVersion("{package}"))'))

    sdf, xgb = version("scDblFinder"), version("xgboost")
    if not ((xgb >= (3,) and sdf < (1, 24)) or (xgb >= (3, 4) and sdf < (1, 27, 9))):
        return

    fix = "update scDblFinder to >= 1.27.9"
    if sdf < (1, 24):
        fix = f"install R xgboost 1.7.x, or {fix}"
    warnings.warn(
        f"scDblFinder {'.'.join(map(str, sdf))} cannot train its classifier "
        f"with R xgboost {'.'.join(map(str, xgb))}. It catches the training "
        "error and keeps its starting heuristic score, "
        "(cxds_score + ratio / max(ratio)) / 2, so the scores and calls are "
        f"that heuristic, not the trained model. To fix: {fix}.",
        UserWarning,
        stacklevel=3,
    )


def _try_imports():
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import PackageNotInstalledError, importr

        robjects.r("options(warn=-1)")
        import anndata2ri  # noqa: F401
        from rpy2.rinterface_lib.embedded import RRuntimeError  # noqa: F401
        from rpy2.robjects import numpy2ri, pandas2ri  # noqa: F401
        from rpy2.robjects.conversion import localconverter  # noqa: F401

        importr("scater")
        importr("scDblFinder")
        importr("BiocParallel")
        importr("SingleCellExperiment")

    except ModuleNotFoundError:
        message = (
            "scdblfinder requires rpy2 and anndata2ri to be installed.\n"
            "please install with one of the following:\n"
            "$ pip install rpy2 anndata2ri\n"
            "or\n"
            "$ conda install -c conda-forge rpy2 anndata2ri\n"
        )
        print(message)
        raise ModuleNotFoundError(message)

    except PackageNotInstalledError:
        message = (
            "scdblfinder requires the following R packages to be installed: scDblFinder, and SingleCellExperiment.\n"
            "> \n"
            "> if (!require('BiocManager', quietly = TRUE)) install.packages('BiocManager');\n"
            "> BiocManager::install(c('scDblFinder', 'SingleCellExperiment'));\n"
            "> \n"
        )
        print(message)
        raise ImportError(message)


def _python_deps_available() -> bool:
    return all([find_spec(dep) is not None for dep in PYTHON_DEPENDENCIES])


def _r_deps_available() -> bool:
    if not find_spec("rpy2"):
        return False

    try:
        from rpy2.robjects import r

        installed_packages = list(r("installed.packages()"))
    except Exception:  # rpy2 present but cannot load R (e.g. built for a newer R)
        return False
    return all([dep in installed_packages for dep in R_DEPENDENCIES])
