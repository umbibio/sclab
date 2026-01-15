from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData


def scdblfinder_is_available() -> bool:
    try:
        _try_imports()
    except ImportError:
        return False
    return True


def scdblfinder(
    adata: AnnData,
    layer: str = "X",
    key_added: str = "scDblFinder",
    clusters_col: str | bool | None = None,
    samples_col: str | None = None,
    clust_cor: np.ndarray | int | None = None,
    artificial_doublets: int | None = None,
    known_doublets_col: int | None = None,
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
):
    _try_imports()

    import anndata2ri
    from rpy2.robjects import ListVector, StrVector
    from rpy2.robjects import r as R
    from rpy2.robjects.conversion import localconverter

    if clusters_col is not None:
        assert clusters_col in adata.obs.columns, (
            f"Column {clusters_col} not found in adata.obs"
        )

    adata_ = adata.copy()
    if layer == "X":
        X = adata_.X
    else:
        X = adata_.layers[layer]

    for layer in list(adata_.layers.keys()):
        adata_.layers.pop(layer, None)

    adata_.layers["counts"] = X.astype(float)

    # keep obs columns used for doublet detection
    cols = [col for col in adata_.obs.columns if col in [clusters_col, samples_col]]
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
        "dbrSd": dbr_sd,
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

        vector = R("sce$scDblFinder.class")
        adata.obs[f"{key_added}_label"] = pd.Categorical(vector, ["singlet", "doublet"])
        adata.obs[f"{key_added}_score"] = R("sce$scDblFinder.score")


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
