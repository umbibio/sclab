from importlib.util import find_spec

import pandas as pd
from anndata import AnnData

from ..utils import aggregate_and_filter

PYTHON_DEPENDENCIES = ["rpy2", "anndata2ri"]
R_DEPENDENCIES = ["edgeR", "limma", "MAST", "SingleCellExperiment"]


def limma_is_available() -> bool:
    return _python_deps_available() and _r_deps_available()


def pseudobulk_limma(
    adata: AnnData,
    group_key: str,
    condition_group: str | list[str] | None = None,
    reference_group: str | None = None,
    cell_identity_key: str | None = None,
    batch_key: str | None = None,
    layer: str | None = None,
    replicas_per_group: int = 5,
    min_cells_per_group: int = 30,
    bootstrap_sampling: bool = False,
    use_cells: dict[str, list[str]] | None = None,
    aggregate: bool = True,
    verbosity: int = 0,
) -> dict[str, pd.DataFrame]:
    """Pseudobulk differential expression analysis using limma-voom.

    Aggregates single cells into pseudobulk samples, then fits a linear
    model with limma-voom (via R) and computes top-table statistics for
    each requested contrast.

    Requires R with the packages ``limma``, ``edgeR``, ``MAST``, and
    ``SingleCellExperiment``, as well as the Python packages ``rpy2`` and
    ``anndata2ri``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key : str
        Column in ``adata.obs`` defining the experimental groups.
    condition_group : str or list of str or None, optional
        Group(s) to test against ``reference_group``. If None, each group
        is contrasted with the corresponding reference. Default is None.
    reference_group : str or None, optional
        Reference group for contrasts. If None, each condition group is
        contrasted with all remaining cells. Default is None.
    cell_identity_key : str or None, optional
        Column in ``adata.obs`` for stratifying contrasts by cell type or
        identity. Separate DE results are returned per identity.
        Default is None.
    batch_key : str or None, optional
        Column in ``adata.obs`` to include as a covariate in the design
        matrix for batch correction. Default is None.
    layer : str or None, optional
        Layer containing raw counts required by limma/edgeR. Uses
        ``adata.X`` if None. Default is None.
    replicas_per_group : int, optional
        Number of pseudobulk replicas to create per group. Default is 5.
    min_cells_per_group : int, optional
        Minimum number of cells required for a group to be included.
        Default is 30.
    bootstrap_sampling : bool, optional
        If True, use bootstrap sampling when creating pseudobulk replicas.
        Default is False.
    use_cells : dict or None, optional
        Restrict analysis to specific cell subsets. Keys are ``adata.obs``
        columns and values are lists of categories to include. Default is
        None.
    aggregate : bool, optional
        If True, aggregate cells into pseudobulk samples before fitting.
        Default is True.
    verbosity : int, optional
        Verbosity level (0 = silent). Default is 0.

    Returns
    -------
    dict of str to pd.DataFrame
        One DataFrame per contrast (keyed by contrast label), with columns:

        - ``logFC`` — log2 fold change.
        - ``AveExpr`` — average log2 expression.
        - ``t`` — moderated t-statistic.
        - ``P.Value`` — raw p-value.
        - ``adj.P.Val`` — Benjamini-Hochberg adjusted p-value.
        - ``B`` — log-odds of differential expression.
        - ``pct_expr_cnd`` / ``pct_expr_ref`` — fraction of expressing
          cells in condition/reference group.
    """
    _try_imports()
    import anndata2ri
    import rpy2.robjects as robjects
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    R = robjects.r

    if aggregate:
        aggr_adata = aggregate_and_filter(
            adata,
            group_key,
            cell_identity_key,
            layer,
            replicas_per_group,
            min_cells_per_group,
            bootstrap_sampling,
            use_cells,
        )
    else:
        aggr_adata = adata.copy()

    with localconverter(anndata2ri.converter):
        R.assign("aggr_adata", aggr_adata)

    # defines the R function for fitting the model with limma
    R(_fit_model_r_script)

    if condition_group is None:
        condition_group_list = aggr_adata.obs[group_key].unique()
    elif isinstance(condition_group, str):
        condition_group_list = [condition_group]
    else:
        condition_group_list = condition_group

    if cell_identity_key is not None:
        cids = aggr_adata.obs[cell_identity_key].unique()
    else:
        cids = [""]

    tt_dict = {}
    for condition_group in condition_group_list:
        if reference_group is not None and condition_group == reference_group:
            continue

        if verbosity > 0:
            print(f"Fitting model for {condition_group}...")

        if reference_group is not None:
            gk = group_key
        else:
            gk = f"{group_key}_{condition_group}"

        try:
            R(f"""
                outs <- fit_limma_model(aggr_adata, "{gk}", "{cell_identity_key}", verbosity = {verbosity})
                fit <- outs$fit
                v <- outs$v
            """)

        except RRuntimeError as e:
            print("Error fitting model for", condition_group)
            print("Error:", e)
            print("Skipping...", flush=True)
            continue

        if reference_group is None:
            new_contrasts_tuples = [
                (
                    condition_group,  # common prefix
                    "",  # condition group
                    "not",  # reference group
                    cid,  # cell identity
                )
                for cid in cids
            ]

        else:
            new_contrasts_tuples = [
                (
                    "",  # common prefix
                    condition_group,  # condition group
                    reference_group,  # reference group
                    cid,  # cell identity
                )
                for cid in cids
            ]

        new_contrasts = [
            f"group{cnd}{prefix}_{cid}".strip("_")
            + "-"
            + f"group{ref}{prefix}_{cid}".strip("_")
            for prefix, cnd, ref, cid in new_contrasts_tuples
        ]

        for contrast, contrast_tuple in zip(new_contrasts, new_contrasts_tuples):
            prefix, cnd, ref, cid = contrast_tuple

            if ref == "not":
                cnd, ref = "", "rest"

            contrast_key = f"{prefix}{cnd}_vs_{ref}"
            if cid:
                contrast_key = f"{cell_identity_key}:{cid}|{contrast_key}"

            if verbosity > 0:
                print(f"Computing contrast: {contrast_key}... ({contrast})")

            R(f"myContrast <- makeContrasts('{contrast}', levels = v$design)")
            R("fit2 <- contrasts.fit(fit, myContrast)")
            R("fit2 <- eBayes(fit2)")
            R("tt <- topTable(fit2, n = Inf)")
            tt: pd.DataFrame = pandas2ri.rpy2py(R("tt"))
            tt.index.name = "gene_ids"

            genes = tt.index
            cnd, ref = [c[5:] for c in contrast.split("-")]
            tt["pct_expr_cnd"] = aggr_adata.var[f"pct_expr_{cnd}"].loc[genes]
            tt["pct_expr_ref"] = aggr_adata.var[f"pct_expr_{ref}"].loc[genes]
            tt["num_expr_cnd"] = aggr_adata.var[f"num_expr_{cnd}"].loc[genes]
            tt["num_expr_ref"] = aggr_adata.var[f"num_expr_{ref}"].loc[genes]
            tt["tot_expr_cnd"] = aggr_adata.var[f"tot_expr_{cnd}"].loc[genes]
            tt["tot_expr_ref"] = aggr_adata.var[f"tot_expr_{ref}"].loc[genes]
            tt["mean_cnd"] = tt["tot_expr_cnd"] / tt["num_expr_cnd"]
            tt["mean_ref"] = tt["tot_expr_ref"] / tt["num_expr_ref"]
            tt_dict[contrast_key] = tt

    return tt_dict


_fit_model_r_script = """
suppressPackageStartupMessages({
    library(edgeR)
    library(limma)
    library(MAST)
})

fit_limma_model <- function(adata, group_key, cell_identity_key = "None", batch_key = "None", verbosity = 0){

    if (verbosity > 0){
        cat("Group key:", group_key, "\n")
        cat("Cell identity key:", cell_identity_key, "\n")
    }

    # create a vector that is concatentation of condition and cell type that we will later use with contrasts
    if (cell_identity_key == "None"){
        group <- colData(adata)[[group_key]]
    } else {
        group <- paste0(colData(adata)[[group_key]], "_", colData(adata)[[cell_identity_key]])
    }

    if (verbosity > 1){
        cat("Group(s):", group, "\n")
    }

    group   <- factor(group)
    replica <- factor(colData(adata)$replica)

    # create a design matrix
    if (batch_key == "None"){
        design <- model.matrix(~ 0 + group + replica)
    } else {
        batch  <- factor(colData(adata)[[batch_key]])
        design <- model.matrix(~ 0 + group + replica + batch)
    }
    colnames(design) <- make.names(colnames(design))

    # create an edgeR object with counts and grouping factor
    y <- DGEList(assay(adata, "X"), group = group)

    # filter out genes with low counts
    if (verbosity > 1){
        cat("Dimensions before subsetting:", dim(y), "\n")
    }

    keep <- filterByExpr(y, design = design)
    y <- y[keep, , keep.lib.sizes=FALSE]
    if (verbosity > 1){
        cat("Dimensions after subsetting:", dim(y), "\n")
    }

    # normalize
    y <- calcNormFactors(y)

    # Apply voom transformation to prepare for linear modeling
    v <- voom(y, design = design)

    # fit the linear model
    fit <- lmFit(v, design)
    ne <- limma::nonEstimable(design)
    if (!is.null(ne) && verbosity > 0) cat("Non-estimable:", ne, "\n")
    fit <- eBayes(fit)

    return(list("fit"=fit, "design"=design, "v"=v))
}
"""


def _try_imports():
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import PackageNotInstalledError, importr

        robjects.r("options(warn=-1)")
        import anndata2ri  # noqa: F401
        from rpy2.rinterface_lib.embedded import RRuntimeError  # noqa: F401
        from rpy2.robjects import numpy2ri, pandas2ri  # noqa: F401
        from rpy2.robjects.conversion import localconverter  # noqa: F401

        importr("edgeR")
        importr("limma")
        importr("MAST")
        importr("SingleCellExperiment")

    except ModuleNotFoundError:
        message = (
            "pseudobulk_limma requires rpy2 and anndata2ri to be installed.\n"
            "please install with one of the following:\n"
            "$ pip install rpy2 anndata2ri\n"
            "or\n"
            "$ conda install -c conda-forge rpy2 anndata2ri\n"
        )
        print(message)
        raise ModuleNotFoundError(message)

    except PackageNotInstalledError:
        message = (
            "pseudobulk_limma requires the following R packages to be installed: limma, edgeR, MAST, and SingleCellExperiment.\n"
            "> \n"
            "> if (!require('BiocManager', quietly = TRUE)) install.packages('BiocManager');\n"
            "> BiocManager::install(c('limma', 'edgeR', 'MAST', 'SingleCellExperiment'));\n"
            "> \n"
        )
        print(message)
        raise ImportError(message)


def _python_deps_available() -> bool:
    return all([find_spec(dep) is not None for dep in PYTHON_DEPENDENCIES])


def _r_deps_available() -> bool:
    if not find_spec("rpy2") or not find_spec("anndata2ri"):
        return False

    from rpy2.robjects import r

    installed_packages = list(r("installed.packages()"))
    return all([dep in installed_packages for dep in R_DEPENDENCIES])
