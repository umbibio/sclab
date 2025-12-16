import pandas as pd
from anndata import AnnData

from ..utils import aggregate_and_filter


def edger_is_available() -> bool:
    try:
        _try_imports()
    except ImportError:
        return False
    return True


def pseudobulk_edger(
    adata_: AnnData,
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
    """
    Fits a model using edgeR and computes top tags for a given condition vs
    reference group.

    Parameters
    ----------
    adata_ : AnnData
        Annotated data matrix.
    group_key : str
        Key in AnnData object to use to group cells.
    condition_group : str | list[str] | None, optional
        Condition group to compare to reference group. If None, each group will be
        contrasted to the corresponding reference group.
    reference_group : str | None, optional
        Reference group to compare condition group(s) to. If None, the condition group
        is compared to the rest of the cells.
    cell_identity_key : str | None, optional
        If provided, separate contrasts will be computed for each identity. Defaults to None.
    layer : str | None, optional
        Layer in AnnData object to use. EdgeR requires raw counts. Defaults to None.
    replicas_per_group : int, optional
        Number of replicas to create for each group. Defaults to 10.
    min_cells_per_group : int, optional
        Minimum number of cells required for a group to be included. Defaults to 30.
    bootstrap_sampling : bool, optional
        Whether to use bootstrap sampling to create replicas. Defaults to True.
    use_cells : dict[str, list[str]] | None, optional
        If not None, only use the specified cells. Defaults to None. Dictionary key
        is a categorical variable in the obs dataframe and the dictionary value is a
        list of categories to include.
    aggregate : bool, optional
        Whether to aggregate cells before fitting the model. EdgeR requires a small
        number of samples, so if adata_ is a single-cell experiment, the cells should
        be aggregated. Defaults to True.
    verbosity : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of dataframes, one for each contrast, with the following columns:

        * gene_ids : str
            Gene IDs.
        * logFC : float
            Log2 fold change.
        * logCPM : float
            Log2 CPM.
        * F: float
            F-statistic.
        * PValue : float
            p-value.
        * FDR : float
            False discovery rate.
        * pct_expr_cnd : float
            Percentage of cells in condition group expressing the gene.
        * pct_expr_ref : float
            Percentage of cells in reference group expressing the gene.
    """
    _try_imports()
    import anndata2ri  # noqa: F401
    import rpy2.robjects as robjects
    from rpy2.rinterface_lib.embedded import RRuntimeError  # noqa: F401
    from rpy2.robjects import pandas2ri  # noqa: F401
    from rpy2.robjects.conversion import localconverter  # noqa: F401

    R = robjects.r

    if aggregate:
        aggr_adata = aggregate_and_filter(
            adata_,
            group_key,
            cell_identity_key,
            layer,
            replicas_per_group,
            min_cells_per_group,
            bootstrap_sampling,
            use_cells,
        )
    else:
        aggr_adata = adata_.copy()

    with localconverter(anndata2ri.converter):
        R.assign("aggr_adata", aggr_adata)

    # defines the R function for fitting the model with edgeR
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
                outs <- fit_edger_model(aggr_adata, "{gk}", "{cell_identity_key}", "{batch_key}", verbosity = {verbosity})
                fit <- outs$fit
                y <- outs$y
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

            R(f"myContrast <- makeContrasts('{contrast}', levels = y$design)")
            R("qlf <- glmQLFTest(fit, contrast=myContrast)")
            R("tt <- topTags(qlf, n = Inf)$table")
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
    library(MAST)
})

fit_edger_model <- function(adata_, group_key, cell_identity_key = "None", batch_key = "None", verbosity = 0){

    if (verbosity > 0){
        cat("Group key:", group_key, "\n")
        cat("Cell identity key:", cell_identity_key, "\n")
    }

    # create a vector that is concatentation of condition and cell type that we will later use with contrasts
    if (cell_identity_key == "None"){
        group <- colData(adata_)[[group_key]]
    } else {
        group <- paste0(colData(adata_)[[group_key]], "_", colData(adata_)[[cell_identity_key]])
    }

    if (verbosity > 1){
        cat("Group(s):", group, "\n")
    }

    group   <- factor(group)
    replica <- factor(colData(adata_)$replica)

    # create a design matrix
    if (batch_key == "None"){
        design <- model.matrix(~ 0 + group + replica)
    } else {
        batch  <- factor(colData(adata_)[[batch_key]])
        design <- model.matrix(~ 0 + group + replica + batch)
    }
    colnames(design) <- make.names(colnames(design))

    # create an edgeR object with counts and grouping factor
    y <- DGEList(assay(adata_, "X"), group = group)

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

    # estimate dispersion
    y <- estimateDisp(y, design = design)
    # fit the model
    fit <- glmQLFit(y, design)

    return(list("fit"=fit, "design"=design, "y"=y))
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
        importr("MAST")
        importr("SingleCellExperiment")

    except ModuleNotFoundError:
        message = (
            "edger_pseudobulk requires rpy2 and anndata2ri to be installed.\n"
            "please install with one of the following:\n"
            "$ pip install rpy2 anndata2ri\n"
            "or\n"
            "$ conda install -c conda-forge rpy2 anndata2ri\n"
        )
        print(message)
        raise ModuleNotFoundError(message)

    except PackageNotInstalledError:
        message = (
            "edger_pseudobulk requires the following R packages to be installed: edgeR, MAST, and SingleCellExperiment.\n"
            "> \n"
            "> if (!require('BiocManager', quietly = TRUE)) install.packages('BiocManager');\n"
            "> BiocManager::install(c('edgeR', 'MAST', 'SingleCellExperiment'));\n"
            "> \n"
        )
        print(message)
        raise ImportError(message)
