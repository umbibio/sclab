from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from .._citations import record_citations
from ._scanpy_port import get_pp

# flavor -> its citation key
HVG_METHODS = {
    "seurat": "satija2015",
    "cell_ranger": "zheng2017",
    "seurat_v3": "stuart2019",
    "seurat_v3_paper": "stuart2019",
}
HVG_COMBINE = ("union", "intersection")


def hvg_citations(methods: str | Sequence[str]) -> tuple[str, ...]:
    """The citation keys of the HVG ``methods``."""
    methods = (methods,) if isinstance(methods, str) else methods
    return tuple(dict.fromkeys(HVG_METHODS[m] for m in methods))


def highly_variable_genes(
    adata: AnnData,
    methods: str | Sequence[str] = ("seurat", "seurat_v3_paper"),
    combine: str = "union",
    *,
    counts_layer: str = "counts",
    log1p_layer: str | None = None,
    n_top_genes: int | None = None,
    batch_key: str | None = None,
    inplace: bool = True,
    backend: str = "auto",
) -> pd.DataFrame | None:
    """Highly variable genes by one or several methods, combined.

    Each method is a flavor of :func:`scanpy.pp.highly_variable_genes`, run by
    scanpy or by sclab's port of it (see ``backend``):

    - ``"seurat"`` (Satija et al. 2015) and ``"cell_ranger"`` (Zheng et al.
      2017): dispersion-based, on ``log1p_layer``;
    - ``"seurat_v3"`` and ``"seurat_v3_paper"`` (Stuart et al. 2019):
      normalized variance on ``counts_layer``; need ``scikit-misc``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    methods : str or sequence of str, optional
        The flavors to run, in order. Default ``("seurat", "seurat_v3_paper")``.
    combine : {"union", "intersection"}, optional
        A gene is highly variable if any (``"union"``) or every
        (``"intersection"``) method selects it. Default ``"union"``.
    counts_layer : str, optional
        Layer with raw counts. Default ``"counts"``.
    log1p_layer : str or None, optional
        Layer with log data for the dispersion flavors. Default
        ``f"{counts_layer}_log1p"``; computed from the counts (and not kept)
        if missing.
    n_top_genes : int or None, optional
        Genes each method selects. If None (default), the dispersion flavors
        use scanpy's default cutoffs and the seurat_v3 flavors select as many
        genes as the first method did (2000 if they come first).
    batch_key : str or None, optional
        Column of ``adata.obs``: each method selects genes per batch and
        merges them, as scanpy's ``batch_key``. Default None.
    inplace : bool, optional
        If True (default), each method writes its statistics to ``adata.var``
        as scanpy does (a later method overwrites shared columns such as
        ``means``), followed by ``highly_variable_<method>`` per method and
        the combined ``highly_variable``. If False, return a DataFrame with
        those flag columns.
    backend : {"auto", "native", "scanpy"}, optional
        ``"scanpy"`` runs :func:`scanpy.pp.highly_variable_genes`,
        ``"native"`` sclab's port of it, ``"auto"`` (default) scanpy if it is
        installed.

    Returns
    -------
    pandas.DataFrame or None
        The flags if ``inplace=False``.

    References
    ----------
    Satija et al. (2015). Spatial reconstruction of single-cell gene
    expression data. *Nature Biotechnology*, 33, 495–502.
    https://doi.org/10.1038/nbt.3192

    Zheng et al. (2017). Massively parallel digital transcriptional profiling
    of single cells. *Nature Communications*, 8, 14049.
    https://doi.org/10.1038/ncomms14049

    Stuart et al. (2019). Comprehensive Integration of Single-Cell Data.
    *Cell*, 177, 1888–1902.e21. https://doi.org/10.1016/j.cell.2019.05.031

    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0
    """
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    if not methods or any(m not in HVG_METHODS for m in methods):
        raise ValueError(f"methods must be among {tuple(HVG_METHODS)}, got {methods}")
    if combine not in HVG_COMBINE:
        raise ValueError(f"combine must be one of {HVG_COMBINE}, got {combine!r}")
    pp = get_pp(backend)
    log1p_layer = log1p_layer or f"{counts_layer}_log1p"
    temporary = log1p_layer not in adata.layers and any(
        not m.startswith("seurat_v3") for m in methods
    )
    if temporary:
        adata.layers[log1p_layer] = pp.log1p(adata.layers[counts_layer].copy())

    flags: dict[str, np.ndarray] = {}
    try:
        for method in methods:
            if method.startswith("seurat_v3"):
                first = next(iter(flags.values()), None)
                layer = counts_layer
                n_top = n_top_genes or (2000 if first is None else int(first.sum()))
            else:
                layer, n_top = log1p_layer, n_top_genes
            df = pp.highly_variable_genes(
                adata,
                layer=layer,
                flavor=method,
                n_top_genes=n_top,
                batch_key=batch_key,
                inplace=inplace,
            )
            hv = adata.var["highly_variable"] if inplace else df["highly_variable"]
            if not hv.index.equals(adata.var_names):  # batched, by cutoffs
                hv = hv.loc[adata.var_names]
            flags[method] = hv.to_numpy(dtype=bool)
    finally:
        if temporary:
            del adata.layers[log1p_layer]

    stacked = np.column_stack(list(flags.values()))
    combined = stacked.any(axis=1) if combine == "union" else stacked.all(axis=1)
    record_citations(adata, "scanpy", *hvg_citations(methods))
    result = pd.DataFrame(
        {f"highly_variable_{m}": f for m, f in flags.items()}, index=adata.var_names
    )
    result["highly_variable"] = combined
    if not inplace:
        return result
    for column in result:
        adata.var[column] = result[column].to_numpy()
    return None
