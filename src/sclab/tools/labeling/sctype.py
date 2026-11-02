import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csc_matrix, csr_matrix, issparse

from ...preprocess import pool_neighbors
from ...utils import looks_like_raw_counts

logger = logging.getLogger(__name__)


def _count_cell_types(markers: pd.DataFrame, marker_class_key: str) -> int:
    """Number of cell-type categories in `markers`, excluding "Unknown"."""
    return int((markers[marker_class_key].cat.categories != "Unknown").sum())


def _count_above_gene_min(X: NDArray | csr_matrix | csc_matrix) -> NDArray:
    """Per gene, the number of cells whose value is above that gene's minimum."""
    if not issparse(X):
        return (X > X.min(axis=0)).sum(axis=0)

    X = X.tocsc()
    n_cells = X.shape[0]
    stored = np.diff(X.indptr)
    gene_min = X.min(axis=0).toarray().ravel()
    above = np.bincount(
        np.repeat(np.arange(X.shape[1]), stored),
        weights=X.data > np.repeat(gene_min, stored),
        minlength=X.shape[1],
    )
    # implicit zeros count as above the minimum only when the minimum is negative
    return above + np.where(gene_min < 0, n_cells - stored, 0)


def _get_classification_scores_matrix(
    adata: AnnData,
    markers: pd.DataFrame,
    marker_class_key: str,
    neighbors_key: Optional[str] = None,
    weighted_pooling: bool = False,
    directed_pooling: bool = True,
    layer: Optional[str] = None,
    penalize_non_specific: bool = True,
    specificity: Literal["sctype", "observed_range"] = "sctype",
    n_types: Optional[int] = None,
):
    # Ianevski, A., Giri, A.K. & Aittokallio, T.
    # Fully-automated and ultra-fast cell-type identification using specific
    # marker combinations from single-cell transcriptomic data.
    # Nat Commun 13, 1246 (2022).
    # https://doi.org/10.1038/s41467-022-28803-w

    if layer is not None:
        X = adata.layers[layer]
        X_name = f"layer '{layer}'"

    else:
        X = adata.X
        X_name = "adata.X"

    if looks_like_raw_counts(X):
        raise ValueError(
            f"sctype expects log-normalized expression; {X_name} looks like raw "
            "counts. Use a normalized layer (e.g. adata.X after Preprocess)."
        )

    if specificity not in ("sctype", "observed_range"):
        raise ValueError(
            f"specificity must be 'sctype' or 'observed_range', got {specificity!r}"
        )

    if n_types is None:
        n_types = _count_cell_types(markers, marker_class_key)

    # a gene is expressed in the cells above its own minimum; a global minimum
    # would let nearly every gene pass on z-scored data
    n_cells = _count_above_gene_min(X)
    mask = n_cells > 5
    print(f"using {mask.sum()} genes")

    markers = markers.loc[markers["names"].isin(adata.var_names[mask])].copy()
    classes = markers[marker_class_key].cat.categories

    x = markers[[marker_class_key, "names"]].groupby("names").count()[marker_class_key]
    if penalize_non_specific:
        if specificity == "sctype":
            # published ScType: rescale by the number of cell types in the table
            if n_types > 1:
                S = 1.0 - (x - 1.0) / (n_types - 1.0)
            else:
                S = x * 0.0 + 1.0
        else:
            # legacy: rescale by the range observed in this dataset
            if x.max() > x.min():
                S = 1.0 - (x - x.min()) / (x.max() - x.min())
            else:
                S = x * 0.0 + 1.0
        S = S[S > 0]
    else:
        S = x * 0.0 + 1.0

    X: NDArray | csr_matrix | csc_matrix
    if neighbors_key is not None:
        X = pool_neighbors(
            adata[:, S.index],
            key=layer,
            neighbors_key=neighbors_key,
            weighted=weighted_pooling,
            directed=directed_pooling,
            copy=True,
        )

    elif layer is not None:
        X = adata[:, S.index].layers[layer].copy()

    else:
        X = adata[:, S.index].X.copy()

    if issparse(X):
        X = np.asarray(X.todense("C"))

    # constant genes would be NaN after z-scoring
    nonconstant = X.std(axis=0) > 0
    if not nonconstant.all():
        markers = markers.loc[~markers["names"].isin(S.index[~nonconstant])]
        X, S = X[:, nonconstant], S[nonconstant]

    Z: NDArray
    Z = stats.zscore(X, axis=0)
    Xp = Z * S.values

    Xc = np.zeros((adata.shape[0], len(classes)))
    for c, cell_class in enumerate(classes):
        if cell_class == "Unknown":
            continue
        up_genes = markers.loc[
            (markers[marker_class_key] == cell_class) & (markers["logfoldchanges"] > 0),
            "names",
        ]
        dw_genes = markers.loc[
            (markers[marker_class_key] == cell_class) & (markers["logfoldchanges"] < 0),
            "names",
        ]
        x_up = Xp[:, S.index.isin(up_genes)]
        x_dw = Xp[:, S.index.isin(dw_genes)]
        if len(up_genes) > 0:
            Xc[:, c] += x_up.sum(axis=1) / np.sqrt(len(up_genes))
        if len(dw_genes) > 0:
            Xc[:, c] -= x_dw.sum(axis=1) / np.sqrt(len(dw_genes))

    return Xc


def classify_cells(
    adata: AnnData,
    markers: pd.DataFrame,
    marker_class_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    layer: Optional[str] = None,
    key_added: Optional[str] = None,
    threshold: float = 0.25,
    penalize_non_specific: bool = True,
    neighbors_key: Optional[str] = None,
    save_scores: bool = False,
    specificity: Literal["sctype", "observed_range"] = "sctype",
):
    """
    Classify cells based on a set of marker genes.

    Ianevski, A., Giri, A.K. & Aittokallio, T.
    Fully-automated and ultra-fast cell-type identification using specific
    marker combinations from single-cell transcriptomic data.
    Nat Commun 13, 1246 (2022).
    https://doi.org/10.1038/s41467-022-28803-w

    Parameters
    ----------
    adata
        AnnData object.
    markers
        Marker genes.
    marker_class_key
        Column in `markers` that contains the cell type information.
    cluster_key
        Column in `adata.obs` that contains the cluster information. If
        not provided, the classification will be performed on a cell by cell
        basis, pooling across neighbor cells. This pooling can be avoided by
        setting `force_pooling` to `False`.
    layer
        Layer to use for classification. Defaults to `X`.
    key_added
        Key under which to add the classification information.
    threshold
        Confidence threshold for classification. Defaults to `0.25`.
    penalize_non_specific
        Whether to penalize non-specific markers. Defaults to `True`. If
        `False`, all markers get weight 1.
    neighbors_key
        If provided, counts will be pooled across neighbor cells using the
        distances in `adata.uns[neighbors_key]["distances"]`. Defaults to `None`.
    save_scores
        Whether to save the classification scores. Defaults to `False`
    specificity
        How marker specificity weights are computed when
        `penalize_non_specific=True`. With `x` the number of cell types that
        list a gene as a marker:

        - `"sctype"` (default): the published ScType weight,
          `1 - (x - 1) / (n_types - 1)`, where `n_types` is the number of cell
          types in `markers` (excluding "Unknown"). Weights depend only on the
          marker table, not on the dataset. Genes with weight <= 0 are dropped.
        - `"observed_range"`: legacy formula
          `1 - (x - x.min()) / (x.max() - x.min())`, with `x` computed only over
          marker genes expressed in the dataset (weight 1 if all counts are
          equal). It is dataset-dependent and kept to reproduce/compare earlier
          results.

    Notes
    -----
    Expression (`layer` or `X`) must be log-normalized, since genes are
    z-scored internally; raw counts raise a `ValueError`.

    Returns
    -------
    None
        Results are written in place to `adata.obs[key_added]` (category
        dtype, `pd.NA` for calls below `threshold`) and
        `adata.obs[key_added + "_noNA"]` (best-guess label regardless of
        confidence). `key_added` defaults to `marker_class_key` when not
        given. If `save_scores=True`, also writes
        `adata.obs[key_added + "_score"]` (max per-cell confidence score) and
        `adata.obsm[key_added + "_scores"]` (full class-by-cell score
        matrix).
    """
    # cite("10.1038/s41467-022-28803-w", __package__)

    if marker_class_key is not None:
        marker_class = markers[marker_class_key]
        if not marker_class.dtype.name.startswith("category"):
            markers[marker_class_key] = marker_class.astype("category")
    else:
        col_mask = markers.dtypes == "category"
        assert col_mask.sum() == 1, (
            "markers_df must have exactly one column of type 'category'"
        )
        marker_class_key = markers.loc[:, col_mask].squeeze().name

    classes = markers[marker_class_key].cat.categories
    dtype = markers[marker_class_key].dtype
    n_types = _count_cell_types(markers, marker_class_key)

    # if doing cell by cell classification, we should pool counts to use cell
    # neighborhood information. This allows to estimate the confidence of the
    # classification. We specify pooling by providing a neighbors_key.
    posXc = _get_classification_scores_matrix(
        adata,
        markers.query("logfoldchanges > 0"),
        marker_class_key,
        neighbors_key,
        weighted_pooling=True,
        directed_pooling=True,
        layer=layer,
        penalize_non_specific=penalize_non_specific,
        specificity=specificity,
        n_types=n_types,
    )
    negXc = _get_classification_scores_matrix(
        adata,
        markers.query("logfoldchanges < 0"),
        marker_class_key,
        neighbors_key,
        weighted_pooling=True,
        directed_pooling=True,
        layer=layer,
        penalize_non_specific=penalize_non_specific,
        specificity=specificity,
        n_types=n_types,
    )
    Xc = posXc + negXc

    if cluster_key is not None:
        mappings = {}
        mappings_nona = {}
        for c in adata.obs[cluster_key].cat.categories:
            cluster_scores_matrix = Xc[adata.obs[cluster_key] == c]
            n_cells_in_cluster = cluster_scores_matrix.shape[0]

            scores = cluster_scores_matrix.sum(axis=0)
            confidence = scores.max() / n_cells_in_cluster
            if confidence >= threshold:
                mappings[c] = classes[np.argmax(scores)]
            else:
                mappings[c] = pd.NA
                logger.warning(
                    f"Cluster {str(c):>5} classified as Unknown with confidence score {confidence: 8.2f}"
                )
            mappings_nona[c] = classes[np.argmax(scores)]
        classifications = adata.obs[cluster_key].map(mappings).astype(dtype)
        classifications_nona = adata.obs[cluster_key].map(mappings_nona).astype(dtype)
    else:
        if neighbors_key is not None:
            n_neigs = adata.uns[neighbors_key]["params"]["n_neighbors"]
        else:
            n_neigs = 1
        index = adata.obs_names
        classifications = classes.values[Xc.argmax(axis=1)]
        classifications = pd.Series(classifications, index=index).astype(dtype)
        classifications_nona = classifications.copy()
        classifications.loc[Xc.max(axis=1) < threshold * n_neigs] = pd.NA

        N = len(classifications)
        n_unknowns = pd.isna(classifications).sum()
        n_estimated = N - n_unknowns

        logger.info(f"Estimated types for {n_estimated} cells ({n_estimated / N:.2%})")

    if key_added is None:
        key_added = marker_class_key

    adata.obs[key_added] = classifications
    adata.obs[key_added + "_noNA"] = classifications_nona

    if save_scores:
        adata.obs[key_added + "_score"] = Xc.max(axis=1)
        adata.obsm[key_added + "_scores"] = Xc
