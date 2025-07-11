import logging
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csc_matrix, csr_matrix, issparse

from ...preprocess import pool_neighbors

logger = logging.getLogger(__name__)


def _get_classification_scores_matrix(
    adata: AnnData,
    markers: pd.DataFrame,
    marker_class_key: str,
    neighbors_key: Optional[str] = None,
    weighted_pooling: bool = False,
    directed_pooling: bool = True,
    layer: Optional[str] = None,
    penalize_non_specific: bool = True,
):
    # Ianevski, A., Giri, A.K. & Aittokallio, T.
    # Fully-automated and ultra-fast cell-type identification using specific
    # marker combinations from single-cell transcriptomic data.
    # Nat Commun 13, 1246 (2022).
    # https://doi.org/10.1038/s41467-022-28803-w

    if layer is not None:
        X = adata.layers[layer]

    else:
        X = adata.X

    min_val: np.number = X.min()
    M = X > min_val
    n_cells = np.asarray(M.sum(axis=0)).squeeze()
    mask = n_cells > 5
    print(f"using {mask.sum()} genes")

    markers = markers.loc[markers["names"].isin(adata.var_names[mask])].copy()
    classes = markers[marker_class_key].cat.categories

    x = markers[[marker_class_key, "names"]].groupby("names").count()[marker_class_key]
    if penalize_non_specific:
        S = 1.0 - (x - x.min()) / (x.max() - x.min())
        S = S[S > 0]
    else:
        S = x * 0.0 + 1.0

    X: NDArray | csr_matrix | csc_matrix
    if neighbors_key is not None:
        X = pool_neighbors(
            adata[:, S.index],
            layer=layer,
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
        Whether to penalize non-specific markers. Defaults to `True`.
    neighbors_key
        If provided, counts will be pooled across neighbor cells using the
        distances in `adata.uns[neighbors_key]["distances"]`. Defaults to `None`.
    save_scores
        Whether to save the classification scores. Defaults to `False`
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
