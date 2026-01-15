from importlib.util import find_spec

import pandas as pd
from anndata import AnnData
from numpy import ndarray
from numpy.ma.core import MaskedArray


def doubletdetection_is_available() -> bool:
    return find_spec("doubletdetection") is not None


def doubletdetection(
    adata: AnnData,
    layer: str = "X",
    key_added: str = "doubletdetection",
    boost_rate=0.25,
    n_components=30,
    n_top_var_genes=10000,
    replace=False,
    clustering_algorithm="phenograph",
    clustering_kwargs=None,
    n_iters=10,
    normalizer=None,
    pseudocount=0.1,
    random_state=0,
    verbose=False,
    standard_scaling=False,
    n_jobs=1,
) -> None:
    from doubletdetection.doubletdetection import BoostClassifier

    clf = BoostClassifier(
        boost_rate=boost_rate,
        n_components=n_components,
        n_top_var_genes=n_top_var_genes,
        replace=replace,
        clustering_algorithm=clustering_algorithm,
        clustering_kwargs=clustering_kwargs,
        n_iters=n_iters,
        normalizer=normalizer,
        pseudocount=pseudocount,
        random_state=random_state,
        verbose=verbose,
        standard_scaling=standard_scaling,
        n_jobs=n_jobs,
    )

    if layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]

    # raw_counts is a cells by genes count matrix
    labels: ndarray = clf.fit(X).predict()
    # higher means more likely to be doublet
    scores: MaskedArray = clf.doublet_score()

    _labels = list(map(lambda v: "doublet" if v else "singlet", labels))
    _scores: ndarray = scores.data

    clusters: ndarray = clf.communities_[-1]
    _clusters = pd.Categorical(clusters.astype(int))
    categories: pd.Index = _clusters.categories
    _clusters = _clusters.rename_categories(categories.astype(str))

    adata.obs[f"{key_added}_cluster"] = _clusters
    adata.obs[f"{key_added}_label"] = pd.Categorical(_labels, ["singlet", "doublet"])
    adata.obs[f"{key_added}_score"] = _scores
