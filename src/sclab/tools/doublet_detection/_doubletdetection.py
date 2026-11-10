from importlib.util import find_spec

import pandas as pd
from anndata import AnnData
from numpy import ndarray
from numpy.ma.core import MaskedArray

from sclab._citations import cites
from sclab.utils import check_raw_counts


def doubletdetection_is_available() -> bool:
    return find_spec("doubletdetection") is not None


@cites("doubletdetection")
def doubletdetection(
    adata: AnnData,
    layer: str = "X",
    key_added: str = "doubletdetection",
    check_counts: bool = True,
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
    """Detect doublet cells using DoubletDetection's ``BoostClassifier``.

    Adds ``{key_added}_cluster``, ``{key_added}_label`` and
    ``{key_added}_score`` to ``adata.obs``. Remaining parameters are
    forwarded to ``BoostClassifier``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    layer : str, optional
        Layer to use as the count matrix. Use ``"X"`` for ``adata.X``.
        Default is ``"X"``.
    key_added : str, optional
        Prefix for the columns added to ``adata.obs``. Default is
        ``"doubletdetection"``.
    check_counts : bool, optional
        If True, raise ``ValueError`` when the selected matrix does not
        look like raw counts (non-negative integers). Set to False to
        bypass the check, e.g. for non-integer corrected counts. Default
        is True.
    random_state : int, optional
        Seed of the PCA, the synthetic doublets and the ``"louvain"`` /
        ``"leiden"`` clustering. Default is 0. PhenoGraph (the default
        ``clustering_algorithm``) clusters with a Louvain binary that takes
        no seed, so its scores vary from run to run; use
        ``clustering_algorithm="leiden"`` (as the dashboard does) for
        reproducible results.
    """
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

    if check_counts:
        check_raw_counts(
            X,
            name=f"layer '{layer}'",
            hint=(
                "Doublet detection needs raw UMI counts; use layer='counts' "
                "(or check_counts=False for non-integer corrected counts)."
            ),
        )

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
