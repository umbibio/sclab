import threading
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any

import pandas as pd
from anndata import AnnData
from numpy import ndarray

from sclab._citations import cites
from sclab._compat import require
from sclab.utils import check_raw_counts


def scrublet_is_available() -> bool:
    return find_spec("scrublet") is not None


_annoy_lock = threading.Lock()


@contextmanager
def _single_threaded_annoy():
    """Make scrublet build its annoy index on one thread.

    annoy (>= 1.17) builds its trees on all CPUs by default, and scrublet
    cannot change that: even with the seed scrublet sets, the approximate
    neighbors, hence the doublet scores, then vary from run to run. One thread
    is reproducible. ``annoy.AnnoyIndex`` is swapped only during the call,
    under a lock (scrublet imports it when it builds the index).
    """
    if find_spec("annoy") is None:  # scrublet then uses exact neighbors
        yield
        return
    import annoy

    class AnnoyIndex(annoy.AnnoyIndex):
        def build(self, n_trees, n_jobs=-1):
            return super().build(n_trees, n_jobs=1)

    with _annoy_lock:
        original, annoy.AnnoyIndex = annoy.AnnoyIndex, AnnoyIndex
        try:
            yield
        finally:
            annoy.AnnoyIndex = original


@cites("scrublet")
def scrublet(
    adata: AnnData,
    layer: str = "X",
    key_added: str = "scrublet",
    check_counts: bool = True,
    total_counts: ndarray | None = None,
    sim_doublet_ratio: float = 2.0,
    n_neighbors: int = None,
    expected_doublet_rate: float = 0.1,
    stdev_doublet_rate: float = 0.02,
    random_state: int = 0,
    scrub_doublets_kwargs: dict[str, Any] = dict(  # noqa: B006
        synthetic_doublet_umi_subsampling=1.0,
        use_approx_neighbors=True,
        distance_metric="euclidean",
        get_doublet_neighbor_parents=False,
        min_counts=3,
        min_cells=3,
        min_gene_variability_pctl=85,
        log_transform=False,
        mean_center=True,
        normalize_variance=True,
        n_prin_comps=30,
        svd_solver="arpack",
        verbose=True,
    ),
):
    """Detect doublet cells using Scrublet.

    Simulates synthetic doublets from the observed count matrix and uses
    a k-NN classifier to assign each cell a doublet score. Cells are then
    labelled as ``"singlet"`` or ``"doublet"``.

    Requires ``scrublet`` to be installed (``pip install scrublet``).
    Without ``annoy`` (e.g. under Pyodide, where scrublet is installed with
    ``micropip.install('scrublet', deps=False)``) the neighbors are exact
    (``use_approx_neighbors=False``) instead of approximate; scrublet's
    umap-learn and numba dependencies are not used here.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    layer : str, optional
        Layer to use as the count matrix. Use ``"X"`` for ``adata.X``.
        Default is ``"X"``.
    key_added : str, optional
        Prefix for the columns added to ``adata.obs``. Results are stored
        as ``{key_added}_score`` and ``{key_added}_label``. Default is
        ``"scrublet"``.
    check_counts : bool, optional
        If True, raise ``ValueError`` when the selected matrix does not
        look like raw counts (non-negative integers). Set to False to
        bypass the check, e.g. for non-integer corrected counts. Default
        is True.
    total_counts : ndarray or None, optional
        Pre-computed per-cell total counts. If None, Scrublet computes
        them internally. Default is None.
    sim_doublet_ratio : float, optional
        Number of synthetic doublets to simulate relative to the number of
        observed cells. Default is 2.0.
    n_neighbors : int or None, optional
        Number of neighbors used to classify doublets. If None, Scrublet
        uses a heuristic based on the number of cells. Default is None.
    expected_doublet_rate : float, optional
        Expected fraction of doublets in the dataset. Default is 0.1.
    stdev_doublet_rate : float, optional
        Uncertainty in the expected doublet rate. Default is 0.02.
    random_state : int, optional
        Random seed for reproducibility. Default is 0. The approximate
        (annoy) neighbor index is built on a single thread so that the
        scores are reproducible.
    scrub_doublets_kwargs : dict, optional
        Additional keyword arguments forwarded to
        :meth:`scrublet.Scrublet.scrub_doublets`.

    Returns
    -------
    None
        Adds the following columns to ``adata.obs``:

        - ``{key_added}_score`` (float): Doublet score for each cell.
        - ``{key_added}_label`` (Categorical): ``"singlet"`` or
          ``"doublet"``.
    """
    require("scrublet")
    from scrublet import Scrublet  # noqa: E402

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

    scrub = Scrublet(
        counts_matrix=X,
        total_counts=total_counts,
        sim_doublet_ratio=sim_doublet_ratio,
        n_neighbors=n_neighbors,
        expected_doublet_rate=expected_doublet_rate,
        stdev_doublet_rate=stdev_doublet_rate,
        random_state=random_state,
    )

    kwargs = dict(scrub_doublets_kwargs)
    if find_spec("annoy") is None:  # e.g. Pyodide; scrublet would print a warning
        kwargs["use_approx_neighbors"] = False  # exact: sklearn NearestNeighbors
    with _single_threaded_annoy():
        _scores, labels = scrub.scrub_doublets(**kwargs)
    if labels is not None:
        _labels = list(map(lambda v: "doublet" if v else "singlet", labels))
        _labels = pd.Categorical(_labels, ["singlet", "doublet"])
        adata.obs[f"{key_added}_label"] = _labels
    else:
        adata.obs[f"{key_added}_label"] = "singlet"

    adata.obs[f"{key_added}_score"] = _scores
