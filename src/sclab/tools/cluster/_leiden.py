"""Leiden clustering of the neighbor graph, with scanpy or sclab's port of
``scanpy.tl.leiden(flavor="igraph")``.

The port follows scanpy 1.12.5 (https://github.com/scverse/scanpy,
BSD-3-Clause, Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab; see
THIRD_PARTY_NOTICES.md), files ``src/scanpy/tools/_leiden.py`` (igraph
flavor), ``src/scanpy/_utils/__init__.py`` (``get_igraph_from_adjacency``) and
``src/scanpy/_utils/random.py`` (``_RNGIgraph``, ``set_igraph_random_state``),
so the same graph, arguments and seeding give the same labels. The clustering
itself is igraph's ``community_leiden`` (Traag, Waltman & van Eck 2019).
"""

import random
from contextlib import contextmanager

import numpy as np
import pandas as pd
from anndata import AnnData

from ..._citations import record_citations
from ...preprocess._scanpy_port import resolve_backend


def leiden(
    adata: AnnData,
    resolution: float = 1.0,
    *,
    n_iterations: int = 2,
    random_state: int = 0,
    key_added: str = "leiden",
    backend: str = "auto",
) -> None:
    """Leiden clustering of the neighbor graph (modularity, igraph).

    Runs :func:`scanpy.tl.leiden` with ``flavor="igraph"``,
    ``directed=False`` and the connectivities as edge weights.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a neighbor graph (see
        :func:`sclab.preprocess.neighbors`). Modified in-place.
    resolution : float, optional
        Higher values give more clusters. Default is 1.0.
    n_iterations : int, optional
        Iterations of the Leiden algorithm; negative: until stable. Default
        is 2.
    random_state : int, optional
        Seed of igraph's random number generator. Default is 0.
    key_added : str, optional
        ``adata.obs`` column of the labels and ``adata.uns`` key of the
        parameters. Default is ``"leiden"``.
    backend : {"auto", "native", "scanpy"}, optional
        ``"scanpy"`` runs :func:`scanpy.tl.leiden`, ``"native"`` sclab's port
        of it (the same igraph graph, ``community_leiden`` call and seeding,
        so the same labels), ``"auto"`` (default) scanpy if it is installed.

    Returns
    -------
    None
        Stores the labels as a categorical ``adata.obs[key_added]`` ("0",
        "1", ...) and ``adata.uns[key_added]`` (``params``, ``modularity``).

    References
    ----------
    Traag, Waltman & van Eck (2019). From Louvain to Leiden: guaranteeing
    well-connected communities. *Scientific Reports*, 9, 5233.
    https://doi.org/10.1038/s41598-019-41695-z

    Csardi & Nepusz (2006). The igraph software package for complex network
    research. *InterJournal, Complex Systems*, 1695. https://igraph.org

    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0
    """
    if resolve_backend(backend) == "scanpy":
        import scanpy as sc

        sc.tl.leiden(
            adata,
            resolution=resolution,
            flavor="igraph",
            n_iterations=n_iterations,
            directed=False,
            random_state=random_state,
            key_added=key_added,
        )
    else:
        _leiden_igraph(adata, resolution, n_iterations, random_state, key_added)
    record_citations(adata, "scanpy", "leiden", "igraph")


def _leiden_igraph(
    adata: AnnData,
    resolution: float,
    n_iterations: int,
    random_state: int,
    key_added: str,
) -> None:
    """``scanpy.tl.leiden(adata, resolution, flavor="igraph", n_iterations,
    directed=False, random_state, key_added)``."""
    import igraph as ig

    # scanpy's _choose_graph: the connectivities of uns["neighbors"]
    if "neighbors" not in adata.uns:
        raise ValueError("You need to run the neighbors first (no neighbor graph).")
    key = adata.uns["neighbors"].get("connectivities_key", "connectivities")
    adjacency = adata.obsp[key]

    # get_igraph_from_adjacency(adjacency, directed=False): an edge per
    # stored nonzero (both directions), in CSR order
    coo = adjacency.tocoo()
    nonzero = coo.data != 0
    sources, targets = coo.row[nonzero], coo.col[nonzero]
    g = ig.Graph(directed=False)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources.tolist(), targets.tolist())))
    g.es["weight"] = coo.data[nonzero].tolist()

    with _igraph_random_state(random_state):
        part = g.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=resolution,
            n_iterations=n_iterations,
        )

    groups = np.array(part.membership)
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=[str(c) for c in np.unique(groups)],  # natsorted integers
    )
    adata.uns[key_added] = dict(
        params=dict(
            resolution=resolution,
            random_state=random_state,
            n_iterations=n_iterations,
        ),
        modularity=part.modularity,
    )


class _RNGIgraph:
    """scanpy's ``_RNGIgraph``: a numpy ``RandomState`` as igraph's random
    number generator, so the global ``random`` seed is not changed."""

    def __init__(self, random_state: int = 0) -> None:
        from sklearn.utils import check_random_state

        self._rng = check_random_state(random_state)

    def getrandbits(self, k: int) -> int:
        return self._rng.tomaxint() & ((1 << k) - 1)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b + 1)

    def __getattr__(self, attr: str):
        return getattr(self._rng, "normal" if attr == "gauss" else attr)


@contextmanager
def _igraph_random_state(random_state: int):
    """scanpy's ``set_igraph_random_state``."""
    import igraph

    igraph.set_random_number_generator(_RNGIgraph(random_state))
    try:
        yield
    finally:
        igraph.set_random_number_generator(random)
