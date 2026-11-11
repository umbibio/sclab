"""numpy/scipy port of :func:`scanpy.tl.umap` and the parts of umap-learn it
runs, for environments without scanpy, umap-learn and numba (e.g. Pyodide).

Ported from

- scanpy 1.12.5 (https://github.com/scverse/scanpy, BSD-3-Clause, Copyright (c)
  2017 F. Alexander Wolf, P. Angerer, Theis Lab), ``src/scanpy/tools/_umap.py``
  (``umap``: defaults, epoch count, ``uns["umap"]``) and ``tools/_utils.py``
  (``_choose_representation``);
- umap-learn 0.5.12 (https://github.com/lmcinnes/umap, BSD-3-Clause, Copyright
  (c) 2017 Leland McInnes), ``umap/umap_.py`` (``find_ab_params``,
  ``make_epochs_per_sample``, ``noisy_scale_coords``,
  ``simplicial_set_embedding``), ``umap/spectral.py`` (``spectral_layout``,
  ``multi_component_layout``, ``component_layout``) and ``umap/layouts.py``
  (``optimize_layout_euclidean``);

see THIRD_PARTY_NOTICES.md. Only the options sclab uses are ported: the
Euclidean output metric, ``init_pos`` ``"spectral"`` or ``"random"``, no densMAP.

Deviation from umap-learn: its numba kernel runs the SGD edge by edge
("Hogwild"), each update seeing the positions moved by the previous ones, with
a per-vertex xorshift RNG for the negative samples. Here each epoch is one
vectorized step: the edges sampled in that epoch (the same schedule, from
``make_epochs_per_sample``) all read the positions at the start of the epoch,
their clipped attractive updates are summed, then their negative samples
(the same number per edge as umap-learn's, drawn by a numpy ``Generator``
seeded from ``random_state``) read the positions after the attractive step and their repulsive updates are
summed. The layout is deterministic for a given ``random_state`` but not the
same as umap-learn's; its quality is checked against scanpy's in
``tests/sclab/tools/embedding/test_umap_port.py``.

References: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform
Manifold Approximation and Projection for Dimension Reduction. arXiv
1802.03426. https://doi.org/10.48550/arXiv.1802.03426 -- McInnes, L., Healy,
J., Saul, N., & Großberger, L. (2018). UMAP: Uniform Manifold Approximation and
Projection. Journal of Open Source Software, 3(29), 861.
https://doi.org/10.21105/joss.00861 -- Wolf, F. A., Angerer, P., & Theis, F. J.
(2018). SCANPY. Genome Biology, 19, 15.
https://doi.org/10.1186/s13059-017-1382-0
"""

import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from anndata import AnnData
from scipy.optimize import curve_fit
from sklearn.utils import check_random_state

_N_PCS = 50  # scanpy settings.N_PCS


def umap(
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: str = "spectral",
    random_state=0,
    a: float | None = None,
    b: float | None = None,
) -> None:
    """``scanpy.tl.umap`` (scanpy 1.12.5 ``tools/_umap.py``) on the graph in
    ``adata.obsp["connectivities"]`` and ``adata.uns["neighbors"]``, with the
    vectorized layout of this module. Stores ``obsm["X_umap"]`` (float32) and
    ``uns["umap"]["params"]`` as scanpy does."""
    if "neighbors" not in adata.uns:
        raise ValueError("Did not find .uns['neighbors']. Run neighbors first.")
    if init_pos not in ("spectral", "random"):
        raise NotImplementedError("only init_pos='spectral' or 'random' is ported")
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    adata.uns["umap"] = dict(params=dict(a=a, b=b))
    if random_state != 0:
        adata.uns["umap"]["params"]["random_state"] = random_state
    random_state = check_random_state(random_state)

    params = adata.uns["neighbors"].get("params", {})
    graph = adata.obsp["connectivities"]
    n_epochs = (500 if graph.shape[0] <= 10000 else 200) if maxiter is None else maxiter
    adata.obsm["X_umap"] = simplicial_set_embedding(
        data=_choose_representation(adata, params.get("use_rep"), params.get("n_pcs")),
        graph=graph.tocoo(copy=True),
        n_components=n_components,
        initial_alpha=alpha,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init_pos,
        random_state=random_state,
        metric=params.get("metric", "euclidean"),
        metric_kwds=params.get("metric_kwds", {}),
    )


def _choose_representation(adata: AnnData, use_rep, n_pcs):
    """scanpy's ``_choose_representation``: the data the graph was built on.
    umap-learn only uses it to place disconnected components."""
    if use_rep is None and n_pcs == 0:
        use_rep = "X"
    if use_rep == "X":
        return adata.X
    if use_rep is None:
        if adata.n_vars <= _N_PCS:
            return adata.X
        use_rep = next((k for k in ("pca", "X_pca") if k in adata.obsm), None)
        if use_rep is None:  # shortcut: scanpy would run PCA; X is fine for centroids
            return adata.X
    return adata.obsm[use_rep][:, :n_pcs]


def find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """umap-learn's ``find_ab_params``: fit ``1 / (1 + a * x ** (2 * b))`` to
    an exponential decay offset by ``min_dist``."""

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def make_epochs_per_sample(weights: np.ndarray, n_epochs: int) -> np.ndarray:
    """umap-learn's ``make_epochs_per_sample``: an edge of weight ``w`` is
    sampled every ``max(weights) / w`` epochs (-1: never)."""
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


def _noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(
        np.float32
    )


def simplicial_set_embedding(
    data,
    graph,
    n_components: int,
    initial_alpha: float,
    a: float,
    b: float,
    gamma: float,
    negative_sample_rate: int,
    n_epochs: int,
    init: str,
    random_state: np.random.RandomState,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
) -> np.ndarray:
    """umap-learn's ``simplicial_set_embedding`` (``densmap=False``, Euclidean
    output): prune the edges too weak to be sampled in ``n_epochs``,
    initialize, rescale to [0, 10] and optimize."""
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    default_epochs = 500 if graph.shape[0] <= 10000 else 200
    if n_epochs is None:
        n_epochs = default_epochs
    cut = n_epochs if n_epochs > 10 else default_epochs
    graph.data[graph.data < (graph.data.max() / float(cut))] = 0.0
    graph.eliminate_zeros()

    if init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    else:  # spectral, with a little noise to avoid local minima
        embedding = spectral_layout(
            data, graph, n_components, random_state, metric, metric_kwds or {}
        )
        embedding = _noisy_scale_coords(embedding, random_state)

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
    lo, hi = np.min(embedding, 0), np.max(embedding, 0)
    embedding = (10.0 * (embedding - lo) / (hi - lo)).astype(np.float32, order="C")
    return optimize_layout_euclidean(
        embedding,
        graph.row,
        graph.col,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        random_state,
        gamma,
        initial_alpha,
        negative_sample_rate,
    )


def spectral_layout(
    data, graph, dim: int, random_state, metric="euclidean", metric_kwds=None
) -> np.ndarray:
    """umap-learn's ``spectral_layout`` (``_spectral_layout``, ``init="random"``,
    eigsh): the eigenvectors of the normalized graph Laplacian with the
    smallest nonzero eigenvalues; each connected component laid out on its
    own (:func:`_multi_component_layout`). Falls back to a random layout, with
    umap-learn's warning, when ARPACK fails."""
    n_comps, labels = scipy.sparse.csgraph.connected_components(graph)
    if n_comps > 1:
        return _multi_component_layout(
            data, graph, n_comps, labels, dim, random_state, metric, metric_kwds or {}
        )
    n = graph.shape[0]
    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    eye = scipy.sparse.identity(n, dtype=np.float64)
    d = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, n, n)
    lap = eye - d * graph * d
    k = dim + 1
    ncv = max(2 * k + 1, int(np.sqrt(n)))
    # shortcut: umap-learn switches to lobpcg above 2e6 cells; eigsh only here
    random_state.normal(size=(n, k))  # umap-learn draws an (unused) start block
    try:
        evals, evecs = scipy.sparse.linalg.eigsh(
            lap, k, which="SM", ncv=ncv, tol=1e-4, v0=np.ones(n), maxiter=n * 5
        )
    except scipy.sparse.linalg.ArpackError:
        warnings.warn(
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!",
            UserWarning,
            stacklevel=2,
        )
        return random_state.uniform(low=-10.0, high=10.0, size=(n, dim))
    order = np.argsort(evals)[1:k]
    return evecs[:, order]


def _component_layout(data, n_comps, labels, dim, random_state, metric, metric_kwds):
    """umap-learn's ``component_layout``: place the components by a spectral
    embedding of their centroids' affinities."""
    from sklearn.manifold import SpectralEmbedding
    from sklearn.metrics import pairwise_distances

    centroids = np.empty((n_comps, data.shape[1]), dtype=np.float64)
    for label in range(n_comps):
        centroids[label] = np.asarray(data[labels == label].mean(axis=0)).ravel()
    distance_matrix = pairwise_distances(centroids, metric=metric, **metric_kwds)
    affinity_matrix = np.exp(-(distance_matrix**2))
    embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed", random_state=random_state
    ).fit_transform(affinity_matrix)
    embedding /= embedding.max()
    return embedding


def _multi_component_layout(
    data, graph, n_comps, labels, dim, random_state, metric, metric_kwds
):
    """umap-learn's ``multi_component_layout``: a spectral layout per
    connected component (random for tiny ones), scaled to half the distance
    to the nearest other component and moved to its place in a meta layout."""
    from sklearn.metrics import pairwise_distances

    result = np.empty((graph.shape[0], dim), dtype=np.float32)
    if n_comps > 2 * dim:
        meta = _component_layout(
            data, n_comps, labels, dim, random_state, metric, metric_kwds
        )
    else:
        k = int(np.ceil(n_comps / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta = np.vstack([base, -base])[:n_comps]

    graph = graph.tocsr()
    for label in range(n_comps):
        mask = labels == label
        component_graph = graph[mask][:, mask].tocoo()
        distances = pairwise_distances([meta[label]], meta)
        data_range = distances[distances > 0.0].min() / 2.0
        size = component_graph.shape[0]
        if size < 2 * dim or size <= dim + 1:
            result[mask] = (
                random_state.uniform(low=-data_range, high=data_range, size=(size, dim))
                + meta[label]
            )
        else:
            embedding = spectral_layout(None, component_graph, dim, random_state)
            embedding *= data_range / np.max(np.abs(embedding))
            result[mask] = embedding + meta[label]
    return result


def optimize_layout_euclidean(
    embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    n_epochs: int,
    n_vertices: int,
    epochs_per_sample: np.ndarray,
    a: float,
    b: float,
    random_state: np.random.RandomState,
    gamma: float = 1.0,
    initial_alpha: float = 1.0,
    negative_sample_rate: float = 5.0,
) -> np.ndarray:
    """umap-learn's ``optimize_layout_euclidean`` (``move_other=True``,
    ``embedding`` both head and tail), one vectorized step per epoch (see the
    module docstring). Modifies and returns ``embedding``.

    The gradients, clipping to [-4, 4], learning-rate decay and sampling
    schedule (positive and negative) are umap-learn's ``layouts.py``.
    """
    # one contiguous float32 array per dimension: gathers by column are much
    # faster than row gathers on an (n, dim) array
    cols = [
        np.ascontiguousarray(embedding[:, d], dtype=np.float32)
        for d in range(embedding.shape[1])
    ]
    f32 = np.float32  # float32 scalars keep the float32 arrays from upcasting
    a, b, gamma = f32(a), f32(b), f32(gamma)
    tiny = np.finfo(f32).tiny  # d2 = 0 means diff = 0: the update is 0 either way
    # umap-learn seeds its per-vertex RNG with 3 int32 draws; here a Generator
    rng = np.random.default_rng(random_state.randint(np.iinfo(np.int32).max, size=3))
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    alpha = initial_alpha

    def diffs(j, k):
        diff = [c.take(j) - c.take(k) for c in cols]
        return diff, sum(dx * dx for dx in diff)

    def move(diff, coef, alpha, j, k=None):
        for c, dx in zip(cols, diff):
            grad = dx * coef
            np.clip(grad, -4.0, 4.0, out=grad)
            grad *= f32(alpha)
            step = np.bincount(j, weights=grad, minlength=n_vertices)
            if k is not None:  # move_other
                step -= np.bincount(k, weights=grad, minlength=n_vertices)
            c += step

    for n in range(n_epochs):
        active = np.flatnonzero(epoch_of_next_sample <= n)
        if active.size:
            j, k = head[active], tail[active]
            # attraction along the sampled edges: both ends move
            diff, d2 = diffs(j, k)
            d2 = np.maximum(d2, tiny)
            p = d2**b
            coef = f32(-2.0) * a * b * (p / d2) / (a * p + f32(1.0))
            move(diff, coef, alpha, j, k)
            epoch_of_next_sample[active] += epochs_per_sample[active]

            # repulsion from random vertices: only the head moves
            n_neg = np.trunc(
                (n - epoch_of_next_negative_sample[active])
                / epochs_per_negative_sample[active]
            )
            jn = np.repeat(j, np.maximum(n_neg, 0).astype(np.intp))  # int32 on wasm32
            diff, d2 = diffs(jn, rng.integers(n_vertices, size=jn.size, dtype=np.intp))
            coef = f32(2.0) * gamma * b / ((f32(0.001) + d2) * (a * d2**b + f32(1.0)))
            move(diff, coef, alpha, jn)
            epoch_of_next_negative_sample[active] += (
                n_neg * epochs_per_negative_sample[active]
            )
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    for d, c in enumerate(cols):
        embedding[:, d] = c
    return embedding
