import logging
import os

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, eigsh

logger = logging.getLogger(__name__)


N_CPUS = os.cpu_count()


def _whiten_dense_with_ridge(
    X: NDArray,
    ridge: float = 0.0,
    center: bool = True,
) -> NDArray:
    """
    Whiten a dense matrix in feature space with ridge regularization.

    Parameters
    ----------
    X : (n, p) ndarray
        Input data, cells x features.
    ridge : float
        Ridge parameter added to the feature variances.
    center : bool
        If True, center columns before scaling.

    Returns
    -------
    Xw : (n, p) ndarray
        Whitened data.
    """
    X = np.asarray(X, dtype=float)
    if center:
        X = X - X.mean(axis=0, keepdims=True)

    # feature-wise variance
    var = X.var(axis=0, ddof=1)
    if ridge > 0:
        var = var + ridge

    eps = np.finfo(X.dtype).eps
    scale = np.sqrt(var) + eps

    Xw = X / scale
    return Xw


def mcca_sample_space_linear(
    Xs: list[NDArray],
    *,
    n_components: int = 30,
    ridge: float = 0.0,
    center: bool = True,
    normalize: bool = False,
    tol: float = 1e-5,
    maxiter: int | None = None,
    random_state: int | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Multi-set CCA / SUMCOR-style embedding in sample space using a LinearOperator.

    Parameters
    ----------
    Xs : list of (n_i, p) ndarrays
        Each element is a view (batch) with the same number of features p.
        Must be dense for now.
    n_components : int
        Number of leading canonical components.
    ridge : float
        Ridge regularization added to per-view feature variances in whitening.
        ridge = 0.0 corresponds to standard z-scoring.
    center : bool
        Whether to center columns before whitening.
    normalize : bool
        If True, L2-normalize rows of the output embedding.
    tol : float
        Convergence tolerance passed to eigsh.
    maxiter : int or None
        Maximum number of iterations for eigsh.
    random_state : int or None
        Random seed used to initialize eigsh (via v0).

    Returns
    -------
    Z : (sum_i n_i, n_components) ndarray
        Global cell embeddings, stacked in the same order as Xs.
    eigvals : (n_components,) ndarray
        Leading eigenvalues of the MCCA operator.
    """
    m = len(Xs)
    if m < 2:
        raise ValueError("Need at least two views for MCCA.")

    # ensure dense and whiten each view with ridge regularization
    Xs_proc: list[NDArray] = []
    n_cells: list[int] = []
    n_features: list[int] = []

    for i, X in enumerate(Xs):
        if issparse(X):
            raise ValueError(
                "mcca_sample_space_linear currently expects dense arrays. "
                "Consider using a dense basis (e.g. PCA) for MCCA."
            )
        X = np.asarray(X, dtype=float)
        n_i, p_i = X.shape
        n_cells.append(n_i)
        n_features.append(p_i)
        Xw = _whiten_dense_with_ridge(X, ridge=ridge, center=center)
        Xs_proc.append(Xw)

    if len(set(n_features)) != 1:
        raise ValueError("All views must have the same number of features.")
    p = n_features[0]

    N = sum(n_cells)
    logger.info(
        f"MCCA (LinearOperator): {m} views, total cells {N}, features {p}, "
        f"ridge={ridge}, n_components={n_components}"
    )

    # cumulative offsets for splitting/stacking
    offsets = np.cumsum([0] + n_cells)

    # define matvec for the block operator A
    def matvec(v: NDArray) -> NDArray:
        if v.ndim != 1 or v.shape[0] != N:
            raise ValueError(f"v must be a 1D array of length {N}.")

        # split v into per-view pieces
        v_blocks = [v[offsets[i] : offsets[i + 1]] for i in range(m)]

        # t_j = X_j^T v_j
        # shape: (p,) for each j
        t_list = [Xs_proc[j].T @ v_blocks[j] for j in range(m)]

        # T = sum_j t_j
        T = np.sum(t_list, axis=0)  # shape (p,)

        # y_i = X_i (T - t_i) / (p - 1)
        y_blocks = []
        for i in range(m):
            y_i = Xs_proc[i] @ (T - t_list[i])
            y_blocks.append(y_i)

        y = np.concatenate(y_blocks, axis=0)
        y /= p - 1  # scale like cross-covariance
        return y

    A = LinearOperator(
        shape=(N, N),
        matvec=matvec,
        rmatvec=matvec,  # symmetric
        dtype=float,
    )

    # number of components cannot exceed N - 1
    k = min(n_components, N - 1)

    # optional deterministic initial vector
    v0 = None
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        v0 = rng.normal(size=N)
        v0 /= np.linalg.norm(v0)

    logger.info("Running eigsh on MCCA LinearOperator...")
    eigvals, eigvecs = eigsh(
        A,
        k=k,
        which="LA",  # largest algebraic
        tol=tol,
        maxiter=maxiter,
        v0=v0,
    )

    # sort in descending order
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    Z = eigvecs  # (N, k)

    if normalize:
        logger.info("Normalizing MCCA canonical variables (L2 row norm)...")
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    logger.info("MCCA (LinearOperator) complete.")
    return Z, eigvals
