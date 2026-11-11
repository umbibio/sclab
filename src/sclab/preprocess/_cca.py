import logging
import os
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy import matrix
from numpy.typing import NDArray
from scipy.linalg import lu, qr, svd
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip

logger = logging.getLogger(__name__)


N_CPUS = os.cpu_count()


def cca(
    X: NDArray | csr_matrix | csc_matrix,
    Y: NDArray | csr_matrix | csc_matrix,
    n_components=None,
    svd_solver: Literal["full", "partial", "randomized"] = "randomized",
    normalize: bool = False,
    random_state=42,
    n_jobs: int = N_CPUS,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    CCA-style integration for two single-cell matrices with unequal numbers of cells.

    Parameters
    ----------
    X, Y : array-like, shape (n_cells, n_features)
        feature-by-cell matrices with same column space (variable genes/pcs) in the same order.
    n_components : int or None
        Dimensionality of the canonical space (default = all that the smaller
        dataset allows).
    svd_solver : {'full', 'partial', 'randomized'}
        'randomized' uses Halko et al. algorithm (`sklearn.utils.extmath.randomized_svd`)
        and is strongly recommended when only the leading few components are needed.
        All solvers fix the sign of each component with ``svd_flip`` (largest
        absolute entry of each ``U`` column positive), so they agree in sign.
    random_state : int or None
        Seed of the randomized SVD, and of the initial vector of the partial
        (ARPACK) SVD, for reproducibility.

    Returns
    -------
    U : (n_cells(X), k) ndarray
    V : (n_cells(Y), k) ndarray
        Cell-level canonical variables.

    Notes
    -----
    Follows Seurat's ``RunCCA`` (Butler et al. 2018; Stuart et al. 2019): the
    canonical vectors are the leading singular vectors of the cell-by-cell
    cross-covariance ``Xc @ Yc.T / (p - 1)``. Dense inputs are z-scored per
    feature, sparse inputs are only centred. For the 'partial' and 'randomized'
    solvers that matrix is applied implicitly (:func:`_cross_covariance_operator`)
    and never formed, so memory scales with the inputs, not with ``n1 * n2``.

    References
    ----------
    Butler et al. (2018). Integrating single-cell transcriptomic data across
    different conditions, technologies, and species. *Nature Biotechnology*,
    36, 411-420. https://doi.org/10.1038/nbt.4096

    Stuart et al. (2019). Comprehensive Integration of Single-Cell Data.
    *Cell*, 177, 1888-1902.e21. https://doi.org/10.1016/j.cell.2019.05.031
    """
    n1, p1 = X.shape
    n2, p2 = Y.shape
    if p1 != p2:
        raise ValueError("The two matrices must have the same number of features.")

    k = n_components or min(n1, n2)
    if not 0 < k <= min(n1, n2):
        raise ValueError(
            f"n_components={k} must be between 1 and the smaller group size "
            f"({min(n1, n2)})"
        )
    if svd_solver == "partial" and k >= min(n1, n2):
        raise ValueError(
            f"n_components={k} must be smaller than the smaller group size "
            f"({min(n1, n2)}) for svd_solver='partial'"
        )

    if svd_solver != "full":
        C = _cross_covariance_operator(X, Y)
    elif issparse(X):  # a full SVD needs the explicit n1 x n2 matrix anyway
        C = _cross_covariance_sparse(X, Y, n_jobs=n_jobs)
    else:
        C = _cross_covariance_dense(X, Y)

    logger.info(f"Cross-covariance computed. Shape: {C.shape}")

    Uc, s, Vct = _svd_decomposition(C, k, svd_solver, random_state)

    # canonical variables
    # Left and right singular vectors are cell embeddings
    U = Uc  # (n1 x k)
    V = Vct.T  # (n2 x k)

    if normalize:
        logger.info("Normalizing canonical variables...")
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

    logger.info("Done.")

    return U, s, V


def _svd_decomposition(
    C: NDArray | LinearOperator,
    k: int,
    svd_solver: Literal["full", "partial", "randomized"],
    random_state: int | None,
) -> tuple[NDArray, NDArray, NDArray]:
    if svd_solver == "full":
        logger.info("SVD decomposition with full SVD...")
        Uc, s, Vct = svd(C, full_matrices=False)
        Uc, s, Vct = Uc[:, :k], s[:k], Vct[:k, :]
        Uc, Vct = svd_flip(Uc, Vct)  # same sign convention as 'randomized'

    elif svd_solver == "partial":
        logger.info("SVD decomposition with partial SVD...")
        Uc, s, Vct = svds(C, k=k, random_state=random_state)
        Uc, Vct = svd_flip(Uc, Vct)  # same sign convention as 'randomized'

    elif svd_solver == "randomized":
        logger.info("SVD decomposition with randomized SVD...")
        Uc, s, Vct = _randomized_svd(C, k, random_state)

    else:
        raise ValueError("svd_solver must be 'full' or 'partial'.")

    order = np.argsort(-s)
    s = s[order]
    Uc = Uc[:, order]
    Vct = Vct[order, :]

    return Uc, s, Vct


def _randomized_svd(
    C: NDArray | LinearOperator, k: int, random_state: int | None
) -> tuple[NDArray, NDArray, NDArray]:
    """``sklearn.utils.extmath.randomized_svd`` with its defaults, for operators.

    Halko, Martinsson & Tropp (2011), Finding structure with randomness.
    *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
    Ported from scikit-learn's ``randomized_svd`` / ``randomized_range_finder``
    (https://github.com/scikit-learn/scikit-learn, sklearn/utils/extmath.py,
    BSD-3-Clause, GPL-compatible), whose public function rejects
    ``LinearOperator`` input. Same random draws, power iterations (LU), and
    sign flip, so it matches ``randomized_svd(C, k, random_state=...)`` on the
    explicit matrix up to rounding.
    """
    rng = check_random_state(random_state)
    transpose = C.shape[0] < C.shape[1]
    M = C.T if transpose else C
    n_iter = 7 if k < 0.1 * min(M.shape) else 4

    Q = rng.normal(size=(M.shape[1], k + 10))  # n_oversamples=10
    if M.dtype == np.float32:
        Q = Q.astype(np.float32, copy=False)
    for _ in range(n_iter):
        Q, _ = lu(M @ Q, permute_l=True, check_finite=False)
        Q, _ = lu(M.T @ Q, permute_l=True, check_finite=False)
    Q, _ = qr(M @ Q, mode="economic", check_finite=False)

    B = (M.T @ Q).T  # Q.T @ M
    Uhat, s, Vt = svd(B, full_matrices=False)
    U = Q @ Uhat
    U, Vt = svd_flip(U, Vt, u_based_decision=not transpose)

    if transpose:
        return Vt[:k].T, s[:k], U[:, :k].T
    return U[:, :k], s[:k], Vt[:k]


def _cross_covariance_operator(X, Y) -> LinearOperator:
    """The cross-covariance of `cca` as a LinearOperator, never formed explicitly.

    Applies ``C = (X - 1 mx)(Y - 1 my).T / (p - 1)`` as ``X @ (Y.T @ v)`` plus
    rank-one mean corrections, so sparse inputs stay sparse. Matches
    `_cross_covariance_dense` (z-scored) and `_cross_covariance_sparse`
    (centred only) for dense and sparse ``X`` respectively.
    """
    p = X.shape[1]
    if p != Y.shape[1]:
        raise ValueError("The two matrices must have the same number of features.")

    if not issparse(X):
        X, Y = _dense_scale(X), _dense_scale(Y)
    mx = np.asarray(X.mean(axis=0)).reshape(1, -1)
    my = np.asarray(Y.mean(axis=0)).reshape(1, -1)

    def apply(A, ma, B, mb, V):  # (A - 1 ma) @ (B - 1 mb).T @ V
        V = V.reshape(B.shape[0], -1)
        W = B.T @ V - mb.T @ V.sum(axis=0, keepdims=True)
        return (np.asarray(A @ W) - ma @ W) / (p - 1)

    return LinearOperator(
        shape=(X.shape[0], Y.shape[0]),
        dtype=np.result_type(mx, my),
        matvec=lambda v: apply(X, mx, Y, my, v),
        rmatvec=lambda v: apply(Y, my, X, mx, v),
        matmat=lambda V: apply(X, mx, Y, my, V),
        rmatmat=lambda V: apply(Y, my, X, mx, V),
    )


def _cross_covariance_sparse(X: csr_matrix, Y: csr_matrix, n_jobs=N_CPUS) -> NDArray:
    _, p1 = X.shape
    _, p2 = Y.shape
    if p1 != p2:
        raise ValueError("The two matrices must have the same number of features.")

    p = p1

    # TODO: incorporate sparse scaling

    logger.info("Computing cross-covariance on sparse matrices...")

    mux: matrix = X.mean(axis=0)
    muy: matrix = Y.mean(axis=0)

    XYt: csr_matrix = _spmm_parallel(X, Y.T, n_jobs=n_jobs)
    Xmuyt: matrix = X.dot(muy.T)
    muxYt: matrix = Y.dot(mux.T).T
    muxmuyt: float = (mux @ muy.T)[0, 0]

    C = (XYt - Xmuyt - muxYt + muxmuyt) / (p - 1)

    return np.asarray(C)


def _cross_covariance_dense(X: NDArray, Y: NDArray) -> NDArray:
    _, p1 = X.shape
    _, p2 = Y.shape
    if p1 != p2:
        raise ValueError("The two matrices must have the same number of features.")

    p = p1

    logger.info("Computing cross-covariance on dense matrices...")
    X = _dense_scale(X)
    Y = _dense_scale(Y)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    C: NDArray = (X @ Y.T) / (p - 1)

    return C


def _dense_scale(A: NDArray) -> NDArray:
    A = np.asarray(A)
    eps = np.finfo(A.dtype).eps
    return A / (A.std(axis=0, ddof=1, keepdims=True) + eps)


def _spmm_chunk(A_csr, X, start, stop):
    return A_csr[start:stop, :] @ X


def _spmm_parallel(A_csr: csr_matrix, X_csc: csc_matrix, n_jobs=N_CPUS):
    n = A_csr.shape[0]

    bounds = np.linspace(0, n, n_jobs + 1, dtype=int)
    Ys = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_spmm_chunk)(A_csr, X_csc, bounds[i], bounds[i + 1])
        for i in range(n_jobs)
    )
    return sparse_vstack(Ys)  # result is sparse if X is sparse, dense otherwise
