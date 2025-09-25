import logging
import os
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy import matrix
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse import vstack as sparse_vstack
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

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
    random_state : int or None
        Passed through to the randomized SVD for reproducibility.

    Returns
    -------
    U : (n_cells(X), k) ndarray
    V : (n_cells(Y), k) ndarray
        Cell-level canonical variables.
    """
    n1, p1 = X.shape
    n2, p2 = Y.shape
    if p1 != p2:
        raise ValueError("The two matrices must have the same number of features.")

    k = n_components or min(n1, n2)

    if issparse(X):
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
    C: NDArray,
    k: int,
    svd_solver: Literal["full", "partial", "randomized"],
    random_state: int | None,
) -> tuple[NDArray, NDArray, NDArray]:
    if svd_solver == "full":
        logger.info("SVD decomposition with full SVD...")
        Uc, s, Vct = svd(C, full_matrices=False)
        Uc, s, Vct = Uc[:, :k], s[:k], Vct[:k, :]

    elif svd_solver == "partial":
        logger.info("SVD decomposition with partial SVD...")
        Uc, s, Vct = svds(C, k=k)

    elif svd_solver == "randomized":
        logger.info("SVD decomposition with randomized SVD...")
        Uc, s, Vct = randomized_svd(C, n_components=k, random_state=random_state)

    else:
        raise ValueError("svd_solver must be 'full' or 'partial'.")

    order = np.argsort(-s)
    s = s[order]
    Uc = Uc[:, order]
    Vct = Vct[order, :]

    return Uc, s, Vct


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
