import logging
import os

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import bmat as sparse_bmat
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import eigsh

from ._cca import _cross_covariance_dense, _cross_covariance_sparse

logger = logging.getLogger(__name__)


N_CPUS = os.cpu_count()


def mcca_sample_space(
    Xs: list[NDArray | csr_matrix | csc_matrix],
    *,
    n_components: int = 30,
    normalize: bool = False,
    n_jobs: int = N_CPUS,
) -> tuple[NDArray, NDArray]:
    """
    Multi-set CCA / SUMCOR-style embedding in sample space.

    Parameters
    ----------
    Xs : list of array-like, length m
        Each element Xs[i] is (n_i, p) with shared features across batches.
        Can be dense numpy arrays or csr_matrix.
    n_components : int
        Number of canonical components to compute.
    normalize : bool
        If True, L2-normalize rows of the output embedding.
    n_jobs : int
        Number of jobs to use in sparse cross-cov computation.

    Returns
    -------
    Z : (sum_i n_i, n_components) ndarray
        Global cell embeddings for all cells from all batches, concatenated
        in the same order as Xs.
    eigvals : (n_components,) ndarray
        Leading eigenvalues of the block cross-covariance matrix.
    """
    m = len(Xs)
    if m < 2:
        raise ValueError("Need at least two views for MCCA.")

    # sizes and sanity checks
    n_cells = [Xi.shape[0] for Xi in Xs]
    n_features = [Xi.shape[1] for Xi in Xs]
    if len(set(n_features)) != 1:
        raise ValueError("All views must have the same number of features.")
    p = n_features[0]

    # build pairwise cross-cov blocks
    # C_ij = cross-cov between Xs[i] and Xs[j] over features
    logger.info("Building pairwise cross-covariance blocks for MCCA...")
    C_blocks = [[None for _ in range(m)] for _ in range(m)]

    for i in range(m):
        # diagonal blocks: zeros (we use only cross-view covariances)
        C_blocks[i][i] = csr_matrix((n_cells[i], n_cells[i]))

    for i in range(m):
        for j in range(i + 1, m):
            Xi = Xs[i]
            Xj = Xs[j]
            if issparse(Xi) or issparse(Xj):
                # ensure CSR / CSC types
                if not issparse(Xi):
                    Xi = csr_matrix(Xi)
                if not issparse(Xj):
                    Xj = csr_matrix(Xj)
                C_ij = _cross_covariance_sparse(Xi.tocsr(), Xj.tocsr(), n_jobs=n_jobs)
            else:
                C_ij = _cross_covariance_dense(np.asarray(Xi), np.asarray(Xj))

            # upper and lower blocks: C_ij and C_ji = C_ij.T
            C_blocks[i][j] = csr_matrix(C_ij)
            C_blocks[j][i] = csr_matrix(C_ij.T)

    # assemble big block matrix
    logger.info("Assembling block cross-covariance matrix...")
    C_big = sparse_bmat(C_blocks, format="csr")
    N = C_big.shape[0]

    # cap n_components to N - 1
    k = min(n_components, N - 1)

    logger.info(f"Running eigendecomposition on block matrix of shape {C_big.shape}...")
    # largest algebraic eigenvalues / eigenvectors
    eigvals, eigvecs = eigsh(C_big, k=k, which="LA")

    # Sort eigenvalues descending
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    Z = eigvecs  # (N x k), one row per cell across all batches

    if normalize:
        logger.info("Normalizing multi-set canonical variables...")
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    logger.info("MCCA complete.")
    return Z, eigvals
