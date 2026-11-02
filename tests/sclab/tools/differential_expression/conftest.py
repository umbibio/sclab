import random

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix


def _make_counts_adata(n_genes: int) -> AnnData:
    rng = np.random.default_rng(0)
    n_per = 45
    cond = np.repeat(["A", "B"], 2 * n_per)
    ctype = np.tile(np.repeat(["x", "y"], n_per), 2)
    n = cond.size
    mu = np.full((n, n_genes), 5.0)
    mu[cond == "A", 0] = 20.0
    mu[cond == "A", 1] = 20.0
    X = rng.poisson(mu).astype(float)
    obs = pd.DataFrame(
        {"cond": cond, "ctype": ctype}, index=[f"c{i}" for i in range(n)]
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = AnnData(csr_matrix(X), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata


@pytest.fixture
def counts_adata() -> AnnData:
    """Two conditions x two cell types, 6 genes; genes 0-1 up in condition A."""
    return _make_counts_adata(6)


@pytest.fixture
def r_counts_adata() -> AnnData:
    """Same design with 200 genes, for the real R fits.

    TMM assumes most genes are not DE; with only 6 genes the two DE genes
    shift the normalization and make null genes look down-regulated.
    """
    random.seed(0)  # aggregate_and_filter shuffles cells into replicas
    return _make_counts_adata(200)


@pytest.fixture
def r_three_group_adata() -> AnnData:
    """Three conditions, 200 genes; genes 0-1 up in A.

    With one replica per group, "A vs rest" has three samples and two
    coefficients, so one residual degree of freedom remains.
    """
    random.seed(0)
    rng = np.random.default_rng(0)
    cond = np.repeat(["A", "B", "C"], 45)
    mu = np.full((cond.size, 200), 5.0)
    mu[cond == "A", :2] = 20.0
    obs = pd.DataFrame({"cond": cond}, index=[f"c{i}" for i in range(cond.size)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(200)])
    adata = AnnData(csr_matrix(rng.poisson(mu).astype(float)), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata
