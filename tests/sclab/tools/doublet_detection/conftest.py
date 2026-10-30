import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

N_SINGLETS = 550
N_DOUBLETS = 25


@pytest.fixture(scope="module")
def doublet_data():
    """Three cell populations plus heterotypic doublets (mixtures of 2 populations).

    Returns (AnnData of raw counts, boolean array marking the true doublets).
    """
    rng = np.random.default_rng(0)
    G = 300
    m1 = rng.gamma(1, 1, G) * (np.arange(G) < 150)
    m2 = rng.gamma(1, 1, G) * (np.arange(G) >= 150)
    m3 = rng.gamma(1, 1, G) * (np.arange(G) % 3 == 0)

    def cells(m, n):
        return rng.poisson(m * 3, (n, G))

    X = np.vstack(
        [cells(m1, 200), cells(m2, 200), cells(m3, 150), cells(m1, 25) + cells(m2, 25)]
    ).astype(float)
    is_doublet = np.r_[np.zeros(N_SINGLETS, bool), np.ones(N_DOUBLETS, bool)]
    obs = pd.DataFrame(index=[f"c{i}" for i in range(X.shape[0])])
    return AnnData(csr_matrix(X), obs=obs), is_doublet


@pytest.fixture
def adata(doublet_data):
    return doublet_data[0].copy()
