"""sclab.tools.cluster.leiden: native port vs scanpy.tl.leiden(flavor="igraph")."""

import random

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab._citations import CITATIONS_KEY, UNS_KEY
from sclab.preprocess import neighbors
from sclab.tools.cluster import leiden


def make_graph(n_obs=600, seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (8, 12))
    x = centers[rng.integers(0, 8, n_obs)] + rng.normal(size=(n_obs, 12))
    adata = AnnData(np.zeros((n_obs, 1), dtype=np.float32))
    adata.obsm["X_pca"] = x.astype(np.float32)
    neighbors(adata, backend="native")
    return adata


def test_native_labels():
    adata = make_graph()
    state = random.getstate()
    leiden(adata, backend="native")
    assert random.getstate() == state  # igraph's generator is restored
    labels = adata.obs["leiden"]
    assert isinstance(labels.dtype, pd.CategoricalDtype)
    n = len(labels.cat.categories)
    assert list(labels.cat.categories) == [str(i) for i in range(n)] and n > 1
    params = adata.uns["leiden"]["params"]
    assert params == dict(resolution=1.0, random_state=0, n_iterations=2)
    assert 0 < adata.uns["leiden"]["modularity"] < 1
    assert {"leiden", "igraph"} <= set(adata.uns[UNS_KEY][CITATIONS_KEY])

    again = make_graph()
    leiden(again, backend="native")
    assert again.obs["leiden"].equals(labels)  # seeded


@pytest.mark.parametrize("resolution, seed", [(1.0, 0), (0.3, 1), (3.0, 42)])
def test_native_matches_scanpy(resolution, seed):
    sc = pytest.importorskip("scanpy")
    a = make_graph(n_obs=1500)
    b = a.copy()
    sc.tl.leiden(
        a,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=seed,
    )
    leiden(b, resolution, random_state=seed, backend="native")
    pd.testing.assert_series_equal(a.obs["leiden"], b.obs["leiden"])
    assert a.uns["leiden"]["params"] == b.uns["leiden"]["params"]
    assert a.uns["leiden"]["modularity"] == b.uns["leiden"]["modularity"]


def test_needs_a_graph():
    adata = AnnData(np.zeros((10, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="neighbor"):
        leiden(adata, backend="native")
