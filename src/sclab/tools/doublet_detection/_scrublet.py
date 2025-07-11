from importlib.util import find_spec
from typing import Any

import pandas as pd
from anndata import AnnData
from numpy import ndarray


def scrublet(
    adata: AnnData,
    layer: str = "X",
    key_added: str = "scrublet",
    total_counts: ndarray | None = None,
    sim_doublet_ratio: float = 2.0,
    n_neighbors: int = None,
    expected_doublet_rate: float = 0.1,
    stdev_doublet_rate: float = 0.02,
    random_state: int = 0,
    scrub_doublets_kwargs: dict[str, Any] = dict(
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
    if find_spec("scrublet") is None:
        raise ImportError(
            "scrublet is not installed. Install with:\npip install scrublet"
        )
    from scrublet import Scrublet  # noqa: E402

    if layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]

    scrub = Scrublet(
        counts_matrix=X,
        total_counts=total_counts,
        sim_doublet_ratio=sim_doublet_ratio,
        n_neighbors=n_neighbors,
        expected_doublet_rate=expected_doublet_rate,
        stdev_doublet_rate=stdev_doublet_rate,
        random_state=random_state,
    )

    _scores, labels = scrub.scrub_doublets(**scrub_doublets_kwargs)
    if labels is not None:
        _labels = list(map(lambda v: "doublet" if v else "singlet", labels))
        _labels = pd.Categorical(_labels, ["singlet", "doublet"])
        adata.obs[f"{key_added}_label"] = _labels
    else:
        adata.obs[f"{key_added}_label"] = "singlet"

    adata.obs[f"{key_added}_score"] = _scores
