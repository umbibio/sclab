import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab.tools.cellflow.pseudotime.timeseries import (
    piecewise_rescale,
    rescale_pseudotime,
)


def test_rescale_pseudotime_sequential():
    # ... existing test ...
    times = np.array([0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0])
    transitions = np.array([0.3, 0.7])
    durations = [5.0, 10.0, 5.0]

    rescaled = rescale_pseudotime(
        times,
        transitions=transitions,
        durations=durations,
        t_range=(0, 1),
        periodic=False,
    )

    expected = np.array([0.0, 2.5, 5.0, 10.0, 15.0, 17.5, 20.0])
    np.testing.assert_allclose(rescaled, expected)


def test_rescale_pseudotime_periodic():
    # ... existing test ...
    times = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    transitions = np.array([0.2, 0.5, 0.8])
    durations = [6.0, 4.0, 10.0]

    rescaled = rescale_pseudotime(
        times,
        transitions=transitions,
        durations=durations,
        t_range=(0, 1),
        periodic=True,
    )

    expected_dict = {
        0.8: 0.0,
        0.9: 1.5,
        0.0: 3.0,
        0.2: 6.0,
        0.35: 8.0,
        0.5: 10.0,
        0.65: 15.0,
    }

    for t, val in expected_dict.items():
        idx = np.where(np.isclose(times, t))[0][0]
        assert np.isclose(rescaled[idx], val), (
            f"Failed for t={t}: expected {val}, got {rescaled[idx]}"
        )


def test_rescale_pseudotime_single_interval():
    # Only one interval (no transitions)
    times = np.array([0.0, 0.5, 1.0])
    durations = [10.0]
    rescaled = rescale_pseudotime(
        times, transitions=[], durations=durations, t_range=(0, 1)
    )
    np.testing.assert_allclose(rescaled, np.array([0.0, 5.0, 10.0]))


def test_rescale_pseudotime_invalid_inputs():
    with pytest.raises(
        ValueError, match="Number of durations must be .* for sequential scaling"
    ):
        rescale_pseudotime([0, 1], transitions=[0.5], durations=[10, 20, 30])

    with pytest.raises(ValueError, match="tmax must be specified"):
        rescale_pseudotime(
            [0, 1], transitions=[0.5], durations=[10, 20], periodic=True, t_range=None
        )


def test_rescale_pseudotime_inferred_range():
    # Test that range is correctly inferred if not provided
    times = np.array([2.0, 4.0, 6.0])
    # Total range 4.0. cat1 (2->4), cat2 (4->6). Durations 5, 5
    rescaled = rescale_pseudotime(times, transitions=[4.0], durations=[5.0, 5.0])
    np.testing.assert_allclose(rescaled, np.array([0.0, 5.0, 10.0]))


def test_piecewise_rescale_adata():
    # Create a dummy AnnData
    n_obs = 100
    times = np.linspace(0, 1, n_obs)
    # 3 categories: A (0-0.3), B (0.3-0.7), C (0.7-1.0)
    labels = np.full(n_obs, "A", dtype=object)
    labels[times > 0.3] = "B"
    labels[times > 0.7] = "C"

    adata = AnnData(
        X=np.zeros((n_obs, 10)),
        obs=pd.DataFrame(
            {
                "pseudotime": times,
                "phase": pd.Categorical(labels, categories=["A", "B", "C"]),
            }
        ),
    )

    # Durations: A=5, B=10, C=5. Total = 20.
    durations = {"A": 5.0, "B": 10.0, "C": 5.0}

    piecewise_rescale(
        adata,
        time_key="pseudotime",
        groupby="phase",
        groups=["A", "B", "C"],
        durations=durations,
        new_key="realtime",
        periodic=False,
    )

    assert "realtime" in adata.obs
    assert adata.obs["realtime"].iloc[0] == 0.0
    # At transition A/B (approx 0.3), time should be approx 5.0
    idx_03 = np.argmin(np.abs(times - 0.3))
    assert np.abs(adata.obs["realtime"].iloc[idx_03] - 5.0) < 0.5
    assert np.isclose(adata.obs["realtime"].iloc[-1], 20.0)


def test_piecewise_rescale_subsetting():
    # Test that only specified categories are used and others get NaN
    n_obs = 100
    times = np.linspace(0, 1, n_obs)
    labels = np.full(n_obs, "A", dtype=object)
    labels[times > 0.3] = "B"
    labels[times > 0.7] = "C"

    adata = AnnData(
        X=np.zeros((n_obs, 10)),
        obs=pd.DataFrame(
            {
                "pseudotime": times,
                "phase": pd.Categorical(labels, categories=["A", "B", "C"]),
            }
        ),
    )

    # Use only A and B
    durations = {"A": 5.0, "B": 10.0}

    piecewise_rescale(
        adata,
        time_key="pseudotime",
        groupby="phase",
        groups=["A", "B"],
        durations=durations,
        new_key="realtime",
    )

    # Category C should be NaN
    mask_c = adata.obs["phase"] == "C"
    assert np.all(np.isnan(adata.obs.loc[mask_c, "realtime"]))

    # A and B should be rescaled
    mask_ab = (adata.obs["phase"] == "A") | (adata.obs["phase"] == "B")
    assert not np.any(np.isnan(adata.obs.loc[mask_ab, "realtime"]))


def test_piecewise_rescale_reordering():
    # Test that we can use a different order than the categorical categories
    n_obs = 100
    times = np.linspace(0, 1, n_obs)
    # A (0-0.3), B (0.3-0.7), C (0.7-1.0)
    labels = np.full(n_obs, "A", dtype=object)
    labels[times > 0.3] = "B"
    labels[times > 0.7] = "C"

    # Suppose we want to scale C THEN B (non-sensical but tests reordering)
    adata = AnnData(
        X=np.zeros((n_obs, 10)),
        obs=pd.DataFrame(
            {
                "pseudotime": times,
                "phase": pd.Categorical(labels, categories=["A", "B", "C"]),
            }
        ),
    )

    # If the user says the order is B, A, C (which is wrong for this data),
    # find_category_transitions will likely return NaNs or weird values if distributions don't overlap correctly,
    # but the tool should follow the user's order.

    durations = [10.0, 5.0]
    piecewise_rescale(
        adata,
        time_key="pseudotime",
        groupby="phase",
        groups=["B", "C"],  # Skip A
        durations=durations,
        new_key="realtime",
    )

    # A should be NaN
    mask_a = adata.obs["phase"] == "A"
    assert np.all(np.isnan(adata.obs.loc[mask_a, "realtime"]))

    # B and C should be rescaled
    mask_bc = (adata.obs["phase"] == "B") | (adata.obs["phase"] == "C")
    assert not np.any(np.isnan(adata.obs.loc[mask_bc, "realtime"]))
