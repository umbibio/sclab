# Track Specification: Piecewise Pseudotime Scaling

## Overview
Implement a robust mechanism to transform pseudotime values (typically $0$ to $100\%$ or $0$ to $t_{max}$) into a real-time scale based on known durations of biological or experimental intervals. This involves identifying transition points in pseudotime—either through explicit definition or automated estimation from categorical labels—and applying a piecewise linear transformation.

## Functional Requirements

### 1. Transition Point Estimation (`find_category_transitions`)
- **Input**: Pseudotime values, categorical labels, an ordered list of categories, and a periodicity flag.
- **Logic**: 
    - Implement the quantile-based intersection logic from legacy code.
    - For each pair of adjacent categories (including the wrap-around for periodic data), find the pseudotime point $t_i$ that minimizes the distance between specific quantiles of the two distributions.
    - Handle periodicity by shifting distributions to avoid "cuts" at the $0/t_{max}$ boundary.
- **Output**: A sorted list of transition points.

### 2. Piecewise Scaling Logic (`rescale_pseudotime`)
- **Input**: Pseudotime values, $t_{range}$ (e.g., $(0, 1)$), a list of transition points, and real-time durations (as a list or dictionary).
- **Logic**:
    - Map pseudotime intervals defined by the transitions to real-time intervals defined by durations.
    - Perform linear interpolation within each interval.
    - Handle periodic wrapping (if $t$ starts at $80\%$ and ends at $20\%$ across the boundary).
- **Output**: NumPy array of rescaled real-time values.

### 3. AnnData Wrapper (`piecewise_rescale`)
- **Input**: `adata` object, `time_key`, `groupby` (categorical column), `groups` (explicit list of labels), `durations`, and `periodic`.
- **Logic**: 
    - Validate that `groups` is provided; do not infer from `adata.obs[groupby]`.
    - Filter cells to only those belonging to `groups`.
    - Extract values, call the functional methods, and store the result in `adata.obs`. 
    - Cells not in `groups` should be assigned `NaN` in the output column.
    - Allow users to pass explicit transitions or trigger the estimation.

## Non-Functional Requirements
- **Precision**: Transition points should be computed with high precision (e.g., 12 decimal places).
- **Robustness**: Warn users if there is no overlap between phases during estimation.
- **TDD**: Ensure high test coverage using the established `pytest` suite.

## Acceptance Criteria
- [ ] `find_category_transitions` passes existing tests in `test_timeseries.py`.
- [ ] Rescaled values correctly reflect the provided real-time durations.
- [ ] Periodic trajectories correctly map across the $t_{max}$ boundary.
- [ ] AnnData wrapper successfully updates `obs` with the new scale.

## Out of Scope
- Automatic ordering of categories (the user must provide the sequence).
- Non-linear scaling within intervals (e.g., spline-based).
