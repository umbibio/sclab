# Implementation Plan - Piecewise Pseudotime Scaling

Implement piecewise scaling to transform pseudotime values into a real-time scale, including automated transition detection from categorical data and support for periodic trajectories.

## Phase 1: Functional Core - Transition Detection
Implement and verify the robust quantile-based transition detection logic.

- [x] Task: Implement `find_category_transitions` in `src/sclab/tools/cellflow/pseudotime/timeseries.py`
    - [x] Implement shifting logic to handle periodicity without edge cuts.
    - [x] Implement quantile minimization loss function.
    - [x] Implement transition point calculation for sequential and periodic cases.
- [x] Task: Verify Transition Detection with Existing Tests
    - [x] Run `pytest tests/sclab/tools/cellflow/pseudotime/test_timeseries.py`.
    - [x] Ensure all test cases (periodic/non-periodic, different $t_{max}$, various offsets) pass.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Transition Detection' (Protocol in workflow.md)

## Phase 2: Functional Core - Rescaling Logic
Implement the piecewise linear transformation.

- [x] Task: Write Tests for `rescale_pseudotime`
    - [x] Create `tests/sclab/tools/cellflow/pseudotime/test_rescaling.py`.
    - [x] Define test cases for linear mapping of multiple intervals.
    - [x] Define test cases for periodic rescaling (wrapping).
- [x] Task: Implement `rescale_pseudotime` in `src/sclab/tools/cellflow/pseudotime/timeseries.py`
    - [x] Implement interval-to-interval mapping logic.
    - [x] Implement periodic wrap-around scaling.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Rescaling Logic' (Protocol in workflow.md)

## Phase 3: AnnData Integration
Expose the functionality through a user-friendly AnnData-aware API.

- [x] Task: Write Tests for AnnData Wrapper
    - [x] Add integration tests in `tests/sclab/tools/cellflow/pseudotime/test_rescaling.py`.
    - [x] Verify `adata.obs` updates and parameter handling (dictionary vs. list durations).
- [x] Task: Implement `piecewise_rescale` in `src/sclab/tools/cellflow/pseudotime/timeseries.py`
    - [x] Implement logic to handle `adata` inputs and `obs` column management.
    - [x] Wire up estimation and rescaling steps into a single workflow.
- [x] Task: Conductor - User Manual Verification 'Phase 3: AnnData Integration' (Protocol in workflow.md)
