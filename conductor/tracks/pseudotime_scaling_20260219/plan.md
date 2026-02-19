# Implementation Plan - Piecewise Pseudotime Scaling

Implement piecewise scaling to transform pseudotime values into a real-time scale, including automated transition detection from categorical data and support for periodic trajectories.

## Phase 1: Functional Core - Transition Detection
Implement and verify the robust quantile-based transition detection logic.

- [ ] Task: Implement `find_category_transitions` in `src/sclab/tools/cellflow/pseudotime/timeseries.py`
    - [ ] Implement shifting logic to handle periodicity without edge cuts.
    - [ ] Implement quantile minimization loss function.
    - [ ] Implement transition point calculation for sequential and periodic cases.
- [ ] Task: Verify Transition Detection with Existing Tests
    - [ ] Run `pytest tests/sclab/tools/cellflow/pseudotime/test_timeseries.py`.
    - [ ] Ensure all test cases (periodic/non-periodic, different $t_{max}$, various offsets) pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Transition Detection' (Protocol in workflow.md)

## Phase 2: Functional Core - Rescaling Logic
Implement the piecewise linear transformation.

- [ ] Task: Write Tests for `rescale_pseudotime`
    - [ ] Create `tests/sclab/tools/cellflow/pseudotime/test_rescaling.py`.
    - [ ] Define test cases for linear mapping of multiple intervals.
    - [ ] Define test cases for periodic rescaling (wrapping).
- [ ] Task: Implement `rescale_pseudotime` in `src/sclab/tools/cellflow/pseudotime/timeseries.py`
    - [ ] Implement interval-to-interval mapping logic.
    - [ ] Implement periodic wrap-around scaling.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Rescaling Logic' (Protocol in workflow.md)

## Phase 3: AnnData Integration
Expose the functionality through a user-friendly AnnData-aware API.

- [ ] Task: Write Tests for AnnData Wrapper
    - [ ] Add integration tests in `tests/sclab/tools/cellflow/pseudotime/test_rescaling.py`.
    - [ ] Verify `adata.obs` updates and parameter handling (dictionary vs. list durations).
- [ ] Task: Implement `piecewise_rescale` in `src/sclab/tools/cellflow/pseudotime/timeseries.py`
    - [ ] Implement logic to handle `adata` inputs and `obs` column management.
    - [ ] Wire up estimation and rescaling steps into a single workflow.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: AnnData Integration' (Protocol in workflow.md)
