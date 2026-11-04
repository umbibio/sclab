# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Dashboard "Emphasize values" (on by default): when coloring by a number, cells with zero/negative/missing values shrink (Settings "Min. size", default 0.3) and the rest grow with their percentile; "Size by" and selections take precedence. Works in plotly, jscatter and static plots
- Dashboard notice in 3D when opacity is below 1 (plotly then draws by trace order, not depth)
- Dashboard Settings tab: choose the 2D plot backend (`plotly`, `jscatter` or `static` matplotlib image) and the 3D point limit; also `SCLabDashboard(plot_backend=...)` / `plotter.backend`. 1D/3D always use plotly; path drawing, custom figures and extra traces fall back to plotly with a notice
- `jscatter` optional extra (`jupyter-scatter`) for the jscatter backend: color/size/opacity, selection, lasso (shape recorded), density contours; pinned views restore settings but not zoom
- Workspace "Merge..." dialog and `Workspace.merge()`: a preview (shared/dropped genes, Ensembl IDs vs symbols, counts-layer check, obs conflicts, per-dataset median counts/genes per cell) blocks unsafe merges; the merged dataset starts from raw counts with the label column as its batch key, originals stay unchanged, and "Show code" gives a script covering every parent, the merge and later steps
- `sclab.preprocess.merge_preview()`, `merge_datasets()` (optional `equalize_depth` binomial thinning of deeper datasets to the shallowest median depth) and `downsample_counts()`
- `sclab.Workspace`: several named datasets with a switcher and one active dashboard. File datasets are read only when activated; edited ones are written to `workspace_dir` when you switch away (switching away without one is refused), so history and pinned views survive
- `lazy` optional extra (`xarray`, `dask`): the Workspace lists `.h5ad` datasets with `anndata.experimental.read_lazy` when installed (falls back to a backed read)
- `SCLabDashboard.close()` closes the dashboard and all its widgets; `SCLabDashboard(source=...)` sets the generated script's load line
- Pinned plot views: a strip under the plot saves and restores plot settings plus zoom / 3D camera (one redraw), with badges for missing data or data changed since the pin; stored in `adata.uns["sclab"]["views"]`
- History tab can show, copy and download a Python script reproducing the applied steps (`SCLabDashboard.generate_code()`), including Keep/Drop/Label selections as filter expressions, box bounds and lasso polygons
- `sclab.tools.select_box()` / `select_polygon()` to select cells inside a box or polygon on an embedding or two obs columns
- `sclab.preprocess.neighbors()` and `sclab.tools.embedding.umap()` (used by the Neighbors/UMAP steps); `random_state` on `sclab.preprocess.pca()`
- Dashboard History tab listing applied steps; the history is also stored in `adata.uns["sclab"]["history"]` and shown when a dashboard is built from an AnnData that has one
- `pseudobulk_pylimma()`, a pure-Python limma-voom pseudobulk DE method (pylimma, optional edgePython TMM preprocessing) that matches R limma results
- `de` optional extra (`edgepython`, `pylimma`); `xlsxwriter` is now a core dependency
- `check_raw_counts()` / `looks_like_raw_counts()` input validators in `sclab.utils`
- `check_counts` parameter on `scrublet()`, `doubletdetection()` and `scdblfinder()` (default `True`) to reject non-count input
- `specificity` parameter on sctype `classify_cells()`: `"sctype"` (default, published ScType weights) or `"observed_range"` (previous dataset-dependent weights, kept for comparison)
- Doublet Detection step "Override layer" checkbox to use a layer other than the dataset's counts layer
- Expanded test suite, including headless dashboard/widget tests
- `max_distance` parameter on `transfer_metadata()` and `propagate_metadata()`: neighbors at distance `>= max_distance` are ignored and do not count toward `min_neighs`
- `counts_layer` parameter on `sclab.preprocess.pca()` (default `"counts"`), used to recompute highly variable genes on the reference batch
- `sclab.utils.layer_keys()`: layer names of an AnnData without the `None` key that anndata>=0.13 uses for `X`
- Dashboard "3D full resolution" checkbox (saved in pinned views)
- `check_counts` parameter on `pseudobulk_pylimma()`, `pseudobulk_limma()` and `pseudobulk_edger()` (default `True`) to reject non-count input

### Changed
- Dashboard plots are faster and send less data: recolor, size and selection patch the existing traces (numpy arrays sent as binary); size/selection changes are near-instant and recolors 2–6× faster at 100k–1M cells
- Density contours use a binned Gaussian KDE: under 2 s at 1M cells instead of minutes, within a few percent of the previous contours
- 3D plots show a seeded random subsample of at most 200,000 cells by default (selected cells first, noted in the title); traces are split at 500,000 points and the camera is kept across redraws on the same embedding
- 3D plots no longer sort cells by a numeric color (depth decides what is in front); a numeric recolor sends only colors and sizes (6.1 MB → 1.6 MB at 200k cells)
- Plot traces no longer carry per-cell hover text; cell ids come from an index map (hover values are float32)
- **Breaking:** remove `batch_key` from `pseudobulk_edger()` and `pseudobulk_limma()`
- **Breaking:** `pseudobulk_pylimma()`, `pseudobulk_limma()` and `pseudobulk_edger()` default to `layer="counts"` (pass `layer="X"` to use `adata.X`; `layer=None` raises)
- **Breaking:** `pydeseq2` is no longer a core dependency (it was unused)
- R `pseudobulk_limma()` / `pseudobulk_edger()` drop the non-estimable `replica` term from the design when there is a single replica, as `pseudobulk_pylimma()` does
- Dashboard PCA step delegates to `sclab.preprocess.pca()`
- Support current anndata (0.13), plotly (6/7) and itables (2.9) releases while staying compatible with older ones; remove the `plotly<6.0` and `itables<2.4` pins
- Support pandas 3 (copy-on-write, default `str` dtype for text columns, microsecond datetimes) while staying compatible with pandas 2
- Gene Expression plots show all cells with the active selection highlighted, instead of only the selected cells
- Dashboard plot and obs/var tables redraw once per step or action instead of once per event (new `EventBroker.batch()`, `coalesce()` and `flush()`)
- Results panel "Category" dropdown selects the first result automatically
- **Breaking:** the dashboard QC and Preprocess steps call `sclab.preprocess.qc()` / `sclab.preprocess.preprocess()`; Preprocess with *Group by* + *Scale* now really scales per group (previously a silent no-op)
- **Breaking:** step history entries use `step` instead of `step_description` and record the parameters actually used (including counts layer / batch key), status, error, time, data shapes and package versions; failed runs are recorded (and skipped on replay); `step_history_hash` values change
- Dashboard steps pass an explicit `random_state` (recorded in the history); CCA integration is now deterministic (`random_state=0`)
- **Breaking:** Keep/Drop/Label act on the intersection of the selection filters and the plot selection (previously the last one used won), and record the filters / box or lasso shape / a mask in `adata.uns["sclab"]["selections"]` instead of obs-name lists; Keep with nothing selected does nothing (previously removed every cell); Drop preserves cell order
- The plot's Save button downloads the figure in the browser instead of writing an HTML file to the working directory
- Doublet Detection uses each flavor's library default seed (scDblFinder 31415, scrublet/doubletdetection 0), shown in history and generated code
- `pydeseq2` removed from `uv.lock`
- Main plot keeps a 400px minimum width; the dashboard scrolls horizontally instead of collapsing it
- **Breaking:** `periodic_genes()` is now a single entry point, exported from `sclab.tools.cellflow`, with `method="cosinor"` (default, F-test p-values, tolerates dropped bins) or `method="periodogram"` (the previous behavior; raises if time bins are dropped). All options after `time_key` are keyword-only; `tmin`/`tmax` default to `adata.uns[time_key]["t_range"]` or the data range, `period` to `tmax - tmin`; parameters are recorded in `adata.uns["periodic_genes"]`
- **Breaking:** `periodic_genes()` defaults to `layer="counts"` (pass `layer="X"` to use `adata.X`) and raises `ValueError` on non-count input
- **Breaking:** `SCLabDataset.data_dict` and `SCLabDataset.metadata` are read-only; edit `dataset.adata` and call `update_data_dict()` to refresh embeddings
- **Breaking:** `cca_integrate()` raises `ValueError` when `reference_batch` is given (CCA is symmetric; use Harmony for reference-based integration)
- sctype raises `ValueError` on raw-count input; its expressed-gene filter uses each gene's own minimum, so scaled data works
- The Doublet Detection step always uses the dataset's counts layer by default; the PCA step refuses raw counts in `adata.X`
- `density()` / `real_time()` record the `t_range` and `periodic` actually used in `uns[time_key]["density"]["params"]`
- `density_dynamics(mode="valleys")` with `derivative=0` keeps all density minima by default; `height` selects valleys with density at most `height * max(density)`
- Harmony cosine-normalizes cells without the extra max-division, matching R Harmony
- Transferred boolean metadata uses the nullable `boolean` dtype (cells without enough neighbors are `<NA>`)
- `load_adata_from_url()` raises `ValueError` (instead of `AssertionError`) for invalid URLs and unsupported file types
- ALRA uses a scoped rpy2 converter instead of the deprecated global `numpy2ri.activate()`

### Fixed
- Dashboard
    - Cells with a missing category disappeared from categorical plots under pandas 3; they are drawn as "NA" again
    - Plots without per-cell sizes drew 1px points (since the plot speed-up in 0.5.0.dev6)
    - Garbage-collecting a dashboard could close widgets beyond its own and errored at interpreter shutdown
    - Callbacks crashing on dataset changes (differential expression, integration, neighbors, gene expression)
    - Selection filters wiped on every metadata change while the old selection stayed active; still-valid filters are now kept (ranges clipped to new bounds) and the selection is recomputed when a filter changes
    - PCA step with a reference batch failing when the `{counts}_log1p` layer is missing
    - With anndata>=0.13, `None` appearing as a layer option and the QC step crashing after Preprocess
    - With plotly>=6, the density contours slider failing (`'dict' object has no attribute 'min'`)
    - Plot overflowing its container on first render (plotly>=6) and not following container/window resizes (all plotly versions)
    - Extra plot redraw from Guided Pseudotime when no signal plot was open
    - Lasso selections never reaching Python
    - Redraws resetting marker opacity to 1 while the slider showed another value
- `Processor.apply_step_history()` failing with "Step history mismatch" when replaying a JSON-saved history onto a dashboard that already ran its first steps
    - Gene Expression "Plot Expression" failing while a selection is active
    - Box/lasso selection crashing while density contours are shown
    - Category dropdowns (Selection Criteria, Results) losing their value when options refresh
    - Plotter `dplt_add_vline_request` / `dplt_add_hline_request` events raising instead of adding lines
- `infer_dt_from_index()` assuming nanosecond datetimes (wrong `dt` with pandas 3's microsecond default)
- Text (`str` dtype) and plain `bool` metadata columns missing from the dashboard with pandas 3
- `scdblfinder()` removing `X` along with the layers of its working copy under anndata>=0.13
- `sclab.preprocess.pca()` reference-batch mode with `mask_var="highly_variable"` raising `KeyError` ("X_log1p") and using the PCA input layer as counts
- Embedded R scripts receiving literal newlines instead of `\n` escapes in `cat()` messages
    - Selection filters now AND together correctly when one of them matches no cells
    - Removing an `obs` column no longer breaks the selection controls
    - Gene search no longer strips "nan" from gene descriptions (e.g. "nanos")
    - PCA variance plot sized from the stored result
- Loading 10x `.h5` files from a URL (uploads share the same temp-dir reader)
- `periodic_genes()`: time binning when `tmin != 0`, dense and `csr_array` input, and gaps from dropped bins
- `density()` and `real_time()` ignoring explicit `t_range` / `periodic`
- `density_dynamics()`: valleys never found, caller's `find_peaks_kwargs` mutated, zero delta for a single periodic peak
- `transfer_metadata()` / `propagate_metadata()`: periodic values straddling the wrap boundary (circular mean instead of median), and boolean columns becoming all `True`
- `pca(layer=...)` projecting `adata.X` instead of the layer
- Harmony crashes with a list `key` plus `reference_batch` and with small datasets (< 45 cells)
- `pool_neighbors()`: `key_added` writing obsm/obs results to layers, and truncated neighbor graphs losing their last columns
- `cca()`: clear error when `n_components` exceeds the smaller group size
- sctype: neighbor pooling crash and NaN specificity weights
- Periodic `density_nd()` without `normalize=True` integrating to 1/3 (1-D) or 1/9 (2-D)
- `write_excel()`: `sort_ascending=False` ignored, per-column sort directions misaligned, crash on datetime columns

### Removed
- Unused `rotate_and_project_traces()` plotter helper and the unused `normalize` / `periodic` options of the plotter's internal density estimator

## [0.4.2] - 2026-07-13
### Added
- `call_differential_expression()` utility to call DE genes using robust Z-score thresholding on log-fold-change combined with p-value and expression-percentage filters
- Populate `sclab.tools` package `__init__.py` to expose `cellflow`, `differential_expression`, and `labeling` submodules

### Changed
- Relax R backend availability checks for edgeR, limma, and scDblFinder to no longer require `anndata2ri`
- `periodic_genes()` now normalizes total counts before computing log expression profiles and defaults to a boxcar window with constant detrending for power spectrum estimation

### Fixed
- Fix `cellflow` density plotting issues
    - Broken relative import in `density()` preventing density plots from rendering
    - Histogram bar width now scales with the actual time range instead of a fixed fraction
- Fix `_assign_numerical` in metadata transfer to avoid division-by-zero when weights sum to zero or effective sample size is 1

## [0.4.1] - 2026-03-19
### Added
- Comprehensive documentation site with MkDocs and ReadTheDocs integration
- Getting Started guide with installation, quick start, and optional R setup instructions
- API reference documentation for all public functions and classes using mkdocstrings
- User guide structure covering standard workflow, batch integration, doublet detection, differential expression, and pseudotime analysis
- Developer guide placeholders for custom steps and event system
- Material theme with dark mode, navigation tabs, search, and code highlighting

### Changed
- Update project documentation URL to https://sclab.readthedocs.io
- Add docs dependency group separate from dev dependencies
- Expand .gitignore to exclude MkDocs build output and data files

## [0.4.0] - 2026-03-06
### Added
- Reference-based PCA (`rpca()`) with multiple reference batch support and variance filtering
- `propagate_metadata()` function for filling missing metadata values using neighbor information
- Periodic gene detection utilities using power spectrum analysis and periodogram computation
- ALRA imputation implementation with R integration and scale correction
- Configurable statistics and dummy variable creation in `aggregate_and_filter()`
- Utility modules for embedding and imputation tools
- Expression dynamics analysis (`expression_dynamics()`) for computing gene turnover over pseudotime
- Per-cell metrics: number of activated/repressed genes and transcriptional flux
- Per-gene timing summaries: peak activation/repression and acceleration/deceleration onset
- Support for gene masking, custom layers, and periodic pseudotime trajectories
- Comprehensive test suite covering import validation for all major modules
- Functional tests for preprocessing workflows including group-wise operations and normalization methods
- Pseudotime computation tests for periodic and non-periodic trajectories
- End-to-end integration test for complete analysis workflow (QC → preprocessing → PCA → pseudotime → neighbors → clustering → UMAP)
- Test utility for generating synthetic single-cell data with circular trajectories
- Support for DoubletDetection and scDblFinder methods in addition to Scrublet
- Dynamic detection and registration of available doublet detection methods
- Dense matrix support for entropy-weighted normalization
- Tests for weighted normalization across sparse, dense, and batch modes

### Changed
- Refactor `transfer_metadata()` to use shared `_propagate_metadata()` helper function
- Improve metadata transfer with proper missing value handling using backfill strategy
- Enhance pseudobulk helper functions with `make_stats` and `make_dummies` parameters
- Update edgeR and limma model fitting to use factor types and `make.names()` for R compatibility
- Allow `None` values for `use_rep` parameter in integration step with automatic fallback to "X"
- Weighted normalization now uses all genes in entropy-based scaling instead of quantile filtering for improved robustness and stability
- Improve group-wise scaling implementation to avoid AnnData implicit modification warnings and future breaking changes
- Reorganize cellflow module structure with cleaner separation of pseudotime, density dynamics, and expression dynamics
- Improve Leiden clustering with igraph flavor, 2 iterations, and undirected graphs for better cluster quality
- Update EventClient to support optional broker initialization for more flexible usage patterns
- Use BoundedIntText for minimum neighbors parameter to enforce valid input constraints
- Renamed module: sclab.tools.cellflow.pseudotime -> sclab.tools.cellflow._pseudotime
    - Renamed submodules
    - Moved helper functions to sclab.tools.cellflow.pseudotime_tools
- Refactor `normalize_weighted()` to handle both sparse and dense matrices with separate code paths
- Improve batch-wise normalization to avoid zero division errors and AnnData implicit modification warnings
- Refactor test utility `simple_loop_adata()` into composable helper functions for better maintainability

### Fixed
- Group-wise highly variable gene detection now properly handles batch-specific analysis without batch_key parameter
- Design matrix construction in edgeR/limma with proper factor conversion before filtering
- Column naming in R design matrices for better compatibility across R versions
- Removed quantile filtering from weighted normalization to prevent zero scaling factors and infinite values
- Weighted normalization now uses all genes in entropy-based scaling for improved stability
- Replace deprecated `tableId` parameter with `table_id` in itables.show() calls
- Update Output widgets to use Layout instead of deprecated style parameter
- Fix import paths throughout codebase for reorganized cellflow module structure
- Zero division handling in entropy-weighted normalization
- Batch-wise operations to use index-based masking instead of view slicing

### Documentation
- Add comprehensive NumPy-style docstrings across all public APIs
- Include usage examples, parameter descriptions, and algorithm explanations

## [0.3.3] - 2025-10-17
### Added
- Standalone `preprocess()` function for complete preprocessing workflow with configurable normalization methods
- Standalone `pca()` function with reference batch support and zero-centering option
- Standalone `qc()` function for quality control with barcode ranking
- Limma-voom differential expression analysis via `pseudobulk_limma()` as alternative to edgeR
- Support for nth root selection in periodic pseudotime start estimation
- Quantile parameter (q) to `normalize_weighted()` method for entropy-based gene filtering
- Comprehensive nox test matrix covering Python 3.10-3.14 and multiple versions of numpy, scipy, anndata, and traitlets

### Changed
- Enable normalization by default in CCA integration for better batch correction
- CCA integration now uses `svd_solver="randomized"` by default for improved performance
- Reduce default replicas per group from 10 to 5 and disable bootstrap sampling by default in pseudobulk methods
- Refactor edgeR model fitting with improved design matrix handling
- Add explicit matrix copying in transfer_metadata to prevent unintended modifications
- Improve error messages for missing R dependencies

### Fixed
- Model fitting workflow in edgeR to ensure proper design matrix construction before filtering
- Column name handling in limma design matrices with `make.names()` for R compatibility

### Performance
- Implement parallel sparse matrix multiplication using joblib for improved CCA performance
- Add configurable n_jobs parameter with CPU count detection for optimal parallelization
- Introduce chunked processing for large sparse matrices to reduce memory usage

## [0.3.2] - 2025-09-15
### Added
- Group-wise highly variable gene detection for batch-aware feature selection
- Description labels to dropdown components for better user experience

### Changed
- Replace GridBox with VBox layout in ResultsPanel for cleaner vertical organization
- Change ToggleButtons to Dropdown for category selection in results interface
- Improved highly variable gene detection to handle batch-specific analysis
- Enable pts parameter in differential expression analysis by default

### Fixed
- Highly variable gene detection now properly handles grouped analysis
- ResultsPanel layout improvements for better widget alignment

## [0.3.1] - 2025-09-14
### Added
- TransferMetadata GUI component with dynamic controls for metadata transfer operations
- CCA integration wrapper function for multi-batch data integration workflows
- Zero-centering option to PCA step with improved variance handling
- Density plotting utilities with histogram and derivative visualization support

### Changed
- Enhanced processor layer/obsm dropdown handling for better data selection across layers and obsm keys
- Updated integration step to include CCA as default method option
- Improved PCA step with conditional zero-centering and enhanced projection handling
- Enhanced density dynamics plotting with integrated visualization functions

### Fixed
- Import dependencies in density dynamics module
- Processor control updates for layer and obsm key synchronization

### Removed
- Temporarily disabled URL and upload components in DataLoader interface
- Removed conditional registration requirements for integration methods

## [0.3.0] - 2025-08-21
### Added
- Density dynamics analysis module with periodic pseudotime estimation and real-time conversion
- GuidedPseudotime GUI component with interactive path drawing and automatic periodic detection
- Comprehensive preprocessing module with filtering, normalization, and integration utilities
- CCA integration for batch correction with sparse and dense matrix support
- Doublet detection using scrublet with configurable parameters
- Automated cell type labeling using scType methodology
- Uncertainty quantification for metadata transfer with error estimates
- Excel export utility with advanced formatting and conditional coloring
- Support for weighted normalization using entropy-based scaling
- Dimension range selection for neighbor computation with dynamic PCA range updates
- Reference batch support for harmony integration
- Group-wise scaling in preprocessing step
- Minimum counts filter to QC step
- Development version support in bumpversion configuration

### Changed
- Enhanced bumpversion configuration with flexible pre-release format and regex-based parsing
- Improved metadata transfer with enhanced weighting options (connectivity, distance, constant)
- Optimized PCA projection implementation and removed redundant operations
- Conditional method registration based on available optional dependencies

### Fixed
- Row filtering with in-place subset operation to maintain AnnData object instance
- Optimized transfer_metadata implementation with better memory usage

### Dependencies
- Added R integration optional dependencies
- Pinned itables version for compatibility
- Added conditional registration for harmonypy, scanorama, and scrublet

## [0.2.5] - 2025-04-18
### Dependencies
- Added `matplotlib` as a core dependency
- Added scanpy's optional dependencies for development group

### Fixed
- Upload data buttom at DataLoader

## [0.2.4] - 2025-04-10
### Added
- Dataset Integration component
- GeneExpression visualization component

### Fixed
- Create only one layer during preprocessing

## [0.2.3] - 2025-04-03
### Dependencies
- Relax dependency versions for `tqdm` and `requests`

## [0.2.2] - 2025-04-03
### Fixed
- Added required dependencies: `tqdm`, `requests`

## [0.2.1] - 2025-04-03
### Added
- Local environment variable detection in DataLoader to discover AnnData objects
- Extensible method registry system for organizing processor steps by category
- Decorator for registering methods with custom names and ordering

### Changed
- Moved processor step name and description to class attributes for simpler creation
- Added order attribute to processor steps to support sorting

### Fixed
- Fixed initialization sequence in SCLabDashboard's GridBox parent class

### Dependencies
- Removed scikit-misc from core dependencies (now part of scanpy extra)

## [0.2.0] - 2025-04-03
### Added
- URL-based AnnData loading functionality
- Interactive file upload UI with DataLoader component
- Support for loading h5ad files from both URLs and uploads
- Scanpy-compatible modules for proper file handling

### Changed
- Refactored SCLabDashboard to support multiple data loading methods
- Made copy parameter default to False when loading AnnData objects

### Fixed
- Added missing imports for BytesIO and URL parsing
- Improved error handling for file loading operations

### Dependencies
- Moved scanpy to optional dependencies
- Added scanpy[leiden] for differential expression functionality
- Added jupyterlab to dev dependency group

## [0.1.8] - 2025-02-10
### Fixed
- Resolved plotly version compatibility issue

## [0.1.7] - 2025-02-06
### Changed
- Improved CI/CD pipeline configuration for PyPI trusted publishing
- Updated GitHub Actions workflow structure

## [0.1.4] - 2025-02-06
### Added
- BSD 3-Clause License
- Updated package metadata

## [0.1.3] - 2025-02-05
### Fixed
- Relaxed dependency constraints
- Limited Python version compatibility
- Resolved type errors
- Added missing dependencies

## [0.1.2] - 2025-02-05
### Added
- Pre-commit hooks and ruff configuration
- CI/CD pipeline and testing infrastructure
- Import tests
- Improved project metadata

## [0.1.1] - 2025-02-05
### Fixed
- Resolved circular imports in dataset module
- Added version configuration system

## [0.1.0] - 2025-02-05
### Added
- Initial release of sclab
- Event system with broker and client
- Dataset module for single-cell data handling
- SCLabDashboard for integrated data analysis
- Example processor steps for single-cell analysis
- Comprehensive documentation in README
