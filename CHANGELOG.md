# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Standalone `preprocess()` function for complete preprocessing workflow with configurable normalization methods
- Standalone `pca()` function with reference batch support and zero-centering option
- Standalone `qc()` function for quality control with barcode ranking
- Limma-voom differential expression analysis via `pseudobulk_limma()` as alternative to edgeR
- Support for nth root selection in periodic pseudotime start estimation
- Quantile parameter (q) to `normalize_weighted()` method for entropy-based gene filtering

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
