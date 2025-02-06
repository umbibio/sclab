# SCLab

SCLab is an interactive single-cell analysis toolkit that provides a seamless interface for analyzing and visualizing single-cell RNA sequencing data. Built on top of popular tools like scanpy and AnnData, SCLab offers an event-driven architecture for real-time updates and interactive visualizations.

## Features

- **Interactive Data Analysis**: Built-in dashboard with real-time updates
- **Quality Control**: Comprehensive QC metrics and filtering capabilities
- **Preprocessing**: Normalization, log transformation, and scaling with progress tracking
- **Dimensionality Reduction**: PCA with batch effect correction support
- **Visualization**: Interactive plots and tables using plotly and itables
- **Event System**: Robust event-driven architecture for real-time updates

## Installation

```bash
pip install git+https://github.com/umbibio/sclab.git
```

## Quick Start

```python
from sclab import SCLabDashboard
import scanpy as sc

# Load your data
adata = sc.read_10x_h5("your_data.h5")

# Create dashboard
dashboard = SCLabDashboard(adata, name="My Analysis")

# The dashboard provides easy access to components:
dashboard.ds  # Dataset (wrapper for AnnData)
dashboard.pl  # Plotter
dashboard.pr  # Processor
```

## Components

### SCLabDashboard

The main interface that integrates all components with a tabbed layout:
- Main graph for visualizations
- Observations table
- Genes table
- Event logs

### Dataset

Handles data management with:
- AnnData integration
- Interactive tables
- Row selection and filtering
- Metadata handling

### Processor

Handles data processing steps. It is configurable with custom steps implementing the `ProcessorStepBase` interface. This package provides multiple examples of steps:

- QC
- Preprocessing
- PCA
- Nearest Neighbors
- UMAP
- Clustering

### Plotter

Provides interactive visualizations with:
- Real-time updates
- Customizable plots
- Batch effect visualization
- Export capabilities

## Requirements

- Python ≥ 3.12
- anndata ≥ 0.11.3
- scanpy ≥ 1.10.4
- Other dependencies listed in pyproject.toml

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SCLab in your research, please cite:

```bibtex
@software{sclab2024,
  author = {Arriojas, Argenis},
  title = {SCLab: Interactive Single-Cell Analysis Toolkit},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/umbibio/sclab}
}
