# SCLab

**SCLab** is an interactive single-cell RNA-seq analysis toolkit for Jupyter notebooks. It wraps [AnnData](https://anndata.readthedocs.io) and [Scanpy](https://scanpy.readthedocs.io) with a reactive, widget-based dashboard so you can explore, preprocess, cluster, and visualize your data without leaving your notebook.

---

## Features

- **Interactive dashboard** — tabbed layout with a live scatter plot, interactive tables, and a step-by-step analysis panel
- **Standard scRNA-seq workflow** — QC, normalization, PCA, batch integration, neighbors, UMAP, clustering, all in one place
- **Doublet detection** — Scrublet, DoubletDetection, scDblFinder
- **Differential expression** — pseudobulk analysis with edgeR or limma (optional R integration)
- **Pseudotime & trajectory analysis** — draw a path on the UMAP and let SCLab compute pseudotime
- **Extensible** — add custom analysis steps with a simple subclass API

---

## Quick Start

```python
import scanpy as sc
from sclab import SCLabDashboard

adata = sc.read_10x_h5("data.h5")
dashboard = SCLabDashboard(adata, name="My Analysis")
display(dashboard)
```

Then run the built-in steps in sequence from the sidebar panel:

1. **QC** — filter cells and genes, inspect barcode rank plot
2. **Preprocess** — normalize, log-transform, scale
3. **PCA** — dimensionality reduction
4. **Neighbors** — build the k-NN graph
5. **UMAP** — 2D/3D embedding
6. **Cluster** — Leiden clustering

---

## Installation

```bash
pip install sclab
```

See the [Installation guide](getting-started/installation.md) for full details including optional dependencies.

---

## Links

- [GitHub Repository](https://github.com/umbibio/sclab)
- [PyPI Package](https://pypi.org/project/sclab/)
- [Issue Tracker](https://github.com/umbibio/sclab/issues)
- [Changelog](https://github.com/umbibio/sclab/blob/main/CHANGELOG.md)
