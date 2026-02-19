# Technology Stack - SCLab

## Core Language & Runtime
- **Python (>= 3.10, < 3.13)**: The primary programming language, chosen for its dominance in the bioinformatics and data science communities.

## Bioinformatics Foundation
- **anndata**: For handling annotated data matrices, the standard format for single-cell data in Python.
- **scanpy**: The primary library for single-cell analysis (preprocessing, visualization, clustering, etc.).

## User Interface & Interactivity
- **ipywidgets**: For creating interactive GUI components within Jupyter Notebooks.
- **anywidget**: To build custom, high-performance web-based widgets that integrate seamlessly with the Python backend.
- **itables**: For interactive and searchable display of DataFrames (observations, genes, etc.).

## Visualization
- **plotly**: For interactive, web-based data visualizations and plots.
- **matplotlib**: Used for static plotting and as a foundation for some visualization tasks.

## Data Science & Numerics
- **pandas**: For efficient data manipulation and analysis of tabular data.
- **numpy & scipy**: Fundamental libraries for scientific computing and numerical operations.
- **scikit-learn**: For various machine learning algorithms used in preprocessing and analysis.

## Development & Infrastructure
- **uv**: For fast and reliable dependency management and project environments.
- **flit**: Used as the build backend for simple and clean packaging.
- **nox**: For automated testing and environment management across multiple Python versions.
- **ruff**: A fast, all-in-one Python linter and formatter.
