# Quick Start

This guide walks through loading data and running a standard single-cell analysis in SCLab.

## Load your data

SCLab accepts an AnnData object, a local file path, or a URL:

=== "AnnData object"

    ```python
    import scanpy as sc
    from sclab import SCLabDashboard

    adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
    dashboard = SCLabDashboard(adata, name="My Analysis")
    display(dashboard)
    ```

=== "File path (.h5 / .h5ad / MTX folder)"

    ```python
    from sclab import SCLabDashboard

    dashboard = SCLabDashboard("path/to/data.h5ad", name="My Analysis")
    display(dashboard)
    ```

=== "URL"

    ```python
    from sclab import SCLabDashboard

    dashboard = SCLabDashboard(
        "https://example.com/data.h5ad",
        name="My Analysis",
    )
    display(dashboard)
    ```

=== "No data (interactive loader)"

    ```python
    from sclab import SCLabDashboard

    dashboard = SCLabDashboard()
    display(dashboard)
    ```

    SCLab will show a widget listing any AnnData objects already defined in your notebook, so you can pick one interactively.

---

## Run the standard workflow

Once the dashboard is displayed, run each step from the left-hand panel in order:

| Step | What it does |
|------|-------------|
| **QC** | Filter low-quality cells and genes, compute QC metrics |
| **Preprocess** | Normalize, log-transform, and scale |
| **PCA** | Reduce dimensions; inspect the variance ratio plot |
| **Neighbors** | Build the k-NN graph used for UMAP and clustering |
| **UMAP** | Compute a 2D embedding for visualization |
| **Cluster** | Leiden clustering; results appear as `leiden` in the plot |

---

## Access components programmatically

The dashboard exposes its sub-components as properties:

```python
# AnnData — all results are stored here
adata = dashboard.ds.adata

# Run a step from code
dashboard.pr.steps["QC"].run()

# Access the plotter
dashboard.pl
```

---

## Next steps

- [The Dashboard](../user-guide/dashboard.md) — understand the layout and tabs
- [Standard Workflow](../user-guide/standard-workflow.md) — detailed guide for each step
- [Batch Integration](../user-guide/batch-integration.md) — multi-sample analysis
