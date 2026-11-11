# Installation

## Requirements

- Python 3.12 or newer
- A Jupyter environment (JupyterLab 4+ recommended)

## Basic installation

```bash
pip install sclab
```

## With Scanpy (recommended)

Most workflows require [Scanpy](https://scanpy.readthedocs.io). Install it alongside SCLab:

```bash
pip install "sclab[scanpy]"
```

## With JupyterLab

```bash
pip install "sclab[jupyter]"
```

## With jupyter-scatter (fast 2D plots)

```bash
pip install "sclab[jscatter]"
```

Then pick `jscatter` as the 2D plot backend in the dashboard's Settings tab, or
pass `SCLabDashboard(adata, plot_backend="jscatter")`. It draws large 2D plots
faster than plotly. Pinned views of jscatter plots keep their settings but not
the zoom. Path drawing, custom figures and extra traces still use plotly. The
`static` backend draws a matplotlib image instead.

## Other extras

| Extra | Installs | For |
| --- | --- | --- |
| `bbknn` | bbknn | Neighbors with "Group By Batch" (else sclab's native port of bbknn) |
| `doublets` | scrublet, doubletdetection | Doublet Detection |
| `de` | pylimma | `pseudobulk_limma` differential expression |
| `de-edger` | edgepython | `pseudobulk_edger`, and edgeR-style filtering for `pseudobulk_limma` |
| `lazy` | xarray, dask | lazy loading of large datasets |
| `r` | rpy2, anndata2ri | the R methods (see below) |

A step whose optional package is missing still shows in the dashboard, with
its Run button disabled and the `pip install` command it needs.
`sclab.optional_dependencies()` lists each optional feature, whether its
packages are installed, and how to install them.

## All optional dependencies at once

```bash
pip install "sclab[scanpy,bbknn,doublets,de,de-edger,jupyter,jscatter]"
```

## Development installation (uv)

If you are working on SCLab itself, clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/umbibio/sclab.git
cd sclab
uv sync --all-groups
```

---

## Optional: R dependencies

Some features require R:

- **Differential expression** — the R versions of `pseudobulk_edger` and `pseudobulk_limma`
- **Imputation** — ALRA

See the [R Setup guide](r-setup.md) for instructions.

---

## Verify installation

```python
import sclab
print(sclab.__version__)
```
