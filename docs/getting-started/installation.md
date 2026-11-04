# Installation

## Requirements

- Python 3.10–3.13
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

## All optional dependencies at once

```bash
pip install "sclab[scanpy,jupyter]"
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

- **Differential expression** — `pseudobulk_edger`, `pseudobulk_limma`
- **Imputation** — ALRA

See the [R Setup guide](r-setup.md) for instructions.

---

## Verify installation

```python
import sclab
print(sclab.__version__)
```
