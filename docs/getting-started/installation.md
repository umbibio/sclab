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
