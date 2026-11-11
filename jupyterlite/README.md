# sclab in JupyterLite

A static [JupyterLite](https://jupyterlite.readthedocs.io) site where
`content/sclab_demo.ipynb` installs sclab with micropip, builds a synthetic
dataset (3,000 cells, 2,000 genes, six cell types, two batches) and shows
`SCLabDashboard`. Everything runs in the browser
([Pyodide](https://pyodide.org) 314.0.7, Python 3.14); there is no server.

| File | |
|---|---|
| `requirements.txt` | the build environment: jupyterlite-core, jupyterlite-pyodide-kernel 0.8.7 and the frontend extensions the dashboard needs |
| `jupyter_lite_config.json` | build options: contents from `content/`, output to `dist/` |
| `jupyter-lite.json` | runtime options: Pyodide 314.0.7 from the jsDelivr CDN, the version `tests/pyodide` checks |
| `content/sclab_demo.ipynb` | the demo notebook |
| `build.sh` | builds the site |

## Build and serve

Needs [uv](https://docs.astral.sh/uv/) and network access (PyPI).

```bash
jupyterlite/build.sh           # the notebook installs sclab from PyPI
jupyterlite/build.sh --wheel   # also bundle a wheel built from this checkout
python -m http.server -d jupyterlite/dist 8000
```

Then open <http://localhost:8000/lab/index.html?path=sclab_demo.ipynb>.
`build.sh` installs the build tools in a separate environment
(`$TMPDIR/sclab-jupyterlite-venv`, or `JL_VENV`), not in sclab's. Without
the script:

```bash
pip install -r jupyterlite/requirements.txt
cd jupyterlite && jupyter lite build [--piplite-wheels path/to/sclab-*.whl]
```

`dist/` is a static site: any web server works (GitHub Pages, Read the Docs).

### Which sclab is installed

The notebook calls `piplite.install("sclab>=0.5.0.dev0")`. piplite is the
Pyodide kernel's wrapper of micropip that looks in the site's own wheel
index (`dist/pypi/`) before PyPI. With `--wheel` (or `--piplite-wheels`)
the wheel built from this checkout is found there; otherwise sclab comes from
PyPI. sclab 0.5 is the first release that runs in Pyodide; before it is on
PyPI, build with `--wheel`. Wheels can also be copied into `jupyterlite/pypi/`
before building. The other packages (anndata, numpy, scipy, ...) come from
the Pyodide distribution on the CDN or from PyPI.

The install cell follows `tests/pyodide/smoke.mjs`: typing-extensions is
reinstalled first (anndata needs a newer one than Pyodide's), and edgepython
and scrublet are installed with `deps=False` because numba and annoy have no
WebAssembly builds (sclab swaps in its own code for them).

## Widgets

JupyterLite copies the frontend (lab) extensions found in the build
environment into the site. The kernel side (ipywidgets, anywidget, plotly)
is installed by micropip as sclab dependencies; the frontend side must be in
the build:

| Package in `requirements.txt` | Needed for |
|---|---|
| `jupyterlab_widgets` | the ipywidgets 8 manager: the whole dashboard |
| `anywidget` | plotly >= 6 `FigureWidget` (the dashboard plot) and sclab's own small widgets (resize, download, copy code) |
| `plotly` | the `jupyterlab-plotly` renderer for static figures (`fig.show()`) |

itables needs nothing: sclab renders its tables as HTML with the DataTables
code inline.

Checked in headless Chromium with the site built by `build.sh --wheel`: the
install cell, the dashboard layout, the plotly plot (PCA and UMAP scatter),
the steps QC through UMAP, and the Observations table (search,
pagination, buttons) all work, with no widget errors in the
console.

## Limitations

- No threads in Pyodide: dashboard steps run on the kernel's only thread and
  the notebook is unresponsive until a step finishes. *Run in background* is
  off and disabled, and a running step cannot be cancelled.
- sclab uses its native ports of the scanpy and umap-learn functions (QC,
  Preprocess, PCA, Neighbors, Cluster, UMAP, wilcoxon/t-test/logreg DE),
  pylimma and edgepython for pseudobulk DE, and scrublet with exact
  neighbors. Not available: scanpy itself, Neighbors with *Group By Batch*
  (bbknn; the dashboard ticks it whenever a batch key is set, so the demo
  unticks it), the scanorama integration, doubletdetection, and the R
  methods.
- Size: about 10,000 to 20,000 cells is comfortable; WebAssembly memory is
  limited to 4 GB. The demo notebook's QC, Preprocess, PCA, Neighbors,
  Cluster and UMAP steps take about 16 s at 3,000 cells, 48 s at 10,000 and
  49 s at 20,000 (UMAP runs fewer epochs above 10,000 cells), plus 3 to 20 s
  for harmony integration (Pyodide 314.0.7 under node).
  UMAP is the slowest step.
- The first run downloads over 65 MB of packages (cached by the browser
  afterwards).
- Files dragged into the file browser live in the browser's storage
  (IndexedDB); they reach the kernel through a service worker, so private
  windows may not see them (not tested with sclab).
