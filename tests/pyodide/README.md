# Pyodide smoke test

Checks that sclab installs and works in [Pyodide](https://pyodide.org)
(JupyterLite, browser apps) without scanpy: it builds the sclab wheel, installs
it with micropip in Pyodide under node, builds `SCLabDashboard` on a small
synthetic AnnData and runs QC, Preprocess (two HVG methods, union) and PCA
(sclab's native port of the scanpy functions), UMAP (sclab's port of
umap-learn, on a stand-in kNN graph), CCA and Harmony integration,
Gene Expression, pseudobulk limma differential expression (function and
dashboard step) and pseudobulk edgeR (edgepython installed without numba).

It is not part of the pytest run. It needs node, [uv](https://docs.astral.sh/uv/)
(to build the wheel) and network access to the Pyodide CDN (cdn.jsdelivr.net)
and PyPI (pypi.org, files.pythonhosted.org).

```bash
npm install --prefix "$TMPDIR/pyodide-smoke" pyodide@314.0.7  # the version tested
PYODIDE_JS="$TMPDIR/pyodide-smoke/node_modules/pyodide/pyodide.mjs" \
    node tests/pyodide/smoke.mjs
```

It prints `SMOKE TEST PASSED` and exits 0, or prints the error and exits 1.

Optional environment variables:

- `PYODIDE_CACHE`: a directory to keep downloaded Pyodide packages between runs.
- `SCLAB_WHEEL`: a prebuilt sclab wheel, instead of building one.
- `NODE_USE_ENV_PROXY=1`: behind an HTTP proxy, makes node use `HTTPS_PROXY`.
