#!/usr/bin/env bash
# Build the JupyterLite demo site into jupyterlite/dist. Needs uv and network.
#
#   jupyterlite/build.sh           # the notebook installs sclab from PyPI
#   jupyterlite/build.sh --wheel   # also bundle a wheel built from this checkout
#
# Env: JL_VENV  build environment (default: $TMPDIR/sclab-jupyterlite-venv)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
venv=${JL_VENV:-${TMPDIR:-/tmp}/sclab-jupyterlite-venv}

[ -d "$venv" ] || uv venv -q "$venv"
uv pip install -q --python "$venv" -r "$here/requirements.txt"

args=()
if [ "${1:-}" = "--wheel" ]; then
    wheels=$(mktemp -d)
    uv build -q --wheel --out-dir "$wheels" "$here/.."
    args+=(--piplite-wheels "$wheels"/sclab-*.whl)
fi

cd "$here"
"$venv/bin/jupyter" lite build "${args[@]}"
echo "Serve with: python -m http.server -d $here/dist 8000"
echo "then open http://localhost:8000/lab/index.html?path=sclab_demo.ipynb"
