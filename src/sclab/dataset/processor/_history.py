"""Step history: the record of processor steps applied to a dataset.

``Processor.step_history`` is a list of entries, one per step run (failed
runs included), each a JSON-serializable dict::

    {
        "step": "Preprocess",            # step description (key in Processor.steps)
        "params": {...},                 # params actually used, see below
        "status": "success",             # or "failed"
        "error": None,                   # "ExcType: message" (first line) if failed
        "time": "2026-11-03T10:15:42+01:00",  # ISO 8601, local time, when it ended
        "shape_before": [n_obs, n_vars],
        "shape_after": [n_obs, n_vars],
        "versions": {"sclab": "...", "scanpy": "...", "anndata": "..."},
    }

``params`` are the merged fixed, widget and extra params passed to the step's
``function``, plus whatever the step's ``resolved_params`` hook adds: filled-in
function defaults and implicit inputs read at run time (e.g. ``batch_key``,
``counts_layer``). Values are normalized to JSON types (tuples and
``pd.Index`` become lists). Replay passes back only the params the step's
``function`` accepts; the rest are informational.

The history is mirrored to ``adata.uns["sclab"]["history"]`` as a single JSON
string (h5ad cannot store nested mixed-type lists), with long lists (e.g. the
obs names of a Keep/Drop selection) replaced by a summary
``{"__summary__": "list", "len": n, "sha256": "<16 hex chars>"}``. Summarized
entries can be displayed and compared but not replayed.

Pre-0.5 entries ``{"step_description": ..., "params": ...}`` are still accepted.
"""

import functools
import json
from datetime import datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version

UNS_KEY = "sclab"
UNS_LIST_LIMIT = 50  # lists longer than this are summarized in adata.uns


def _json_default(value):
    if hasattr(value, "tolist"):  # pd.Index, np.ndarray, numpy scalars
        return value.tolist()
    return str(value)


def to_json(value):
    """Deep copy of ``value`` using JSON types only."""
    return json.loads(json.dumps(value, default=_json_default))


def compact(value):
    """Replace long lists by a ``{"__summary__": "list", "len", "sha256"}`` dict."""
    if isinstance(value, list) and len(value) > UNS_LIST_LIMIT:
        digest = sha256(json.dumps(value).encode()).hexdigest()[:16]
        return {"__summary__": "list", "len": len(value), "sha256": digest}
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def is_summarized(value) -> bool:
    if isinstance(value, dict):
        return value.get("__summary__") == "list" or any(
            is_summarized(v) for v in value.values()
        )
    if isinstance(value, list):
        return any(is_summarized(v) for v in value)
    return False


@functools.cache
def _versions() -> dict:
    from ... import __version__

    versions = {"sclab": __version__}
    for package in ("scanpy", "anndata"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def make_entry(
    step: str,
    params: dict,
    status: str,
    error: str | None,
    shape_before,
    shape_after,
) -> dict:
    return {
        "step": step,
        "params": to_json(params),
        "status": status,
        "error": error,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "shape_before": None if shape_before is None else list(shape_before),
        "shape_after": list(shape_after),
        "versions": dict(_versions()),
    }


def normalize_entry(entry: dict) -> dict:
    """Upgrade a pre-0.5 ``{step_description, params}`` entry."""
    if "step" in entry:
        return entry
    return {
        "step": entry["step_description"],
        "params": to_json(entry["params"]),
        "status": "success",
    }


def succeeded(history: list[dict]) -> list[dict]:
    return [e for e in history if e.get("status", "success") == "success"]


def entry_key(entry: dict) -> dict:
    """What identifies a step for comparison and hashing (uns-compacted)."""
    return compact({"step": entry["step"], "params": entry["params"]})


def history_hash(history: list[dict]) -> str:
    keys = [entry_key(e) for e in succeeded(history)]
    return sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


def dumps_uns(history: list[dict]) -> str:
    return json.dumps([compact(e) for e in history])


def loads_uns(adata) -> list[dict]:
    history = adata.uns.get(UNS_KEY, {}).get("history")
    return [normalize_entry(e) for e in json.loads(history)] if history else []


def format_params(params: dict) -> str:
    """Compact one-line ``k=v`` rendering of step params."""
    parts = []
    for k, v in params.items():
        if isinstance(v, dict) and v.get("__summary__") == "list":
            v = f"list({v['len']})"
        elif isinstance(v, dict):  # e.g. the filters of a Keep/Drop selection
            v = format_params(v).join("{}")
        elif isinstance(v, list | tuple) and len(v) > 4:
            v = f"list({len(v)})"
        elif isinstance(v, list | tuple):
            v = ", ".join(map(str, v)).join("()")
        elif isinstance(v, float):
            v = f"{v:.6g}"  # slider floats like 1.0000000000000002
        elif v is None:
            v = "None"
        elif not isinstance(v, int | float | str):
            v = type(v).__name__
        parts.append(f"{k}={v}")
    return ", ".join(parts)
