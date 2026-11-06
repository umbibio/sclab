"""Python code that reproduces a step history (the History tab's script).

Each successful history entry (see :mod:`._history`) becomes one block: a
``# <step>`` comment line and the library calls the step made, with the params
it actually used. The code comes from, in this order:

1. a function registered with :func:`register_step_codegen` for the step name,
   ``fn(params) -> str | None`` (for steps that are not
   :class:`~.step.ProcessorStepBase` subclasses, e.g. ``add_step_func`` steps);
2. the step's ``to_code(params)`` method.

``None`` means the step does not change the data (e.g. a plot): the entry is
skipped. A step with neither gets a ``# <step>: no code available`` comment.

Before a generator sees them, params equal to the dashboard's counts layer or
batch key are replaced by the variable names ``counts_layer`` / ``batch_key``
(a :class:`Name`), which the script header defines. Use :func:`call` to render
a function call with explicit keyword arguments.
"""

import json
import re
from typing import Callable

from . import _history
from ._selection_code import apply_label_code, filter_rows_code

LOAD_PLACEHOLDER = "adata = ...  # load your data here"
MAX_LINE = 88

_CODEGEN: dict[str, Callable[[dict], str | None]] = {}

# params whose value, if it equals the header variable's, is rendered as the name
_VARIABLE_KEYS = {
    "counts_layer": "counts_layer",
    "layer": "counts_layer",
    "batch_key": "batch_key",
}


def register_step_codegen(step: str, fn: Callable[[dict], str | None]) -> None:
    """Register ``fn(params) -> str | None`` as the code generator of ``step``."""
    _CODEGEN[step] = fn


class Name(str):
    """A variable name; :func:`literal` renders it unquoted."""


ADATA = Name("adata")


def literal(value) -> str:
    """Python source for a JSON-type ``value`` (double-quoted strings)."""
    if isinstance(value, Name):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(map(literal, value)) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(map(literal, value)) + ("," * (len(value) == 1)) + ")"
    if isinstance(value, dict):
        items = (f"{literal(k)}: {literal(v)}" for k, v in value.items())
        return "{" + ", ".join(items) + "}"
    return repr(value)


def call(func: str, *args, **kwargs) -> str:
    """``func(arg, ..., key=value, ...)``, one argument per line if long."""
    parts = [*map(literal, args), *(f"{k}={literal(v)}" for k, v in kwargs.items())]
    line = f"{func}({', '.join(parts)})"
    if len(line) <= MAX_LINE:
        return line
    return "\n".join([f"{func}(", *(f"    {p}," for p in parts), ")"])


def _merge_code(params: dict) -> str:
    """The merge of a Workspace dataset; ``adatas`` holds its parents by name."""
    keys = ("join", "label", "counts_layer", "equalize_depth", "random_state")
    kwargs = {k: params[k] for k in keys}
    return "adata = " + call(
        "sclab.preprocess.merge_datasets", Name("adatas"), **kwargs
    )


register_step_codegen("filter_rows", filter_rows_code)
register_step_codegen("apply_label", apply_label_code)
register_step_codegen("load", lambda params: None)  # the script's load line
register_step_codegen("merge", _merge_code)


def _entry_code(entry: dict, steps: dict, variables: dict, used: set) -> str | None:
    from .step import ProcessorStepBase

    name, params = entry["step"], dict(entry["params"])
    names = set()
    for key, var in _VARIABLE_KEYS.items():
        if var in variables and params.get(key) == variables[var]:
            params[key] = Name(var)
            names.add(var)

    fn = _CODEGEN.get(name)
    if fn is None:
        to_code = getattr(steps.get(name), "to_code", None)
        if to_code is None or to_code.__func__ is ProcessorStepBase.to_code:
            return f"# {name}: no code available for this step"
        if _history.is_summarized(params):
            return f"# {name}: params were summarized in adata.uns, no code available"
        fn = to_code
    try:
        code = fn(params)
    except Exception as e:  # e.g. an old entry lacking a param: keep the script
        return f"# {name}: code generation failed ({type(e).__name__}: {e})"
    if code is None:
        return None
    used.update(n for n in names if re.search(rf"\b{n}\b", code))
    return f"# {name}\n{code}"


def _load_line(source: str | None, history: list[dict] = ()) -> str:
    """The script's load line; ``history``'s "load" entry may add a comment."""
    from ..._io import is_valid_url

    if source is None:
        return LOAD_PLACEHOLDER
    func = "load_adata_from_url" if is_valid_url(source) else "read_adata"
    line = f"adata = {call(f'sclab._io.{func}', str(source))}"
    if (
        history
        and history[0]["step"] == "load"
        and history[0]["params"].get("uploaded")
    ):
        line += "  # uploaded through the dashboard and saved to this file"
    return line


def _imports(code: str) -> list[str]:
    imports = []
    if re.search(r"\bnp\.", code):
        imports.append("import numpy as np")
    if re.search(r"\bpd\.", code):
        imports.append("import pandas as pd")
    if re.search(r"\bsc\.", code):
        imports.append("import scanpy as sc")
    modules = re.findall(r"\b(sclab(?:\.\w+)+)\.\w+\(", code)
    imports += [f"import {m}" for m in sorted(set(modules))]
    return imports


def history_to_code(
    history: list[dict],
    *,
    steps: dict | None = None,
    header: bool = True,
    source: str | None = None,
    counts_layer: str | None = None,
    batch_key: str | None = None,
) -> str:
    """A Python script reproducing the successful entries of ``history``.

    Parameters
    ----------
    history
        Step history entries (``Processor.step_history``).
    steps
        Step description -> step object; their ``to_code`` methods generate
        the code (see :mod:`sclab.dataset.processor._codegen`).
    header
        Prepend imports, versions, the load line and the ``counts_layer`` /
        ``batch_key`` variables.
    source
        Path or URL the data was loaded from; without it the load line is a
        placeholder to fill in.
    counts_layer, batch_key
        The dashboard's counts layer and batch key.
    """
    steps = steps or {}
    variables = {"counts_layer": counts_layer, "batch_key": batch_key}
    variables = {k: v for k, v in variables.items() if v is not None}
    used: set[str] = set()
    entries = _history.succeeded(history)
    blocks = [_entry_code(e, steps, variables, used) for e in entries]
    body = "\n\n".join(b for b in blocks if b is not None)
    if not header:
        return body + "\n"

    load = _load_line(source, history)
    lines = ["# Generated by sclab from the dashboard's step history."]
    versions = next((e["versions"] for e in entries[::-1] if e.get("versions")), {})
    if versions:
        lines.append("# " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    lines += ["", *_imports(load + "\n" + body), "", load]
    if "counts_layer" in used:
        lines += [
            f"counts_layer = {literal(variables['counts_layer'])}",
            "if counts_layer not in adata.layers:  # as the dashboard does on load",
            "    adata.layers[counts_layer] = adata.X.copy()",
        ]
    if "batch_key" in used:
        lines.append(f"batch_key = {literal(variables['batch_key'])}")
    return "\n".join(lines) + "\n\n" + body + "\n"
