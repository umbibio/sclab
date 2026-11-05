"""Workspace: several named datasets, one active dashboard at a time.

Each dataset gets its own :class:`SCLabDashboard`, built when the dataset is
activated and closed when another one is. Leaving a dataset read from a file
frees its AnnData; if it was edited (its step history or pinned views in
``adata.uns["sclab"]`` changed since it was loaded) it is first written to
``<workspace_dir>/<name>.h5ad``, which is read back on the next activation.
Without a ``workspace_dir`` such a switch is refused. Datasets passed in as
AnnData objects stay in memory: the caller holds them already.

The history of a dataset read from a file starts with a ``"load"`` entry
(``source``, ``name``). :meth:`Workspace.merge` makes a new dataset from
several (see :func:`sclab.preprocess.merge_datasets`), kept in memory; once
edited, it is written to ``workspace_dir`` (if set) when you leave it.
:func:`workspace_to_code` writes the script that rebuilds it from its parents.
"""

import inspect
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape
from importlib.util import find_spec
from pathlib import Path

from anndata import AnnData, read_h5ad
from ipywidgets.widgets import (
    HTML,
    Button,
    Checkbox,
    Dropdown,
    HBox,
    Layout,
    Text,
    VBox,
)

from ._io import read_adata
from ._sclab import DataLoader, SCLabDashboard, _close_tree, _live_widget_ids
from .dataset.processor import _history
from .dataset.processor._codegen import _imports, _load_line, history_to_code, literal
from .dataset.processor._history import UNS_KEY
from .preprocess import merge_datasets, merge_preview


@dataclass
class DatasetEntry:
    name: str
    source: Path | None  # file the data is read from; None: an in-memory AnnData
    adata: AnnData | None = None  # None while the data is only on disk
    dash: SCLabDashboard | None = None  # only the active dataset has one
    cache_path: Path | None = None  # where its edits were written on leaving it
    parents: list[str] = field(default_factory=list)  # datasets it was made from
    base_hash: str | None = None  # _state_hash when loaded: differs once edited
    shape: tuple[int, int] | None = None  # (n_obs, n_vars); None until known
    batch_key: str | None = None  # its dashboard's batch_key (a merge's label)


def _state_hash(adata: AnnData) -> str:
    uns = adata.uns.get(UNS_KEY, {})
    return sha256(repr((uns.get("history"), uns.get("views"))).encode()).hexdigest()


def _uns_history(path: Path) -> list[dict]:
    """The step history saved in an .h5ad file, without reading its data."""
    import h5py

    if path.suffix != ".h5ad":
        return []
    with h5py.File(path, "r") as f:
        value = f.get(f"uns/{UNS_KEY}/history")
        value = value[()] if value is not None else None
    if isinstance(value, bytes):
        value = value.decode()
    return [_history.normalize_entry(e) for e in json.loads(value)] if value else []


def _with_load_entry(history: list[dict], entry: "DatasetEntry", shape) -> list[dict]:
    """A file dataset's history starts with where it was read from."""
    if entry.source is None or (history and history[0]["step"] == "load"):
        return history
    params = {"source": str(entry.source), "name": entry.name}
    load = _history.make_entry("load", params, "success", None, None, shape)
    return [load, *history]


def _lazy_reader():
    """``anndata.experimental.read_lazy`` if usable (needs xarray and dask)."""
    if not (find_spec("xarray") and find_spec("dask")):
        return None
    try:
        from anndata.experimental import read_lazy
    except ImportError:  # anndata < 0.12
        return None
    return read_lazy


def _open_lazy(source: Path) -> tuple[int, int] | None:
    """Shape of the dataset at ``source`` without reading X.

    None for 10x ``.h5`` files and mtx folders: their shape is known once the
    dataset is activated.
    """
    if source.suffix != ".h5ad":
        return None
    if read_lazy := _lazy_reader():
        import h5py

        with h5py.File(source, "r") as f:
            return tuple(read_lazy(f).shape)
    adata = read_h5ad(source, backed="r")  # reads obs and var; X stays on disk
    try:
        return adata.shape
    finally:
        adata.file.close()


class Workspace(VBox):
    """Several named datasets with a switcher and one active dashboard.

    Parameters
    ----------
    datasets : dict of str to AnnData or path, optional
        Datasets by name. Files (``.h5ad``, 10x ``.h5`` or mtx folder) are
        read when activated; the first dataset is activated right away.
    workspace_dir : path, optional
        Where edited file-backed datasets are written when you switch away
        from them. Without it, switching away from one is refused.

    Examples
    --------
    >>> ws = Workspace({"a": adata, "b": "b.h5ad"}, workspace_dir="ws")
    >>> ws.activate("b")
    >>> ws.dash.processor.steps["QC"].run()
    """

    def __init__(
        self,
        datasets: dict[str, AnnData | str | os.PathLike] | None = None,
        *,
        workspace_dir: str | os.PathLike | None = None,
    ):
        self._preexisting = _live_widget_ids()  # not ours to close, see close()
        user_ns = inspect.currentframe().f_back.f_locals  # for "+ Load"
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.datasets: dict[str, DatasetEntry] = {}
        self.active: str | None = None
        self._syncing = False
        self._watcher = None  # the active dashboard's coalesced _on_state_change

        self.selector = Dropdown(
            description="Dataset:", layout=Layout(width="auto", min_width="420px")
        )
        self.load_button = Button(description="+ Load", layout=Layout(width="80px"))
        self.name_text = Text(placeholder="new name", layout=Layout(width="160px"))
        self.rename_button = Button(description="Rename", layout=Layout(width="80px"))
        self.remove_button = Button(description="Remove", layout=Layout(width="80px"))
        self.merge_button = Button(description="Merge...", layout=Layout(width="90px"))
        self.message = HTML()
        self.loader_box = VBox(layout=Layout(width="100%"))
        self.body = VBox(layout=Layout(width="100%"))
        bar = HBox(
            [
                self.selector,
                self.load_button,
                self.name_text,
                self.rename_button,
                self.remove_button,
                self.merge_button,
            ]
        )
        VBox.__init__(
            self,
            [bar, self.message, self.loader_box, self.body],
            layout=Layout(width="100%"),
        )

        self.selector.observe(self._on_select, "value")
        # user_ns stays in the closure: close() must not walk the user's namespace
        self.load_button.on_click(lambda _: self._show_loader(user_ns))
        self.rename_button.on_click(
            lambda _: self._ui(self.rename, self.active, self.name_text.value)
        )
        self.remove_button.on_click(lambda _: self._ui(self.remove, self.active))
        self.merge_button.on_click(lambda _: self._show_merge())

        for name, data in (datasets or {}).items():
            self.add(name, data)
        if self.datasets:
            self.activate(next(iter(self.datasets)))

    @property
    def dash(self) -> SCLabDashboard | None:
        """The active dataset's dashboard."""
        return self.datasets[self.active].dash if self.active else None

    def add(self, name: str, data: AnnData | str | os.PathLike) -> DatasetEntry:
        """Add a dataset, without loading it if it is a file."""
        self._check_name(name)
        if isinstance(data, AnnData):
            entry = DatasetEntry(
                name, None, adata=data, base_hash=_state_hash(data), shape=data.shape
            )
        else:
            source = Path(data)
            if not source.exists():
                raise FileNotFoundError(source)
            entry = DatasetEntry(name, source, shape=_open_lazy(source))
        self.datasets[name] = entry
        self._refresh()
        return entry

    def activate(self, name: str) -> SCLabDashboard:
        """Show dataset ``name``, leaving the active one (see the module doc)."""
        entry = self.datasets[name]
        if name == self.active:
            return entry.dash
        self._leave()
        loaded = entry.adata is None
        if loaded:
            entry.adata = self._read(entry)
        source = str(entry.source) if entry.source else None
        entry.dash = SCLabDashboard(
            entry.adata, name=name, source=source, batch_key=entry.batch_key
        )
        if entry.parents:  # "Show code": the script that rebuilds it from its parents
            entry.dash.processor.code_generator = lambda: workspace_to_code(
                self, entry.name
            )
            entry.dash.processor._update_code()
        if loaded:
            entry.base_hash = _state_hash(entry.adata)
        entry.shape = entry.adata.shape
        # coalesced: one refresh per batch, after a (background) step is applied
        self._watcher = entry.dash.broker.coalesce(self._on_state_change)
        entry.dash.broker.subscribe("dset_state_change", self._watcher)
        self.active = name
        self.body.children = (entry.dash,)
        self._refresh()
        return entry.dash

    def rename(self, name: str, new_name: str):
        """Rename a dataset, its live dashboard and its cache file, if any."""
        entry = self.datasets[name]
        self._check_name(new_name)
        if entry.cache_path is not None:
            cache = self.workspace_dir / f"{new_name}.h5ad"
            if cache.exists():
                raise FileExistsError(f"{cache} exists; remove it or pick another name")
            entry.cache_path = entry.cache_path.rename(cache)
        if entry.dash is not None:
            entry.dash.name = entry.dash.dataset.name = new_name
        self.datasets = {
            new_name if k == name else k: v for k, v in self.datasets.items()
        }
        entry.name = new_name
        for other in self.datasets.values():
            other.parents = [new_name if p == name else p for p in other.parents]
        if self.active == name:
            self.active = new_name
        self._refresh()

    def remove(self, name: str):
        """Drop a dataset; its edits, if any, are written first as when leaving it."""
        if name not in self.datasets:
            raise KeyError(name)
        if name == self.active:
            self._leave()
        del self.datasets[name]
        self._refresh()

    def close(self):
        """Close the workspace, the active dashboard and all their widgets."""
        _close_tree(self, self._preexisting)

    __del__ = SCLabDashboard.__del__  # GC: close only this widget, see there

    def is_dirty(self, name: str) -> bool:
        """Whether the loaded dataset's history or pinned views changed since loading."""
        entry = self.datasets[name]
        return entry.adata is not None and _state_hash(entry.adata) != entry.base_hash

    def _check_idle(self):
        if self.dash is not None and self.dash.broker.jobs.busy:
            raise RuntimeError(
                f"{self.dash.broker.jobs.name!r} is running in {self.active!r}; "
                "wait for it to finish or cancel it."
            )

    def _leave(self):
        """Close the active dashboard and free its data if it can be read back."""
        if self.active is None:
            return
        self._check_idle()
        entry = self.datasets[self.active]
        # in-memory AnnData: kept, the caller holds it; a merge: kept unless written
        if (entry.source is not None or entry.parents) and self.is_dirty(entry.name):
            if self.workspace_dir is not None:
                entry.cache_path = self._write(entry)  # raises: switch refused
            elif entry.source is not None:
                raise RuntimeError(
                    f"Dataset {entry.name!r} has unsaved edits. Create the "
                    "Workspace with a workspace_dir so they can be written "
                    "to disk when you switch away."
                )
        if entry.source is not None or entry.cache_path is not None:
            entry.adata = None
        entry.dash.broker.unsubscribe("dset_state_change", self._watcher)
        self._watcher = None
        entry.dash.close()
        entry.dash = None
        self.active = None
        self.body.children = ()

    def _read(self, entry: DatasetEntry) -> AnnData:
        adata = read_adata(entry.cache_path or entry.source)
        history = _with_load_entry(_history.loads_uns(adata), entry, adata.shape)
        adata.uns.setdefault(UNS_KEY, {})["history"] = _history.dumps_uns(history)
        return adata

    def history(self, name: str) -> list[dict]:
        """Dataset ``name``'s step history, read from its file if not loaded."""
        entry = self.datasets[name]
        if entry.adata is not None:
            return _history.loads_uns(entry.adata)
        history = _uns_history(entry.cache_path or entry.source)
        return _with_load_entry(history, entry, entry.shape)

    @contextmanager
    def _loaded(self, names: list[str], counts_layer: str = "counts"):
        """The datasets ``names`` by name, read for the block if on disk."""
        entries = [self.datasets[n] for n in names]
        read = [e for e in entries if e.adata is None]
        try:
            for e in read:
                e.adata = self._read(e)
                if counts_layer not in e.adata.layers:  # as the dashboard does
                    e.adata.layers[counts_layer] = e.adata.X.copy()
            yield {e.name: e.adata for e in entries}
        finally:
            for e in read:
                e.adata = None

    def merge_preview(self, names: list[str], **opts) -> dict:
        """:func:`sclab.preprocess.merge_preview` of the datasets ``names``."""
        with self._loaded(names, opts.get("counts_layer", "counts")) as adatas:
            return merge_preview(adatas, **opts)

    def merge(self, names: list[str], new_name: str, **opts) -> DatasetEntry:
        """Add dataset ``new_name``, the merge of ``names``; they are not changed.

        ``opts`` go to :func:`sclab.preprocess.merge_datasets`. The new dataset
        stays in memory, its dashboard's ``batch_key`` is the merge's label.
        """
        self._check_name(new_name)
        self._check_idle()
        with self._loaded(names, opts.get("counts_layer", "counts")) as adatas:
            merged = merge_datasets(adatas, **opts)
        (merge,) = _history.loads_uns(merged)
        entry = DatasetEntry(
            new_name,
            None,
            adata=merged,
            parents=list(names),
            base_hash=_state_hash(merged),
            shape=merged.shape,
            batch_key=merge["params"]["label"],
        )
        self.datasets[new_name] = entry
        self._refresh()
        return entry

    def _write(self, entry: DatasetEntry) -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        path = self.workspace_dir / f"{entry.name}.h5ad"
        tmp = self.workspace_dir / f".{entry.name}.tmp.h5ad"
        try:  # via a temporary file: a failed write keeps the previous cache
            entry.adata.write_h5ad(tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def _check_name(self, name: str):
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid dataset name: {name!r}")
        if name in self.datasets:
            raise ValueError(f"A dataset named {name!r} already exists")

    def _label(self, entry: DatasetEntry) -> str:
        shape = "×".join(map(str, entry.shape)) if entry.shape else "?×?"
        if entry.adata is not None:
            state = "in memory"
        else:
            state = "cached" if entry.cache_path else "on disk"
        edited = entry.cache_path is not None or self.is_dirty(entry.name)
        return f"{entry.name} · {shape} · {state} · {'modified' if edited else 'clean'}"

    def _refresh(self):
        self._syncing = True
        try:
            self.selector.options = [
                (self._label(e), n) for n, e in self.datasets.items()
            ]
            self.selector.value = self.active
        finally:
            self._syncing = False

    def _on_state_change(self):
        """The active dataset's history or pinned views changed: update its label."""
        if (entry := self.datasets.get(self.active)) is not None and entry.dash:
            entry.shape = entry.adata.shape  # a Keep/Drop step changes it
            self._refresh()

    def _ui(self, fn, *args):
        """Run a top-bar action; show its error instead of raising it."""
        self.message.value = ""
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 - a widget callback: show every error
            style = "color:#b91c1c;font-size:12px"
            self.message.value = f"<span style='{style}'>{escape(str(e))}</span>"
        self._refresh()

    def _on_select(self, change):
        if not self._syncing and change["new"] is not None:
            self._ui(self.activate, change["new"])

    def _show_loader(self, user_ns: dict):
        self._show_panel(DataLoader(self._on_loaded, user_ns=user_ns))

    def _show_panel(self, panel=None):
        """Show ``panel`` (the loader or the merge dialog) under the top bar."""
        for old in self.loader_box.children:
            _close_tree(old, self._preexisting)
        self.loader_box.children = (panel,) if panel is not None else ()

    def _show_merge(self):
        self._show_panel(
            MergeDialog(list(self.datasets), self.merge_preview, self._on_merge)
        )

    def _on_merge(self, names: list[str], new_name: str, opts: dict):
        self.message.value = ""
        self.merge(names, new_name, **opts)  # raises: the dialog shows the error
        self._show_panel()
        self._ui(self.activate, new_name)

    def _on_loaded(self, adata: AnnData, source: str | None):
        loader = self.loader_box.children[0]
        name = base = loader.defined_adatas.value or "dataset"
        i = 1
        while name in self.datasets:
            i += 1
            name = f"{base} {i}"
        self.loader_box.children = ()
        _close_tree(loader, self._preexisting)
        self.add(name, adata)
        self._ui(self.activate, name)


class MergeDialog(VBox):
    """The Workspace's "Merge..." panel: pick datasets, preview, create.

    ``preview(names, **opts) -> dict`` and ``create(names, new_name, opts)``
    are the Workspace's; "Create" stays disabled until a preview of the
    current choices has no blocking errors.
    """

    def __init__(self, names: list[str], preview, create):
        self._preview, self._create = preview, create  # methods: _close_tree skips
        self.checks = {
            n: Checkbox(value=False, description=n, indent=False) for n in names
        }
        self.label = Text(value="sample", description="Label:")
        self.join = Dropdown(options=["inner", "outer"], description="Join:")
        self.equalize_depth = Checkbox(
            value=False, description="Equalize depth (binomial thinning)", indent=False
        )
        self.new_name = Text(value="merged", description="Name:")
        self.preview_button = Button(description="Preview")
        self.create_button = Button(
            description="Create merged dataset",
            button_style="success",
            disabled=True,
            layout=Layout(width="200px"),
        )
        self.output = HTML()
        for w in [*self.checks.values(), self.label, self.join, self.equalize_depth]:
            w.observe(lambda _: setattr(self.create_button, "disabled", True), "value")
        self.preview_button.on_click(lambda _: self.run_preview())
        self.create_button.on_click(lambda _: self.run_create())
        VBox.__init__(
            self,
            [
                HTML("<b>Merge datasets</b>"),
                HBox(list(self.checks.values())),
                HBox([self.label, self.join, self.equalize_depth]),
                HBox([self.new_name, self.preview_button, self.create_button]),
                self.output,
            ],
            layout=Layout(width="100%"),
        )

    @property
    def names(self) -> list[str]:
        return [n for n, c in self.checks.items() if c.value]

    @property
    def opts(self) -> dict:
        return {
            "join": self.join.value,
            "label": self.label.value,
            "equalize_depth": self.equalize_depth.value,
        }

    def run_preview(self):
        try:
            preview = self._preview(self.names, **self.opts)
        except Exception as e:  # noqa: BLE001 - a widget callback: show every error
            self.output.value = _error_html([str(e)])
            self.create_button.disabled = True
            return
        self.output.value = preview_html(preview)
        self.create_button.disabled = bool(preview["errors"])

    def run_create(self):
        try:
            self._create(self.names, self.new_name.value, self.opts)
        except Exception as e:  # noqa: BLE001 - a widget callback: show every error
            self.output.value += _error_html([str(e)])


def _error_html(messages: list[str], color: str = "#b91c1c") -> str:
    return "".join(
        f"<div style='color:{color};font-size:12px'>{escape(m)}</div>" for m in messages
    )


def preview_html(preview: dict) -> str:
    """:func:`sclab.preprocess.merge_preview` as a compact HTML table."""
    header = ["Dataset", "Cells", "Genes", "IDs", "Dropped", "Filled", "Counts"]
    header += ["Median counts", "Median genes", "Depth ratio"]

    def fmt(info, key, spec):
        return format(info[key], spec) if key in info else ""

    rows = []
    for name, info in preview["datasets"].items():
        cells = [name, info["n_obs"], info["n_vars"], info["id_kind"]]
        cells += [info["genes_dropped"], info["genes_filled"], info["counts"]]
        cells += [fmt(info, "median_counts", ".0f"), fmt(info, "median_genes", ".0f")]
        cells.append(fmt(info, "depth_ratio", ".2f"))
        rows.append("".join(f"<td>{escape(str(c))}</td>" for c in cells))
    ths = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join(f"<tr>{r}</tr>" for r in rows)
    summary = (
        f"Result: {preview['n_obs']} cells × {preview['n_vars']} genes "
        f"({preview['shared_genes']} shared), {preview['join']} join, "
        f"label {preview['label']!r}"
    )
    return (
        "<style>.sclab-merge td,.sclab-merge th{padding:2px 8px;text-align:left}"
        "</style>"
        f"<table class='sclab-merge'><tr>{ths}</tr>{trs}</table>"
        f"<div style='font-size:12px'>{escape(summary)}</div>"
        + _error_html(preview["errors"])
        + _error_html(preview["warnings"], color="#92400e")
    )


_ENSURE_COUNTS = (
    "if counts_layer not in adata.layers:  # as the dashboard does on load\n"
    "    adata.layers[counts_layer] = adata.X.copy()"
)


def _at_merge(history: list[dict], digest: str) -> list[dict] | None:
    """The successful steps of ``history`` up to the hash ``digest``, if any."""
    steps = _history.succeeded(history)
    for i in range(len(steps), -1, -1):
        if _history.history_hash(steps[:i]) == digest:
            return steps[:i]
    return None


def workspace_to_code(workspace: Workspace, name: str) -> str:
    """A script rebuilding dataset ``name`` of ``workspace`` from its files.

    For a merged dataset: one block per parent (its load line and steps up to
    the merge, ending ``adatas["<parent>"] = adata``), the merge call, then
    the merged dataset's own steps. Otherwise the dataset's plain script. The
    step code comes from the active dashboard's steps.
    """
    entry = workspace.datasets[name]
    history = workspace.history(name)
    steps = workspace.dash.processor.steps if workspace.dash else {}
    merge = next((e for e in history if e["step"] == "merge"), None)
    if merge is None:
        source = str(entry.source) if entry.source else None
        return history_to_code(
            history, steps=steps, source=source, batch_key=entry.batch_key
        )

    params = merge["params"]
    counts_layer = params["counts_layer"]
    blocks = []
    for parent, current in zip(params["parents"], entry.parents):
        lines = [f"# ---- {parent['name']} ----"]
        if current in workspace.datasets:
            parent_history = workspace.history(current)
            at_merge = _at_merge(parent_history, parent["history_hash"])
            if at_merge is None:
                lines.append(f"# {current} changed since the merge: its steps now")
                at_merge = parent_history
        else:
            lines.append("# no longer in the workspace: its steps are unknown")
            at_merge = []
        source = parent["source"]
        if any(e["step"] == "merge" for e in at_merge):
            lines.append(
                "# a merge itself: load it here (merges of merges unsupported)"
            )
            source, at_merge = None, []
        batch_key = getattr(workspace.datasets.get(current), "batch_key", None)
        body = history_to_code(
            at_merge,
            steps=steps,
            header=False,
            counts_layer=counts_layer,
            batch_key=batch_key,
        )
        lines += [_load_line(source), _ENSURE_COUNTS]
        lines += ["", body.strip()] if body.strip() else []
        lines.append(f"adatas[{literal(parent['name'])}] = adata")
        blocks.append("\n".join(lines))
    own = history_to_code(
        history,
        steps=steps,
        header=False,
        counts_layer=counts_layer,
        batch_key=entry.batch_key,
    )
    body = "\n\n".join(blocks) + "\n\n" + own

    lines = ["# Generated by sclab from the workspace's step histories."]
    versions = next((e["versions"] for e in history[::-1] if e.get("versions")), {})
    if versions:
        lines.append("# " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    lines += ["", *_imports(body), "", f"counts_layer = {literal(counts_layer)}"]
    if re.search(r"\bbatch_key\b", body):
        lines.append(f"batch_key = {literal(entry.batch_key)}")
    lines.append("adatas = {}")
    return "\n".join(lines) + "\n\n" + body
