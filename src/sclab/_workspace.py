"""Workspace: several named datasets, one active dashboard at a time.

Each dataset gets its own :class:`SCLabDashboard`, built when the dataset is
activated and closed when another one is. Leaving a dataset read from a file
frees its AnnData; if it was edited (its step history or pinned views in
``adata.uns["sclab"]`` changed since it was loaded) it is first written to
``<workspace_dir>/<name>.h5ad``, which is read back on the next activation.
Without a ``workspace_dir`` such a switch is refused. Datasets passed in as
AnnData objects stay in memory: the caller holds them already.
"""

import inspect
import os
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape
from importlib.util import find_spec
from pathlib import Path

from anndata import AnnData, read_h5ad
from ipywidgets.widgets import HTML, Button, Dropdown, HBox, Layout, Text, VBox

from ._io import read_adata
from ._sclab import DataLoader, SCLabDashboard, _close_tree, _live_widget_ids
from .dataset.processor._history import UNS_KEY


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


def _state_hash(adata: AnnData) -> str:
    uns = adata.uns.get(UNS_KEY, {})
    return sha256(repr((uns.get("history"), uns.get("views"))).encode()).hexdigest()


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

        self.selector = Dropdown(
            description="Dataset:", layout=Layout(width="auto", min_width="420px")
        )
        self.load_button = Button(description="+ Load", layout=Layout(width="80px"))
        self.name_text = Text(placeholder="new name", layout=Layout(width="160px"))
        self.rename_button = Button(description="Rename", layout=Layout(width="80px"))
        self.remove_button = Button(description="Remove", layout=Layout(width="80px"))
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
            entry.adata = read_adata(entry.cache_path or entry.source)
        source = str(entry.source) if entry.source else None
        entry.dash = SCLabDashboard(entry.adata, name=name, source=source)
        if loaded:
            entry.base_hash = _state_hash(entry.adata)
        entry.shape = entry.adata.shape
        self.active = name
        self.body.children = (entry.dash,)
        self._refresh()
        return entry.dash

    def rename(self, name: str, new_name: str):
        entry = self.datasets[name]
        self._check_name(new_name)
        self.datasets = {
            new_name if k == name else k: v for k, v in self.datasets.items()
        }
        entry.name = new_name
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

    def is_dirty(self, name: str) -> bool:
        """Whether the loaded dataset's history or pinned views changed since loading."""
        entry = self.datasets[name]
        return entry.adata is not None and _state_hash(entry.adata) != entry.base_hash

    def _leave(self):
        """Close the active dashboard and free its data if it can be read back."""
        if self.active is None:
            return
        entry = self.datasets[self.active]
        if entry.source is not None:  # in-memory AnnData: kept, the caller holds it
            if self.is_dirty(entry.name):
                if self.workspace_dir is None:
                    raise RuntimeError(
                        f"Dataset {entry.name!r} has unsaved edits. Create the "
                        "Workspace with a workspace_dir so they can be written "
                        "to disk when you switch away."
                    )
                entry.cache_path = self._write(entry)  # raises: switch refused
            entry.adata = None
        entry.dash.close()
        entry.dash = None
        self.active = None
        self.body.children = ()

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
        for loader in self.loader_box.children:
            _close_tree(loader, self._preexisting)
        self.loader_box.children = (DataLoader(self._on_loaded, user_ns=user_ns),)

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
