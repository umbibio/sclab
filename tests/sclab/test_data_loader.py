"""The "+ Load" panel: file browser, URL download, upload, defined AnnData."""

import functools
import tempfile
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

import sclab._sclab as sclab_module
from sclab import SCLabDashboard, Workspace
from sclab._io import download, load_adata_from_url, peek_shape
from sclab._workspace import _open_lazy, workspace_to_code

from .test_utils import simple_loop_adata
from .test_utils import _tenx_files as tx


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def server(tmp_path):
    """A local http.server over ``tmp_path / "srv"``: (folder, base URL)."""
    folder = tmp_path / "srv"
    folder.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(folder))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield folder, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """The notebook folder (cwd): datasets, other files and a subfolder."""
    root = tmp_path / "nb"
    (root / "sub" / "deeper").mkdir(parents=True)
    simple_loop_adata(n_obs=60, n_vars=20).write_h5ad(root / "a.h5ad")
    simple_loop_adata(n_obs=40, n_vars=20).write_h5ad(root / "sub" / "b.h5ad")
    tx.write_10x_h5_v3(root / "v3.h5")
    tx.write_10x_mtx_v3(root / "mtx")
    tx.write_10x_mtx_legacy(root / "legacy")
    (root / "half_mtx").mkdir()  # matrix only: not readable, listed as a folder
    (root / "half_mtx" / "matrix.mtx.gz").write_bytes(b"")
    (root / "notes.txt").write_text("hi")
    (root / "table.csv").write_text("a,b\n")
    (root / "data.zarr").mkdir()  # read_adata does not read zarr
    (root / ".hidden.h5ad").write_bytes(b"")
    monkeypatch.chdir(root)
    return root


def make_loader(**kwargs) -> sclab_module.DataLoader:
    loaded = []
    loader = sclab_module.DataLoader(
        lambda adata, source: loaded.append((adata, source)), user_ns={}, **kwargs
    )
    loader.loaded = loaded
    return loader


def labels(loader) -> list[str]:
    return [label for label, _ in loader.files.options]


def open_loader(ws: Workspace):
    ws.load_button.click()
    (loader,) = ws.loader_box.children
    return loader


def test_listing_shows_only_readable_datasets(root):
    def size(name):
        path = root / name
        files = path.iterdir() if path.is_dir() else [path]
        return sclab_module._fmt_size(sum(f.stat().st_size for f in files))

    loader = make_loader()
    assert loader.tabs.selected_index == 0  # nothing defined: Files
    assert labels(loader) == [
        "data.zarr/",
        "half_mtx/",
        "sub/",
        f"a.h5ad · h5ad · {size('a.h5ad')} · 60×20",
        f"legacy/ · 10x mtx folder · {size('legacy')} · 4×3",
        f"mtx/ · 10x mtx folder · {size('mtx')} · 4×2",
        f"v3.h5 · 10x h5 · {size('v3.h5')} · 4×2",
    ]
    assert "nb" in loader.breadcrumbs.value
    loader = sclab_module.DataLoader(print, user_ns={"x": simple_loop_adata()})
    assert loader.tabs.selected_index == 3  # "Defined in notebook"
    assert loader.defined_adatas.options == ("x",)


def test_peek_shape_matches_read_adata_and_skips_x(root, monkeypatch):
    from sclab._io import read_adata

    for name in ["a.h5ad", "v3.h5", "mtx", "legacy"]:
        assert peek_shape(root / name) == read_adata(root / name).shape, name
    tx.write_10x_h5_legacy(root / "old.h5")
    assert peek_shape(root / "old.h5") == (4, 3)
    assert peek_shape(root / "notes.txt") is None
    assert peek_shape(root / "half_mtx") is None
    (root / "bad.h5ad").write_bytes(b"not hdf5")
    assert peek_shape(root / "bad.h5ad") is None
    # the Workspace's lazy shape now covers 10x files too
    assert _open_lazy(root / "v3.h5") == (4, 2)
    assert _open_lazy(root / "mtx") == (4, 2)


def test_navigation_is_confined_to_the_notebook_folder(root):
    loader = make_loader()
    assert "../" not in labels(loader)  # at the root
    loader.files.value = root / "sub"
    assert loader.folder == root / "sub"
    assert labels(loader)[:2] == ["../", "deeper/"]
    assert labels(loader)[2].startswith("b.h5ad · h5ad")
    assert "nb / sub" in loader.breadcrumbs.value
    loader.files.value = root  # ".."
    assert loader.folder == root and "../" not in labels(loader)

    for text in ["..", "/", "../.."]:
        loader.path_text.value = text
        loader.go_button.click()
        assert loader.folder == root
        assert "outside the notebook folder" in loader.status.value
    loader.path_text.value = "sub/deeper"
    loader.go_button.click()
    assert loader.folder == root / "sub" / "deeper" and loader.status.value == ""
    assert "no .h5ad" in loader.breadcrumbs.value
    loader.path_text.value = "nope"
    loader.go_button.click()
    assert "No such file or folder" in loader.status.value
    loader.path_text.value = "../../notes.txt"
    loader.go_button.click()
    assert "Cannot read notes.txt" in loader.status.value


def test_picking_a_file_adds_it_unread(root, tmp_path):
    ws = Workspace(workspace_dir=tmp_path / "ws")
    loader = open_loader(ws)
    assert loader.tabs.selected_index == 0  # no AnnData defined
    loader.files.value = root / "a.h5ad"
    loader.add_button.click()
    # nothing was active: the picked file is opened
    assert ws.active == "a" and ws.datasets["a"].source == root / "a.h5ad"
    assert ws.loader_box.children == () and loader.comm is None

    loader = open_loader(ws)
    loader.files.value = root / "sub"
    loader.files.value = root / "sub" / "b.h5ad"
    loader.add_button.click()
    loader = open_loader(ws)
    loader.path_text.value = "a.h5ad"  # typed path, relative to the root
    loader.go_button.click()
    loader = open_loader(ws)
    loader.files.value = root / "mtx"
    loader.add_button.click()
    assert list(ws.datasets) == ["a", "b", "a 2", "mtx"]
    for name, shape in [("b", (40, 20)), ("a 2", (60, 20)), ("mtx", (4, 2))]:
        entry = ws.datasets[name]
        assert entry.adata is None and entry.shape == shape  # lazy: not read
    assert ws.active == "a"  # still: adding does not switch
    assert "Added mtx (4×2)" in ws.message.value
    code = workspace_to_code(ws, "b")
    assert "read_adata(" in code and f'"{root / "sub" / "b.h5ad"}"' in code


def test_picking_without_a_pick_or_an_unreadable_file_shows_the_error(root, tmp_path):
    ws = Workspace(workspace_dir=tmp_path / "ws")
    loader = open_loader(ws)
    loader.add_button.click()
    assert "Pick a file" in loader.status.value
    (root / "bad.h5ad").write_bytes(b"not hdf5")
    loader._cd(root)
    assert any(label.startswith("bad.h5ad · h5ad") for label in labels(loader))
    loader.files.value = root / "bad.h5ad"
    loader.add_button.click()
    assert loader.status.value and "bad" not in ws.datasets
    assert ws.loader_box.children == (loader,)


def test_plain_dashboard_loader_reads_the_picked_file(root):
    dash = SCLabDashboard()
    loader = dash.data_loader
    loader.files.value = root / "v3.h5"
    loader.add_button.click()
    assert dash.ds.adata.shape == (4, 2)
    assert dash.source == str(root / "v3.h5")
    code = dash.processor.generate_code()
    assert "read_adata(" in code and f'"{root / "v3.h5"}"' in code


def test_url_download_into_workspace_dir(root, tmp_path, server):
    folder, base = server
    simple_loop_adata(n_obs=30, n_vars=20).write_h5ad(folder / "remote.h5ad")
    url = f"{base}/remote.h5ad"
    ws = Workspace({"a": root / "a.h5ad"}, workspace_dir=tmp_path / "ws")
    loader = open_loader(ws)
    assert "ws/downloads" in loader.url_row.children[1].value
    loader.url.value = url
    loader.load_button.click()
    path = tmp_path / "ws" / "downloads" / "remote.h5ad"
    assert path.read_bytes() == (folder / "remote.h5ad").read_bytes()
    assert not list(path.parent.glob(".*.part"))
    assert "Saved to" in loader.url_progress.value
    entry = ws.datasets["remote"]
    assert entry.source == path and entry.url == url and entry.shape == (30, 20)
    (load,) = ws.history("remote")
    assert load["params"] == {"source": url, "name": "remote", "path": str(path)}
    code = workspace_to_code(ws, "remote")
    assert "sclab._io.load_adata_from_url(" in code and f'"{url}"' in code
    ws.activate("remote")  # read from the downloaded file; the script downloads
    assert ws.dash.ds.adata.shape == (30, 20)
    code = ws.dash.processor.generate_code()
    assert "load_adata_from_url(" in code and f'"{url}"' in code

    loader = open_loader(ws)  # the same URL again: a new file, a new name
    loader.url.value = url
    loader.load_button.click()
    assert ws.datasets["remote-2"].source == path.with_name("remote-2.h5ad")


def test_url_without_workspace_dir_goes_to_a_temporary_folder(
    root, server, tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # mkdtemp in tmp_path
    folder, base = server
    tx.write_10x_h5_v3(folder / "v3.h5")
    loader = make_loader()
    assert "temporary folder" in loader.url_row.children[1].value
    loader.url.value = f"{base}/v3.h5"
    loader.load_button.click()
    ((adata, source),) = loader.loaded
    assert adata.shape == (4, 2) and source == f"{base}/v3.h5"
    assert (loader.dest_dir / "downloads" / "v3.h5").is_file()
    assert loader.dest_dir.parent == tmp_path


@pytest.mark.parametrize(
    "url,error",
    [
        ("not a url", "URL is not valid"),
        ("http://127.0.0.1:1/x.csv", "Unsupported file type"),
        ("{base}/missing.h5ad", "404"),
        ("http://127.0.0.1:1/x.h5ad", "Connection"),  # nothing listens there
    ],
)
def test_url_errors_are_shown_in_the_panel(root, tmp_path, server, url, error):
    _, base = server
    ws = Workspace(workspace_dir=tmp_path / "ws")
    loader = open_loader(ws)
    loader.url.value = url.format(base=base)
    loader.load_button.click()
    assert error in loader.status.value
    assert ws.datasets == {} and not loader.load_button.disabled
    assert not list((tmp_path / "ws").rglob("*.h5ad*"))


def test_download_and_load_adata_from_url_via_http(tmp_path, server):
    folder, base = server
    tx.write_10x_h5_v3(folder / "my data.h5")
    url = f"{base}/my%20data.h5"
    seen = []
    path = download(url, tmp_path / "dl", progress=lambda d, t: seen.append((d, t)))
    assert path == tmp_path / "dl" / "my data.h5"
    assert seen[-1] == (path.stat().st_size,) * 2
    assert load_adata_from_url(url, progress=False).shape == (4, 2)


def upload_value(name: str, content: bytes) -> tuple[dict]:
    return (
        {
            "name": name,
            "type": "application/octet-stream",
            "size": len(content),
            "content": memoryview(content),
            "last_modified": datetime.now(timezone.utc),
        },
    )


def test_upload_is_saved_and_has_a_load_line(root, tmp_path):
    ws = Workspace(workspace_dir=tmp_path / "ws")
    loader = open_loader(ws)
    assert "For small files only" in loader.upload_info.value
    content = (root / "a.h5ad").read_bytes()
    loader.upload.value = upload_value("up.h5ad", content)
    path = tmp_path / "ws" / "uploads" / "up.h5ad"
    assert path.read_bytes() == content
    assert loader.upload.value == ()  # bytes freed
    entry = ws.datasets["up"]
    assert entry.source == path and entry.uploaded and ws.active == "up"
    code = ws.dash.processor.generate_code()
    load = code[code.index("adata = sclab._io.read_adata(") :].split("\n\n")[0]
    assert f'"{path}"' in load
    assert load.endswith("# uploaded through the dashboard and saved to this file")
    assert "large uploads" not in loader.upload_info.value


def test_large_upload_warns_and_bad_uploads_are_refused(root, tmp_path, monkeypatch):
    monkeypatch.setattr(sclab_module, "UPLOAD_WARN_BYTES", 10)
    ws = Workspace(workspace_dir=tmp_path / "ws")
    loader = open_loader(ws)
    loader.upload.value = upload_value("big.h5ad", (root / "a.h5ad").read_bytes())
    assert "large uploads can stall" in loader.upload_info.value
    assert "big" in ws.datasets  # warned, then loaded

    loader = open_loader(ws)
    loader.upload.value = upload_value("table.csv", b"a,b\n")
    assert "Cannot read table.csv" in loader.status.value
    loader.upload.value = upload_value("broken.h5ad", b"not hdf5")
    assert loader.status.value and "broken" not in ws.datasets
    assert not (tmp_path / "ws" / "uploads" / "table.csv").exists()
