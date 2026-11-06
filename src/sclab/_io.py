import gzip
import hashlib
import os
import tempfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from anndata import AnnData, read_h5ad
from tqdm.auto import tqdm

_URL_SUFFIXES = (".h5", ".h5ad")


def read_adata(path: str | Path, var_names: str = "gene_ids") -> AnnData:
    """Read an AnnData object from a local file.

    Supports 10x Genomics HDF5 (.h5), AnnData HDF5 (.h5ad), and 10x MTX
    folders. If a column named ``var_names`` exists in ``adata.var``, it is
    promoted to the index.

    Parameters
    ----------
    path : str or Path
        Path to a ``.h5`` file (10x HDF5), a ``.h5ad`` file, or a directory
        containing 10x MTX files (``matrix.mtx``, ``barcodes.tsv``,
        ``features.tsv``).
    var_names : str, optional
        Column in ``adata.var`` to use as the variable index. Default is
        ``"gene_ids"``.

    Returns
    -------
    AnnData
        Loaded annotated data matrix.

    Raises
    ------
    ValueError
        If the file extension is not ``.h5``, ``.h5ad``, or an empty string
        (directory).
    """
    from .scanpy.readwrite import read_10x_h5, read_10x_mtx

    path = Path(path)

    match "" if is_mtx_dir(path) else path.suffix:  # an mtx folder, whatever its name
        case ".h5":
            adata = read_10x_h5(path)
        case ".h5ad":
            adata = read_h5ad(path)
        case "":
            assert path.is_dir()
            adata = read_10x_mtx(path)
        case _:
            raise ValueError(
                "Input file must be a 10x h5, h5ad or a folder of 10x mtx files"
            )

    if var_names in adata.var:
        adata.var = adata.var.set_index(var_names)

    return adata


def is_mtx_dir(path: str | Path) -> bool:
    """Whether ``path`` is a folder of 10x mtx files that :func:`read_adata` reads."""
    path = Path(path)
    if (path / "genes.tsv").is_file():  # legacy (Cell Ranger < 3), not compressed
        names = ("matrix.mtx", "genes.tsv", "barcodes.tsv")
    else:
        names = ("matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz")
    return path.is_dir() and all((path / n).is_file() for n in names)


def peek_shape(path: str | Path) -> tuple[int, int] | None:
    """``read_adata(path).shape`` from the file's metadata, without reading X.

    For 10x files only the "Gene Expression" features are counted, as
    :func:`read_adata` keeps only those. None if unknown or unreadable.
    """
    import h5py

    path = Path(path)
    try:
        if is_mtx_dir(path):
            legacy = (path / "genes.tsv").is_file()
            opener = open if legacy else gzip.open
            matrix = path / ("matrix.mtx" if legacy else "matrix.mtx.gz")
            with opener(matrix, "rt") as f:  # features x barcodes
                header = next(line for line in f if not line.startswith("%"))
            n_vars, n_obs = map(int, header.split()[:2])
            if not legacy:
                with gzip.open(path / "features.tsv.gz", "rt") as f:
                    types = [line.rstrip("\n").split("\t")[2] for line in f]
                n_vars = types.count("Gene Expression")
            return n_obs, n_vars
        if path.suffix == ".h5ad":
            with h5py.File(path, "r") as f:
                X = f.get("X")
                if isinstance(X, h5py.Dataset):
                    return tuple(int(n) for n in X.shape)
                if X is None:
                    return None
                return tuple(int(n) for n in X.attrs["shape"])  # sparse group
        if path.suffix == ".h5":
            with h5py.File(path, "r") as f:
                if "matrix" in f:  # Cell Ranger >= 3
                    types = f["matrix/features/feature_type"][:].astype(str)
                    n_obs = int(f["matrix/shape"][1])
                    return n_obs, int((types == "Gene Expression").sum())
                (genome,) = f.values()  # legacy: one genome, else read_adata fails too
                n_vars, n_obs = genome["shape"][:]
                return int(n_obs), int(n_vars)
    except (OSError, KeyError, ValueError, IndexError, StopIteration, TypeError):
        return None
    return None


def unique_path(path: Path) -> Path:
    """``path``, or ``<stem>-2<suffix>``, ``-3``, ... if it exists."""
    stem, suffix, i = path.stem, path.suffix, 1
    while path.exists():
        i += 1
        path = path.with_name(f"{stem}-{i}{suffix}")
    return path


def download(
    url: str,
    dest_dir: str | Path,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download a 10x .h5 or .h5ad file from ``url`` into ``dest_dir``.

    The file is named after the URL's last path component (``-2``, ``-3``...
    appended if that file exists) and written via a temporary ``.part`` file,
    so a failed download leaves nothing behind.

    Parameters
    ----------
    url : str
        URL whose path ends in ``.h5`` or ``.h5ad``.
    dest_dir : str or Path
        Folder to save the file in; created if missing.
    progress : callable, optional
        Called as ``progress(bytes_done, bytes_total)`` after each chunk;
        ``bytes_total`` is None if the server does not send a length.

    Returns
    -------
    Path
        The downloaded file.

    Raises
    ------
    ValueError
        If ``url`` is not valid or does not point to a ``.h5`` or ``.h5ad`` file.
    requests.HTTPError
        If the server returns an error status.
    """
    if not is_valid_url(url):
        raise ValueError(f"URL is not valid: {url!r}")
    name = Path(unquote(urlparse(url).path)).name
    if Path(name).suffix not in _URL_SUFFIXES:
        supported = ", ".join(_URL_SUFFIXES)
        raise ValueError(
            f"Unsupported file type {Path(name).suffix!r} in URL; "
            f"supported types: {supported}"
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = unique_path(dest_dir / name)
    tmp = path.with_name(f".{path.name}.part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        done = 0
        try:
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return path


def _read_adata_bytes(
    content: BytesIO | bytes,
    suffix: str,
    var_names: str = "gene_ids",
) -> AnnData:
    """Read an AnnData object from the raw bytes of a ``.h5`` or ``.h5ad`` file.

    The readers need a real file path, so the bytes are written to a file in a
    temporary directory (not a ``NamedTemporaryFile``, which cannot be reopened
    on Windows), read fully into memory, and the directory is removed.

    Parameters
    ----------
    content : BytesIO or bytes
        Raw file content.
    suffix : str
        File extension including the dot, e.g. ``".h5"`` or ``".h5ad"``. It
        selects the reader (see :func:`read_adata`).
    var_names : str, optional
        Column in ``adata.var`` to use as the variable index. Default is
        ``"gene_ids"``.

    Returns
    -------
    AnnData
        Loaded annotated data matrix (in memory, not backed by a file).

    Raises
    ------
    ValueError
        If ``suffix`` is not a supported file extension.
    """
    data = content.getvalue() if isinstance(content, BytesIO) else bytes(content)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"data{suffix}"
        path.write_bytes(data)
        adata = read_adata(path, var_names=var_names)
        if adata.isbacked:
            adata = adata.to_memory()

    return adata


def load_adata_from_url(
    url: str,
    var_names: str = "gene_ids",
    progress: bool = True,
) -> AnnData:
    """
    Load an AnnData object from a URL to a 10x .h5 or an .h5ad file.

    Parameters:
    -----------
    url : str
        URL to the .h5 (10x HDF5) or .h5ad file
    var_names : str
        Name of the variable column to use as the variable index
    progress : bool
        Whether to show a progress bar

    Returns:
    --------
    anndata.AnnData
        Loaded AnnData object

    Raises:
    -------
    ValueError
        If ``url`` is not a valid URL or does not point to a ``.h5`` or
        ``.h5ad`` file.
    """
    if not is_valid_url(url):
        raise ValueError(f"URL is not valid: {url!r}")

    suffix = Path(urlparse(url).path).suffix
    if suffix not in _URL_SUFFIXES:
        supported = ", ".join(_URL_SUFFIXES)
        raise ValueError(
            f"Unsupported file type {suffix!r} in URL; supported types: {supported}"
        )

    file_content = fetch_file(url, progress=progress)
    return _read_adata_bytes(file_content, suffix, var_names=var_names)


def fetch_file(url: str, progress: bool = True) -> BytesIO:
    """Download a file from a URL into an in-memory buffer.

    Parameters
    ----------
    url : str
        URL to download.
    progress : bool, optional
        If True, display a tqdm progress bar. Default is True.

    Returns
    -------
    BytesIO
        In-memory buffer containing the downloaded file content.

    Raises
    ------
    requests.HTTPError
        If the HTTP request returns an error status code.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    if progress:
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    result = BytesIO()
    for data in response.iter_content(block_size):
        result.write(data)
        if progress:
            progress_bar.update(len(data))

    if progress:
        progress_bar.close()

    return result


def is_valid_url(url: str) -> bool:
    """Check whether a string is a valid URL.

    Parameters
    ----------
    url : str
        String to validate.

    Returns
    -------
    bool
        True if ``url`` has both a scheme and a netloc, False otherwise.
    """
    if not isinstance(url, str):
        return False

    result = urlparse(url)
    return all([result.scheme, result.netloc])


def get_file_hash(file_path, algorithm="sha256") -> str:
    """Compute the hash of a file using ``hashlib.file_digest`` (Python 3.11+).

    Parameters
    ----------
    file_path : str or Path
        Path to the file to hash.
    algorithm : str, optional
        Hash algorithm name accepted by :mod:`hashlib`, e.g. ``"sha256"`` or
        ``"md5"``. Default is ``"sha256"``.

    Returns
    -------
    str
        Hexadecimal digest string of the file contents.
    """
    with open(file_path, "rb") as f:
        digest = hashlib.file_digest(f, algorithm)

    return digest.hexdigest()
