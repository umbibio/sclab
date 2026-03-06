import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from anndata import AnnData, read_h5ad
from tqdm.auto import tqdm


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

    match path.suffix:
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


def load_adata_from_url(
    url: str,
    var_names: str = "gene_ids",
    progress: bool = True,
) -> AnnData:
    """
    Load an AnnData object from a URL to an .h5ad file.

    Parameters:
    -----------
    url : str
        URL to the .h5ad file
    var_names : str
        Name of the variable column in the .h5ad file
    progress : bool
        Whether to show a progress bar

    Returns:
    --------
    anndata.AnnData
        Loaded AnnData object
    """
    from .scanpy.readwrite import read_10x_h5

    assert is_valid_url(url), "URL is not valid"
    url_path = Path(urlparse(url).path)

    file_content = fetch_file(url, progress=progress)
    match url_path.suffix:
        case ".h5":
            adata = read_10x_h5(file_content)
        case ".h5ad":
            adata = read_h5ad(file_content)
        case _:
            raise ValueError("Input file must be a 10x h5 or h5ad file")

    if var_names in adata.var:
        adata.var = adata.var.set_index(var_names)

    return adata


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
