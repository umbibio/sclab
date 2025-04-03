from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from anndata import AnnData, read_h5ad
from tqdm.auto import tqdm


def read_adata(path: str | Path, var_names: str = "gene_ids") -> AnnData:
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

    if url_path.suffix == ".h5":
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

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
    if not isinstance(url, str):
        return False

    result = urlparse(url)
    return all([result.scheme, result.netloc])
