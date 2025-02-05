from pathlib import Path

import anndata as ad


def read_adata(path: str | Path, var_names: str = "gene_ids") -> ad.AnnData:
    path = Path(path)

    match path.suffix:
        case ".h5" | "":
            try:
                import scanpy as sc
            except ImportError:
                raise ImportError("Please install scanpy: `pip install scanpy`")

    match path.suffix:
        case ".h5":
            adata = sc.read_10x_h5(path)
        case ".h5ad":
            adata = ad.read_h5ad(path)
        case "":
            assert path.is_dir()
            adata = sc.read_10x_mtx(path)
        case _:
            raise ValueError(
                "Input file must be a 10x h5, h5ad or a folder of 10x mtx files"
            )

    if var_names in adata.var:
        adata.var = adata.var.set_index(var_names)

    return adata
