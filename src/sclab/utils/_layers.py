from anndata import AnnData


def layer_keys(adata: AnnData) -> list[str]:
    """Names of the layers in ``adata``.

    anndata>=0.13 stores ``X`` as ``adata.layers[None]``, so iterating
    ``adata.layers`` also yields ``None``. Only real layer names are returned.
    """
    return [k for k in adata.layers.keys() if k is not None]
