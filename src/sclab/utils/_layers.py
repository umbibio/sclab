from anndata import AnnData


def layer_keys(adata: AnnData) -> list[str]:
    """Names of the layers in ``adata``.

    anndata>=0.13 stores ``X`` as ``adata.layers[None]``, so iterating
    ``adata.layers`` also yields ``None``. Only real layer names are returned.
    """
    return [k for k in adata.layers.keys() if k is not None]


# adata.uns["sclab"]["X"]: what adata.X holds, named like a layer (e.g.
# "counts_normt_log1p_scale"); set by sclab.preprocess.preprocess
UNS_KEY, X_NAME_KEY = "sclab", "X"


def x_name(adata: AnnData) -> str | None:
    """What ``adata.X`` holds (e.g. ``"counts_normt_log1p_scale"``), None if unknown."""
    name = adata.uns.get(UNS_KEY, {}).get(X_NAME_KEY)
    return None if name is None else str(name)


def set_x_name(adata: AnnData, name: str | None) -> None:
    """Record what ``adata.X`` holds (None: unknown).

    ``adata.uns["sclab"]`` is replaced, not modified, so the dashboard's
    shallow copies stay independent.
    """
    uns = {k: v for k, v in adata.uns.get(UNS_KEY, {}).items() if k != X_NAME_KEY}
    if name is not None:
        uns[X_NAME_KEY] = name
    adata.uns[UNS_KEY] = uns


def x_label(adata: AnnData) -> str:
    """``"X (<what it holds>)"``, or ``"X"`` if unknown."""
    name = x_name(adata)
    return f"X ({name})" if name else "X"


def layer_options(adata: AnnData) -> dict[str, str]:
    """Dropdown options, label to value: ``"X"`` (labelled by :func:`x_label`)
    and the layers."""
    return {x_label(adata): "X", **{k: k for k in layer_keys(adata)}}
