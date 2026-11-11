import json

from anndata import AnnData


def layer_keys(adata: AnnData) -> list[str]:
    """Names of the layers in ``adata``.

    anndata>=0.13 stores ``X`` as ``adata.layers[None]``, so iterating
    ``adata.layers`` also yields ``None``. Only real layer names are returned.
    """
    return [k for k in adata.layers.keys() if k is not None]


# adata.uns["sclab"]["X"]: what adata.X holds, named like a layer (e.g.
# "counts_normt_log1p_scale"), and adata.uns["sclab"]["X_scaled"]: how it was
# scaled (see x_scaled); both set by sclab.preprocess.preprocess
UNS_KEY, X_NAME_KEY, X_SCALED_KEY = "sclab", "X", "X_scaled"


def x_name(adata: AnnData) -> str | None:
    """What ``adata.X`` holds (e.g. ``"counts_normt_log1p_scale"``), None if unknown."""
    name = adata.uns.get(UNS_KEY, {}).get(X_NAME_KEY)
    return None if name is None else str(name)


def set_x_name(
    adata: AnnData, name: str | None, scaled: bool | str | None = None
) -> None:
    """Record what ``adata.X`` holds (None: unknown) and how it was scaled.

    ``scaled``: False (not scaled), True (scaled globally), the ``adata.obs``
    column it was scaled per group of, or None (unknown, not recorded); see
    :func:`x_scaled`. ``adata.uns["sclab"]`` is replaced, not modified, so the
    dashboard's shallow copies stay independent.
    """
    keys = (X_NAME_KEY, X_SCALED_KEY)
    uns = {k: v for k, v in adata.uns.get(UNS_KEY, {}).items() if k not in keys}
    if name is not None:
        uns[X_NAME_KEY] = name
        if scaled is not None:
            uns[X_SCALED_KEY] = scaled
    adata.uns[UNS_KEY] = uns


def x_scaled(adata: AnnData) -> bool | str | None:
    """How ``adata.X`` was scaled: False (not scaled), True (globally), the
    ``adata.obs`` column it was scaled per group of, or None if unknown.

    Recorded by :func:`sclab.preprocess.preprocess`; for data preprocessed by
    older sclab, read from the params (``scale``, ``group_by``) of the last
    successful Preprocess step in ``adata.uns["sclab"]["history"]``.
    """
    uns = adata.uns.get(UNS_KEY, {})
    if X_SCALED_KEY in uns:
        value = uns[X_SCALED_KEY]
        return str(value) if isinstance(value, str) else bool(value)  # h5ad: np.bool_
    if not uns.get("history"):
        return None
    for entry in reversed(json.loads(uns["history"])):
        step = entry.get("step", entry.get("step_description"))
        if step == "Preprocess" and entry.get("status", "success") == "success":
            params = entry.get("params", {})
            if params.get("scale") is None:
                return None
            return bool(params["scale"]) and (params.get("group_by") or True)
    return None


def unscaled_layer(adata: AnnData) -> str:
    """Where the log-normalized, unscaled data are: ``"X"`` unless X is scaled
    and the layer :func:`sclab.preprocess.preprocess` kept before scaling
    (named like X without ``_scale``) exists. Also ``"X"`` when X is scaled
    and that layer is missing (data preprocessed by older sclab)."""
    name = x_name(adata)
    if x_scaled(adata) and name and name.endswith("_scale"):
        if (layer := name.removesuffix("_scale")) in layer_keys(adata):
            return layer
    return "X"


def layer_scaled(adata: AnnData, layer: str) -> bool | str | None:
    """How ``layer`` was scaled, as :func:`x_scaled`. ``"X"`` and a layer
    named like X: :func:`x_scaled`; other layers named ``*_scale`` (older
    sclab kept a copy of X under its name): scaled, as X if X is; else False."""
    if layer == "X" or layer == x_name(adata):
        return x_scaled(adata)
    return (x_scaled(adata) or True) if layer.endswith("_scale") else False


def x_label(adata: AnnData) -> str:
    """``"X (<what it holds>)"``, or ``"X"`` if unknown."""
    name = x_name(adata)
    return f"X ({name})" if name else "X"


def layer_options(adata: AnnData) -> dict[str, str]:
    """Dropdown options, label to value: ``"X"`` (labelled by :func:`x_label`)
    and the layers."""
    return {x_label(adata): "X", **{k: k for k in layer_keys(adata)}}
