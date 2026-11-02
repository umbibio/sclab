from anndata import AnnData

from sclab.utils import check_raw_counts

from ..utils import aggregate_and_filter

__all__ = ["aggregate_and_filter", "validate_counts_layer"]


def validate_counts_layer(adata: AnnData, layer: str, check_counts: bool) -> None:
    """Raise ``ValueError`` for a missing layer or (optionally) non-count data.

    ``layer="X"`` selects ``adata.X``.
    """
    if layer is None:
        raise ValueError(
            "layer=None is ambiguous; pass layer='counts' for raw counts "
            "or layer='X' to use adata.X"
        )
    if layer == "X":
        X = adata.X
    elif layer in adata.layers:
        X = adata.layers[layer]
    else:
        raise ValueError(
            f"layer '{layer}' not found; pass layer='X' if adata.X holds raw counts"
        )

    if check_counts:
        check_raw_counts(
            X,
            name=f"layer '{layer}'",
            hint="Pseudobulk DE needs raw counts; use layer='counts' "
            "(or check_counts=False for non-integer corrected counts).",
        )
