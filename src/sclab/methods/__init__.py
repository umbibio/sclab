from .._methods_registry import register_sclab_method
from ..examples.processor_steps import (
    PCA,
    QC,
    UMAP,
    Cluster,
    DifferentialExpression,
    Neighbors,
    Preprocess,
)

__all__ = [
    "QC",
    "Preprocess",
    "PCA",
    "Neighbors",
    "UMAP",
    "Cluster",
    "DifferentialExpression",
]

register_sclab_method("Processing")(QC)
register_sclab_method("Processing")(Preprocess)
register_sclab_method("Processing")(PCA)
register_sclab_method("Processing")(Neighbors)
register_sclab_method("Processing")(UMAP)
register_sclab_method("Processing")(Cluster)
register_sclab_method("Analysis")(DifferentialExpression)
