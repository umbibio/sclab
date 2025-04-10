from .._methods_registry import register_sclab_method
from ..examples.processor_steps import (
    PCA,
    QC,
    UMAP,
    Cluster,
    DifferentialExpression,
    GeneExpression,
    Integration,
    Neighbors,
    Preprocess,
)

__all__ = [
    "QC",
    "Preprocess",
    "PCA",
    "Integration",
    "Neighbors",
    "UMAP",
    "Cluster",
    "GeneExpression",
    "DifferentialExpression",
]

register_sclab_method("Processing")(QC)
register_sclab_method("Processing")(Preprocess)
register_sclab_method("Processing")(PCA)
register_sclab_method("Processing")(Integration)
register_sclab_method("Processing")(Neighbors)
register_sclab_method("Processing")(UMAP)
register_sclab_method("Processing")(Cluster)
register_sclab_method("Analysis")(GeneExpression)
register_sclab_method("Analysis")(DifferentialExpression)
