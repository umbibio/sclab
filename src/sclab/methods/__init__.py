from .._methods_registry import register_sclab_method
from ..examples.processor_steps import (
    PCA,
    QC,
    UMAP,
    Cluster,
    DifferentialExpression,
    DoubletDetection,
    GeneExpression,
    Integration,
    Neighbors,
    Preprocess,
)
from ..gui.components import GuidedPseudotime, TransferMetadata
from ..tools.doublet_detection import any_doublet_detection_available

__all__ = [
    "QC",
    "Preprocess",
    "PCA",
    "Integration",
    "Neighbors",
    "UMAP",
    "Cluster",
    "DoubletDetection",
    "GeneExpression",
    "DifferentialExpression",
    "GuidedPseudotime",
]

register_sclab_method("Processing")(QC)
register_sclab_method("Processing")(Preprocess)
register_sclab_method("Processing")(PCA)
register_sclab_method("Processing")(Integration)
register_sclab_method("Processing")(Neighbors)
register_sclab_method("Processing")(TransferMetadata)
register_sclab_method("Processing")(UMAP)
register_sclab_method("Processing")(Cluster)

if any_doublet_detection_available():
    register_sclab_method("Processing")(DoubletDetection)

register_sclab_method("Analysis")(GeneExpression)
register_sclab_method("Analysis")(DifferentialExpression)
register_sclab_method("Analysis")(GuidedPseudotime)
