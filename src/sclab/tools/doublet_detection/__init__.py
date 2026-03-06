from ._doubletdetection import doubletdetection, doubletdetection_is_available
from ._scdblfinder import scdblfinder, scdblfinder_is_available
from ._scrublet import scrublet, scrublet_is_available


def any_doublet_detection_available() -> bool:
    return any(
        [
            doubletdetection_is_available(),
            scdblfinder_is_available(),
            scrublet_is_available(),
        ]
    )


def get_available_doublet_detection_methods() -> list[str]:
    methods = []
    if doubletdetection_is_available():
        methods.append("doubletdetection")

    if scdblfinder_is_available():
        methods.append("scDblFinder")

    if scrublet_is_available():
        methods.append("scrublet")

    return methods


__all__ = [
    "doubletdetection",
    "scdblfinder",
    "scrublet",
    "any_doublet_detection_available",
    "get_available_doublet_detection_methods",
]
