import inspect
import sys
import types

from ..utils._call_differential_expression import call_differential_expression
from ._differential_expression import differential_expression
from ._pseudobulk_edgepython import pseudobulk_edgepython
from ._pseudobulk_edger import edger_is_available, pseudobulk_edger
from ._pseudobulk_limma import limma_is_available, pseudobulk_limma
from ._pseudobulk_pylimma import (
    edgepython_is_available,
    pseudobulk_pylimma,
    pylimma_is_available,
)

__all__ = [
    "call_differential_expression",
    "differential_expression",
    "pseudobulk_edgepython",
    "pseudobulk_edger",
    "pseudobulk_limma",
    "pseudobulk_pylimma",
    "edger_is_available",
    "limma_is_available",
    "pylimma_is_available",
    "edgepython_is_available",
]


class _CallableModule(types.ModuleType):
    """Lets ``sclab.tools.differential_expression(...)`` call the function
    while the name stays importable as this subpackage."""

    __signature__ = inspect.signature(differential_expression)

    def __call__(self, *args, **kwargs):
        return differential_expression(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
