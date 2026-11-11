from . import methods
from ._citations import active_citations, citation, known_citations
from ._compat import optional_dependencies
from ._sclab import SCLabDashboard
from ._workspace import Workspace

__all__ = [
    "active_citations",
    "citation",
    "known_citations",
    "methods",
    "optional_dependencies",
    "SCLabDashboard",
    "Workspace",
]

__version__ = "0.5.0.dev15"
