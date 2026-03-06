"""CellFlow tools for pseudotime, density dynamics, and expression dynamics.

This module re-exports the main public functions for working with pseudotime-based
analyses in single-cell data, including:

- :func:`pseudotime` — compute pseudotime from a low-dimensional embedding.
- :func:`density_dynamics` — detect density peaks/valleys along pseudotime.
- :func:`real_time` — convert pseudotime to real time via density normalisation.
- :func:`expression_dynamics` — compute per-cell gene turnover from expression.
- :func:`piecewise_rescale` — rescale pseudotime to real time using known phase
  durations.
"""

from . import utils
from ._density_dynamics import density_dynamics, real_time
from ._expression_dynamics import expression_dynamics
from ._pseudotime._pseudotime import pseudotime
from ._pseudotime._timeseries import piecewise_rescale

__all__ = [
    "expression_dynamics",
    "density_dynamics",
    "piecewise_rescale",
    "pseudotime",
    "utils",
    "real_time",
    "utils",
]
