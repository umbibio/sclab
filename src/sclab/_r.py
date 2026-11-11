"""Route the console output and warnings of sclab's own R calls.

``options(warn=-1)`` would silence every R warning for the rest of the R
session, including the user's own R code. Instead, ``r_output`` scopes the
routing to the R code sclab runs inside the ``with`` block:

- R console writes (``cat``, ``print``, ``message()``, progress) go to this
  module's logger at DEBUG, or to the notebook as usual when ``verbose``;
- R warnings are collected with ``withCallingHandlers`` and re-emitted on
  exit as one Python ``UserWarning``, deduplicated with counts;
- R errors propagate as ``RRuntimeError``, as before.

The rpy2 console callbacks are restored on exit and no R option is changed,
so R code run outside the block behaves as R normally does. Show the R
output with ``verbose=True`` where a wrapper has it, or with
``logging.getLogger("sclab._r").setLevel(logging.DEBUG)`` plus a handler.
"""

import logging
import re
import warnings
from collections import Counter
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Regexes (re.search) for R warnings known to be harmless; matches are logged
# at DEBUG instead of warned. Each entry needs a comment saying why it is safe.
IGNORED_R_WARNINGS: list[str] = []

# Evaluates sclab's R code with a calling handler that records each warning
# in `state` and muffles it, so it is neither printed nor deferred to the
# user's next R call. Errors are not handled and propagate.
_HANDLERS_R = """
local({
    collect <- function(state) function(w) {
        state$warnings <- c(state$warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    }
    list(
        eval = function(code, state) withCallingHandlers(
            eval(parse(text = code), envir = globalenv()),
            warning = collect(state)
        ),
        call = function(.f, .state, ...) withCallingHandlers(
            .f(...), warning = collect(.state)
        )
    )
})
"""


class _REval:
    """Drop-in for ``rpy2.robjects.r`` whose evaluations route warnings."""

    def __init__(self, eval_fn, call_fn, state):
        self._eval, self._call, self._state = eval_fn, call_fn, state

    def __call__(self, code: str):
        """Evaluate R code in the global environment, like ``robjects.r``."""
        return self._eval(code, self._state)

    def call(self, fn, *args, **kwargs):
        """Call an R function (e.g. from ``importr``) with routed warnings."""
        return self._call(fn, self._state, *args, **kwargs)

    def __getattr__(self, name):  # assign, matrix, ... as on robjects.r
        import rpy2.robjects as robjects

        return getattr(robjects.r, name)


@contextmanager
def r_output(task: str, verbose: bool = False):
    """Route R output and warnings of the R code run in the block.

    Yields an evaluator to use instead of ``rpy2.robjects.r``: ``R(code)``
    evaluates R code and ``R.call(fn, *args)`` calls an R function.

    Parameters
    ----------
    task : str
        Name used in the warning, e.g. ``"scDblFinder"``.
    verbose : bool, optional
        If True, R console output is shown as usual (the current rpy2
        console callbacks are kept) instead of logged at DEBUG. R warnings
        are collected either way. Default is False.
    """
    import rpy2.robjects as robjects
    from rpy2.rinterface_lib import callbacks
    from rpy2.robjects.conversion import localconverter

    with localconverter(robjects.default_converter):
        helpers = robjects.r(_HANDLERS_R)
        eval_fn, call_fn = helpers.rx2("eval"), helpers.rx2("call")
        state = robjects.r("local({e <- new.env(); e$warnings <- character(0); e})")

    saved = callbacks.consolewrite_print, callbacks.consolewrite_warnerror
    if not verbose:
        callbacks.consolewrite_print = _log_r_console
        callbacks.consolewrite_warnerror = _log_r_console
    try:
        yield _REval(eval_fn, call_fn, state)
    finally:
        callbacks.consolewrite_print, callbacks.consolewrite_warnerror = saved
        with localconverter(robjects.default_converter):
            messages = [str(m) for m in state["warnings"]]
        _warn(task, messages)


def _log_r_console(text: str) -> None:
    if text.strip():
        logger.debug("R: %s", text.rstrip("\n"))


def _warn(task: str, messages: list[str]) -> None:
    kept = []
    for message in messages:
        if any(re.search(p, message) for p in IGNORED_R_WARNINGS):
            logger.debug("R warning during %s (ignored): %s", task, message)
        else:
            kept.append(message)
    if not kept:
        return

    lines = [m if n == 1 else f"{m} (x{n})" for m, n in Counter(kept).items()]
    s = "" if len(kept) == 1 else "s"
    warnings.warn(
        f"R reported {len(kept)} warning{s} during {task}:\n"
        + "\n".join(f"  - {line}" for line in lines),
        UserWarning,
        stacklevel=4,  # _warn < r_output < contextlib __exit__ < the wrapper
    )
