"""Logging and Profiling"""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime, timedelta, timezone
from functools import partial, update_wrapper
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from typing import TYPE_CHECKING

import anndata.logging

if TYPE_CHECKING:
    from typing import IO

    from ._settings import ScanpyConfig


# This is currently the only documented API
__all__ = ["print_versions"]

HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")


class _RootLogger(logging.RootLogger):
    def __init__(self, level):
        super().__init__(level)
        self.propagate = False
        _RootLogger.manager = logging.Manager(self)

    def log(
        self,
        level: int,
        msg: str,
        *,
        extra: dict | None = None,
        time: datetime | None = None,
        deep: str | None = None,
    ) -> datetime:
        from ._settings import settings

        now = datetime.now(timezone.utc)
        time_passed: timedelta = None if time is None else now - time
        extra = {
            **(extra or {}),
            "deep": deep if settings.verbosity.level < level else None,
            "time_passed": time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)


def _set_log_file(settings: ScanpyConfig):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
    h.setFormatter(_LogFormatter())
    h.setLevel(root.level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(h)


def _set_log_level(settings: ScanpyConfig, level: int):
    root = settings._root_logger
    root.setLevel(level)
    for h in list(root.handlers):
        h.setLevel(level)


class _LogFormatter(logging.Formatter):
    def __init__(
        self, fmt="{levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{"
    ):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord):
        format_orig = self._style._fmt
        if record.levelno == INFO:
            self._style._fmt = "{message}"
        elif record.levelno == HINT:
            self._style._fmt = "--> {message}"
        elif record.levelno == DEBUG:
            self._style._fmt = "    {message}"
        if record.time_passed:
            # strip microseconds
            if record.time_passed.microseconds:
                record.time_passed = timedelta(
                    seconds=int(record.time_passed.total_seconds())
                )
            if "{time_passed}" in record.msg:
                record.msg = record.msg.replace(
                    "{time_passed}", str(record.time_passed)
                )
            else:
                self._style._fmt += " ({time_passed})"
        if record.deep:
            record.msg = f"{record.msg}: {record.deep}"
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


print_memory_usage = anndata.logging.print_memory_usage
get_memory_usage = anndata.logging.get_memory_usage


_DEPENDENCIES_NUMERICS = [
    "anndata",  # anndata actually shouldn't, but as long as it's in development
    "umap",
    "numpy",
    "scipy",
    "pandas",
    ("sklearn", "scikit-learn"),
    "statsmodels",
    "igraph",
    "louvain",
    "leidenalg",
    "pynndescent",
]


def _versions_dependencies(dependencies):
    # this is not the same as the requirements!
    for mod in dependencies:
        mod_name, dist_name = mod if isinstance(mod, tuple) else (mod, mod)
        try:
            imp = __import__(mod_name)
            yield dist_name, imp.__version__
        except (ImportError, AttributeError):
            pass


def print_header(*, file=None):
    """\
    Versions that might influence the numerical results.
    Matplotlib and Seaborn are excluded from this.

    Parameters
    ----------
    file
        Optional path for dependency output.
    """

    modules = ["scanpy"] + _DEPENDENCIES_NUMERICS
    print(
        " ".join(f"{mod}=={ver}" for mod, ver in _versions_dependencies(modules)),
        file=file or sys.stdout,
    )


def print_versions(*, file: IO[str] | None = None):
    """\
    Print versions of imported packages, OS, and jupyter environment.

    For more options (including rich output) use `session_info.show` directly.

    Parameters
    ----------
    file
        Optional path for output.
    """
    import session_info

    if file is not None:
        from contextlib import redirect_stdout

        warnings.warn(
            "Passing argument 'file' to print_versions is deprecated, and will be "
            "removed in a future version.",
            FutureWarning,
        )
        with redirect_stdout(file):
            print_versions()
    else:
        session_info.show(
            dependencies=True,
            html=False,
            excludes=[
                "builtins",
                "stdlib_list",
                "importlib_metadata",
                # Special module present if test coverage being calculated
                # https://gitlab.com/joelostblom/session_info/-/issues/10
                "$coverage",
            ],
        )


def print_version_and_date(*, file=None):
    """\
    Useful for starting a notebook so you see when you started working.

    Parameters
    ----------
    file
        Optional path for output.
    """
    from . import __version__

    if file is None:
        file = sys.stdout
    print(
        f"Running Scanpy {__version__}, on {datetime.now():%Y-%m-%d %H:%M}.",
        file=file,
    )


def _copy_docs_and_signature(fn):
    return partial(update_wrapper, wrapped=fn, assigned=["__doc__", "__annotations__"])


def error(
    msg: str,
    *,
    time: datetime = None,
    deep: str | None = None,
    extra: dict | None = None,
) -> datetime:
    """\
    Log message with specific level and return current time.

    Parameters
    ----------
    msg
        Message to display.
    time
        A time in the past. If this is passed, the time difference from then
        to now is appended to `msg` as ` (HH:MM:SS)`.
        If `msg` contains `{time_passed}`, the time difference is instead
        inserted at that position.
    deep
        If the current verbosity is higher than the log function’s level,
        this gets displayed as well
    extra
        Additional values you can specify in `msg` like `{time_passed}`.
    """
    from ._settings import settings

    return settings._root_logger.error(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def warning(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.warning(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def info(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.info(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def hint(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.hint(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def debug(msg, *, time=None, deep=None, extra=None) -> datetime:
    from ._settings import settings

    return settings._root_logger.debug(msg, time=time, deep=deep, extra=extra)
