import logging
import warnings
from importlib.util import find_spec

import pytest

from sclab import _r
from sclab._r import r_output


def _r_loads() -> bool:
    if find_spec("rpy2") is None:
        return False
    try:  # rpy2 present but cannot load R (e.g. built for a newer R)
        from rpy2.robjects import r

        r("1")
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _r_loads(), reason="requires rpy2 and R")


def _r_warnings(record):
    return [w for w in record if "R reported" in str(w.message)]


def test_r_warning_becomes_one_user_warning():
    with pytest.warns(UserWarning) as record:
        with r_output("demo") as R:
            value = R('warning("first"); warning("second"); 42')

    assert value[0] == 42
    [w] = _r_warnings(record)
    assert w.category is UserWarning
    message = str(w.message)
    assert message.startswith("R reported 2 warnings during demo:")
    assert "  - first\n  - second" in message


def test_duplicate_warnings_are_counted():
    with pytest.warns(UserWarning) as record:
        with r_output("demo") as R:
            R('for (i in 1:3) warning("again"); warning("once")')

    [w] = _r_warnings(record)
    assert str(w.message) == (
        "R reported 4 warnings during demo:\n  - again (x3)\n  - once"
    )


def test_function_call_warnings_are_collected():
    import rpy2.robjects as robjects

    f = robjects.r("function(x, y) { warning('in f'); x + y }")
    with pytest.warns(UserWarning, match="in f"):
        with r_output("demo") as R:
            value = R.call(f, 1, y=2)
    assert value[0] == 3


def test_no_warning_without_r_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with r_output("demo") as R:
            R("1")


def test_ignored_warnings_are_dropped(monkeypatch, caplog):
    monkeypatch.setattr(_r, "IGNORED_R_WARNINGS", [r"^harmless"])
    with pytest.warns(UserWarning) as record:
        with caplog.at_level(logging.DEBUG, logger="sclab._r"):
            with r_output("demo") as R:
                R('warning("harmless thing"); warning("real")')

    [w] = _r_warnings(record)
    assert str(w.message) == "R reported 1 warning during demo:\n  - real"
    assert "harmless thing" in caplog.text


def test_r_messages_go_to_logger_not_stdout(capfd, caplog):
    with caplog.at_level(logging.DEBUG, logger="sclab._r"):
        with r_output("demo") as R:
            R('message("a message"); cat("a cat\\n"); print("a print")')

    out, err = capfd.readouterr()
    for text in ("a message", "a cat", "a print"):
        assert text not in out and text not in err
        assert any(
            text in r.getMessage() and r.levelno == logging.DEBUG
            for r in caplog.records
            if r.name == "sclab._r"
        )


def test_verbose_keeps_the_current_console(monkeypatch, caplog):
    from rpy2.rinterface_lib import callbacks

    seen = []
    monkeypatch.setattr(callbacks, "consolewrite_print", seen.append)
    monkeypatch.setattr(callbacks, "consolewrite_warnerror", seen.append)

    with caplog.at_level(logging.DEBUG, logger="sclab._r"):
        with r_output("demo", verbose=True) as R:
            R('message("shown"); cat("also shown\\n")')

    assert "shown\n" in seen and "also shown\n" in seen
    assert "shown" not in caplog.text


def test_state_is_restored_and_outside_warnings_are_untouched(monkeypatch):
    import rpy2.robjects as robjects
    from rpy2.rinterface_lib import callbacks

    seen = []
    monkeypatch.setattr(callbacks, "consolewrite_print", seen.append)
    monkeypatch.setattr(callbacks, "consolewrite_warnerror", seen.append)
    before = (callbacks.consolewrite_print, callbacks.consolewrite_warnerror)
    warn_before = robjects.r('getOption("warn")')[0]

    with pytest.warns(UserWarning, match="inside"):
        with r_output("demo") as R:
            R('warning("inside")')

    assert (callbacks.consolewrite_print, callbacks.consolewrite_warnerror) == before
    assert robjects.r('getOption("warn")')[0] == warn_before
    assert not seen  # nothing reached the console inside the block

    # outside sclab, R prints its (deferred) warning as usual
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        robjects.r('warning("outside")')
    assert "outside" in "".join(seen)


def test_r_error_still_raises_and_restores(monkeypatch):
    from rpy2.rinterface_lib import callbacks
    from rpy2.rinterface_lib.embedded import RRuntimeError

    before = (callbacks.consolewrite_print, callbacks.consolewrite_warnerror)
    with pytest.warns(UserWarning, match="before the error"):
        with pytest.raises(RRuntimeError, match="boom"):
            with r_output("demo") as R:
                R('warning("before the error"); stop("boom")')

    assert (callbacks.consolewrite_print, callbacks.consolewrite_warnerror) == before
