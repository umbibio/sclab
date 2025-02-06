import nox

# Use uv for faster dependency installation
nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run tests on multiple Python versions."""
    session.install(".[test]")
    session.run("pytest", *session.posargs)
