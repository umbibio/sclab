import nox

# Use uv for faster dependency installation
nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

DEFAULT_PYTHON_VERSION = "3.12"


PYTHON_VERSIONS = [
    "3.10",
    "3.11",
    "3.12",
    "3.13",
    "3.14",
]


NUMPY_VERSIONS = [
    "==2.4.*",
    "==2.3.*",
    "==2.2.*",
    "==2.1.*",
]


SCIPY_VERSIONS = [
    "==1.17.*",
    "==1.15.*",
    "==1.14.*",
]


ANNDATA_VERSIONS = [
    "==0.12.*",
    "==0.11.*",
]


TRAITLETS_VERSIONS = [
    "==5.14.*",
    "==5.13.*",
    "==5.12.*",
    "==5.11.*",
]


def _run_tests(session: nox.Session, **dep_versions: dict[str, str]) -> None:
    """Run tests on multiple Python versions."""

    dependencies = [
        f"{pkg_name}{version}" for pkg_name, version in dep_versions.items()
    ]
    dependencies.append(".[test,scanpy]")

    dry_run_output = session.run(
        "uv",
        "pip",
        "install",
        *dependencies,
        "--dry-run",
        silent=True,
        success_codes=[0, 1],
    )

    if (
        "No solution found" in dry_run_output
        or "No matching distribution" in dry_run_output
    ):
        session.skip(
            f"uv cannot resolve {' '.join(dependencies)} for Python {session.python}"
        )

    session.install(*dependencies)

    session.run("pytest", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def tests_python(session: nox.Session, **dep_versions: dict[str, str]) -> None:
    _run_tests(session, **dep_versions)


@nox.session(python=DEFAULT_PYTHON_VERSION)
@nox.parametrize("scipy", SCIPY_VERSIONS)
def tests_scipy(session: nox.Session, **dep_versions: dict[str, str]) -> None:
    _run_tests(session, **dep_versions)


@nox.session(python=DEFAULT_PYTHON_VERSION)
@nox.parametrize("numpy", NUMPY_VERSIONS)
def tests_numpy(session: nox.Session, **dep_versions: dict[str, str]) -> None:
    _run_tests(session, **dep_versions)


@nox.session(python=DEFAULT_PYTHON_VERSION)
@nox.parametrize("anndata", ANNDATA_VERSIONS)
def tests_anndata(session: nox.Session, **dep_versions: dict[str, str]) -> None:
    _run_tests(session, **dep_versions)


@nox.session(python=DEFAULT_PYTHON_VERSION)
@nox.parametrize("traitlets", TRAITLETS_VERSIONS)
def tests_traitlets(session: nox.Session, **dep_versions: dict[str, str]) -> None:
    _run_tests(session, **dep_versions)
