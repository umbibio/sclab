import nox

# Use uv for faster dependency installation
nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = [
    # "3.10",
    # "3.11",
    "3.12",
    # "3.13",
    "3.14",
]


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize(
    "scipy_version",
    [
        # "",
        "==1.17.*",
        # "==1.15.*",
        # "==1.14.*",
    ],
)
@nox.parametrize(
    "numpy_version",
    [
        # "",
        "==2.4.*",
        # "==2.3.*",
        # "==2.2.*",
        # "==2.1.*",
        # "==2.0.*",
    ],
)
def tests(session: nox.Session, numpy_version: str, scipy_version: str) -> None:
    """Run tests on multiple Python versions."""

    dependencies = [
        f"numpy{numpy_version}",
        f"scipy{scipy_version}",
        ".[test,scanpy]",
    ]

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
