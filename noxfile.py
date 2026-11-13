"""Sessions for the tests, lint, docs and the other checks, the same locally and in CI.

Run with ``uv run --only-group nox nox -s <session>``; ``nox --list`` shows all
sessions. Each session gets its own environment in ``.nox/`` and installs the locked
dependencies of its group from ``uv.lock`` into it (``uv sync --locked``), so the
versions are the same as in CI. Arguments after ``--`` go to pytest (or to pre-commit
and mkdocs), e.g. ``nox -s tests-3.14 -- -k plot``.

The MPI sessions need ``mpiexec`` on the PATH (MS-MPI on Windows, MPICH or Open MPI on
Linux). On Windows without ``mpiexec``, the other test sessions uninstall mpi4py, which
can't load without the MPI library (as in CI).
"""

import shutil
import sys

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv"
# Reuse the environments: `uv sync --locked` still makes each one match uv.lock.
nox.options.reuse_venv = "yes"
# `nox` without -s: the quick checks, and the unit tests on the lowest and the highest
# Python version.
nox.options.sessions = ["lint", "deptry", "tests-3.11", "tests-3.14", "docs"]

# All supported versions (requires-python in pyproject.toml), and the subset for the
# slow tests (as in CI).
PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
EXTENDED_PYTHON_VERSIONS = ["3.11", "3.14"]
# The version of .python-version, for the sessions that run once.
DEFAULT_PYTHON = "3.14"

MPI_RANKS = "2"
# pytest-xdist: one worker per CPU; the tests of a module or class on the same worker.
XDIST_ARGS = ["-n", "auto", "--dist=loadscope", "--max-worker-restart=0"]
MPI_PYTEST_ARGS = ["--with-mpi", "-p", "no:xdist"]
NON_MPI = "not (mpi or mock_mpi)"
MPI = "(mpi or mock_mpi)"


def _sync(session: nox.Session, *args: str) -> None:
    """Install the locked dependencies (uv.lock) into the environment of the session.

    Parameters
    ----------
    session
        The nox session.
    *args
        Arguments for ``uv sync`` that select the groups, e.g. ``--group=test``.
    """
    session.run_install(
        "uv",
        "sync",
        "--locked",
        f"--python={session.virtualenv.location}",
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


def _sync_tests(session: nox.Session, *, mpi: bool = False) -> None:
    """Install the package and the ``test`` group; no mpi4py on Windows without MPI.

    Parameters
    ----------
    session
        The nox session.
    mpi
        Whether the session runs the MPI tests, which need ``mpiexec``.
    """
    has_mpiexec = shutil.which("mpiexec") is not None
    if mpi and not has_mpiexec:
        session.error(
            "mpiexec was not found: install MPI (MS-MPI on Windows, MPICH or Open MPI"
            " on Linux) to run the MPI tests."
        )
    _sync(session, "--no-default-groups", "--group=test")
    if sys.platform == "win32" and not has_mpiexec:
        session.log("No mpiexec on Windows: uninstalling mpi4py (needs MS-MPI).")
        session.run_install(
            "uv",
            "pip",
            "uninstall",
            f"--python={session.virtualenv.location}",
            "mpi4py",
        )


def _python(session: nox.Session) -> str:
    """Return the Python interpreter of the session, to run it with ``mpiexec``.

    Parameters
    ----------
    session
        The nox session.

    Returns
    -------
    str
        The path of the interpreter in the environment of the session.
    """
    python = shutil.which("python", path=session.bin)
    if python is None:
        session.error(f"No Python interpreter in {session.bin}.")
    return python


def _pytest(session: nox.Session, marker: str, *args: str) -> None:
    """Run the non-MPI tests with ``marker`` in parallel.

    Parameters
    ----------
    session
        The nox session.
    marker
        The marker expression of the tests to run, e.g. ``unit``.
    *args
        Extra pytest arguments.
    """
    session.run(
        "python",
        "-m",
        "pytest",
        *XDIST_ARGS,
        "-m",
        f"{marker} and {NON_MPI}",
        *args,
        *session.posargs,
    )


def _pytest_mpi(session: nox.Session, marker: str, *args: str) -> None:
    """Run the MPI tests with ``marker`` on ``MPI_RANKS`` ranks.

    Parameters
    ----------
    session
        The nox session.
    marker
        The marker expression of the tests to run, e.g. ``unit``.
    *args
        Extra pytest arguments.
    """
    session.run(
        "mpiexec",
        "-n",
        MPI_RANKS,
        _python(session),
        "-m",
        "pytest",
        *MPI_PYTEST_ARGS,
        "-m",
        f"{MPI} and {marker}",
        *args,
        *session.posargs,
        external=True,
    )


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Unit tests without MPI."""
    _sync_tests(session)
    _pytest(session, "unit", "-q")


@nox.session(python=PYTHON_VERSIONS)
def tests_mpi(session: nox.Session) -> None:
    """Unit tests with MPI and mock MPI (needs mpiexec)."""
    _sync_tests(session, mpi=True)
    _pytest_mpi(session, "unit", "-q")


@nox.session(python=EXTENDED_PYTHON_VERSIONS)
def verification(session: nox.Session) -> None:
    """Verification tests without MPI (slow)."""
    _sync_tests(session)
    _pytest(session, "verification", "-q")


@nox.session(python=DEFAULT_PYTHON)
def verification_mpi(session: nox.Session) -> None:
    """Verification tests with MPI and mock MPI (slow, needs mpiexec)."""
    _sync_tests(session, mpi=True)
    _pytest_mpi(session, "verification", "-q")


@nox.session(python=EXTENDED_PYTHON_VERSIONS)
def e2e(session: nox.Session) -> None:
    """End-to-end tests without MPI (slow)."""
    _sync_tests(session)
    _pytest(session, "e2e")


@nox.session(python=DEFAULT_PYTHON)
def e2e_mpi(session: nox.Session) -> None:
    """End-to-end tests with MPI and mock MPI (slow, needs mpiexec)."""
    _sync_tests(session, mpi=True)
    _pytest_mpi(session, "e2e")


@nox.session(python=DEFAULT_PYTHON)
def coverage(session: nox.Session) -> None:
    """Coverage of all tests, with and without MPI (slow, needs mpiexec).

    Writes ``coverage.xml``.
    """
    _sync_tests(session, mpi=True)
    session.run("python", "-m", "coverage", "erase")
    session.run(
        "python",
        "-m",
        "pytest",
        *XDIST_ARGS,
        "-m",
        NON_MPI,
        "--cov=sbo",
        "--cov-append",
        "-q",
    )
    session.run(
        "mpiexec",
        "-n",
        MPI_RANKS,
        _python(session),
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        "-m",
        "pytest",
        "-p",
        "no:xdist",
        "-m",
        MPI,
        "-q",
        external=True,
    )
    session.run("python", "-m", "coverage", "combine", "--append")
    session.run("python", "-m", "coverage", "xml")
    session.run("python", "-m", "coverage", "report")


@nox.session(python=DEFAULT_PYTHON)
def lint(session: nox.Session) -> None:
    """Pre-commit hooks on all files (Ruff, mdformat, codespell, whitespace)."""
    _sync(session, "--only-group=lint")
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    # The hooks run `uv run --no-sync <tool>`: use this environment instead of .venv.
    session.run(
        "pre-commit", *args, env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    )


@nox.session(python=DEFAULT_PYTHON)
def docs(session: nox.Session) -> None:
    """Build the docs with ``mkdocs build --strict`` (``-- serve`` for a preview)."""
    _sync(session, "--no-default-groups", "--group=docs")
    args = session.posargs or ["build", "--strict"]
    session.run("mkdocs", *args, env={"DISABLE_MKDOCS_2_WARNING": "true"})


@nox.session(python=DEFAULT_PYTHON)
def deptry(session: nox.Session) -> None:
    """Check the dependency declarations of src/ (unused, missing, transitive)."""
    _sync(session)
    session.run("deptry", "src")


@nox.session(venv_backend="none")
def audit(session: nox.Session) -> None:
    """Check the locked dependencies (uv.lock, all groups) for known vulnerabilities."""
    # Ignored: two setuptools advisories (sdist on macOS; legacy wheel install on
    # Windows), fixed in setuptools 83/84. setuptools stays <81 while "module": "mf2"
    # is supported (see pyproject.toml); remove both ignores with that pin in 2.0.0.
    session.run(
        "uv",
        "audit",
        "--locked",
        "--preview-features",
        "audit-command",
        "--ignore",
        "PYSEC-2026-3447",
        "--ignore",
        "GHSA-xmj3-9gf7-h52w",
        *session.posargs,
        external=True,
    )
