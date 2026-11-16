# Testing

The tests are in `tests/` and run with `pytest`. Every test gets a marker from the folder it is in, so a level can be
selected with `-m`.

## Test Levels

| Folder                      | Marker         | What it checks                                                                                                                                                           | Speed |
| --------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| `tests/tests_unittest/`     | `unit`         | Each class and function on its own, mostly with dummy components.                                                                                                        | Fast  |
| `tests/tests_verification/` | `verification` | Results against reference implementations: SMT's Kriging, RBF and EGO, a quadrature of GEI, the hierarchical Kriging equations, and known optima of benchmark functions. | Slow  |
| `tests/tests_e2e/`          | `e2e`          | Complete optimisations on the cheap 1D Forrester function, single- and multi-fidelity: checks that a full run works, not accuracy.                                       | Slow  |

Tests that need MPI are marked `mpi` (real MPI, run with `mpiexec`) or `mock_mpi` (the `mockmpi` package, which
simulates several ranks in one process). Both are skipped on Windows CI.

```bash
pytest -m unit                              # fast unit tests
pytest -m verification                      # verification tests
pytest -m e2e                               # end-to-end tests
pytest tests/tests_unittest/test_sbo.py -k plot     # a subset
mpiexec -n 4 python -m pytest --with-mpi -p no:xdist -m "mpi or mock_mpi"   # MPI tests
```

`pytest-xdist` is installed, so `-n auto` runs the tests in parallel (not for the MPI tests).

The tests write no files: each test runs in its own temporary working directory. To look at the comparison plots of the
Kriging and RBF verification tests (custom against SMT), set `SBO_TEST_PLOT_DIR` to a directory, and the tests save them
there:

```bash
SBO_TEST_PLOT_DIR=verification_plots pytest tests/tests_verification/surrogate   # plots in verification_plots/KRG and /RBF
```

## Deprecation Warnings

`filterwarnings` in `[tool.pytest.ini_options]` (`pyproject.toml`) turns a `DeprecationWarning` or `FutureWarning` that
is raised from `sbo` code into an error, so a deprecated call in the own code fails the tests. The same warnings from
other packages (smt, matplotlib, ...) raised in their own code don't, so a new Python or dependency version doesn't
block the tests. A test that checks a deprecation of `sbo` itself catches the warning with `pytest.warns`.

## Python Versions

The package supports Python 3.11 to 3.14 (`requires-python = ">=3.11,<3.15"`). The CI tests:

| Workflow       | Jobs                                                                                   | Python versions        |
| -------------- | -------------------------------------------------------------------------------------- | ---------------------- |
| `ci.yml`       | Unit tests on Linux and Windows (MPI unit tests on Linux)                              | 3.11, 3.12, 3.13, 3.14 |
| `extended.yml` | Verification and end-to-end tests on Linux and Windows                                 | 3.11 and 3.14          |
| `extended.yml` | MPI verification, MPI end-to-end tests and coverage (Linux)                            | 3.14                   |
| `lint.yml`     | pre-commit; `docs.yml`: building the docs; `ci.yml`: deptry (once, on Linux with 3.14) | 3.14                   |

The development environment (`.venv`) uses the version in `.python-version`. To test the other versions locally, use the
nox sessions below: each version gets its own environment. Testing the lowest and the highest version (`tests-3.11`,
`tests-3.14`) is enough; the CI covers the versions in between.

## nox Sessions

`noxfile.py` defines a [nox](https://nox.thea.codes/) session for every check that CI runs, and the workflows run these
sessions, so a check runs the same way locally and in CI. Each session has its own environment in `.nox/` (created with
uv, which also downloads a missing Python version) and installs the locked versions of its dependency group from
`uv.lock` into it (`uv sync --locked`), so it doesn't touch `.venv`. nox itself is in the `nox` dependency group:

```bash
uv run --only-group nox nox --list                    # all sessions
uv run --only-group nox nox                           # lint, deptry, tests-3.11, tests-3.14 and docs
uv run --only-group nox nox -s tests-3.14             # one session
uv run --only-group nox nox -s tests                  # unit tests on all four versions
uv run --only-group nox nox -s tests-3.14 -- -k plot  # arguments after -- go to pytest
```

| Session                       | Runs                                                                 | Python versions        | In CI          |
| ----------------------------- | -------------------------------------------------------------------- | ---------------------- | -------------- |
| `tests`                       | Unit tests without MPI, in parallel (`-n auto`)                      | 3.11, 3.12, 3.13, 3.14 | `ci.yml`       |
| `tests_mpi`                   | Unit tests with MPI and mock MPI (`mpiexec -n 2`, `--with-mpi`)      | 3.11, 3.12, 3.13, 3.14 | `ci.yml`       |
| `verification`, `e2e`         | Verification and end-to-end tests without MPI                        | 3.11 and 3.14          | `extended.yml` |
| `verification_mpi`, `e2e_mpi` | Verification and end-to-end tests with MPI and mock MPI              | 3.14                   | `extended.yml` |
| `coverage`                    | All tests with coverage, with and without MPI; writes `coverage.xml` | 3.14                   | `extended.yml` |
| `lint`                        | The pre-commit hooks on all files                                    | 3.14                   | `lint.yml`     |
| `docs`                        | `mkdocs build --strict` (`-- serve` for a live preview)              | 3.14                   | `docs.yml`     |
| `deptry`                      | `deptry src`                                                         | 3.14                   | `ci.yml`       |
| `audit`                       | `uv audit --locked` (with the ignored advisories, see `noxfile.py`)  | (no environment)       | `audit.yml`    |

The MPI sessions (`tests_mpi`, `verification_mpi`, `e2e_mpi`, `coverage`) need `mpiexec` on the PATH: MS-MPI on Windows,
MPICH or Open MPI on Linux. On Windows without `mpiexec`, the other test sessions uninstall mpi4py from their
environment, as it can't load without MS-MPI (the same as on Windows CI). The OS-specific setup stays in the workflows:
installing MPI (in `.github/actions/setup-env`) and the Windows long-paths fix.

The environments are reused between runs, and `uv sync --locked` updates them when `uv.lock` changes; delete `.nox/` to
start fresh. For a quick run while working on the code, `uv run pytest ...` in `.venv` (above) is still the fastest.

!!! note "Precision limits"

    Some unit tests compare to `rel=1e-9` or `1e-12`, and the Kriging tests compare with SMT to `rtol=1e-5`. A newer NumPy
    or SciPy can change the last digits. If a test fails on only one version, look at the size of the difference before
    loosening a limit: a large difference means the results really changed.

## Helpers

`tests/helpers/` is on the Python path of the tests (`pythonpath` in `pyproject.toml`):

| File                   | Provides                                                                                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dummy_surrogate.py`   | `DummySurrogate`, `DummySurrogateWithVariance`, ...: surrogates with fixed predictions, for testing infill strategies and optimisers without training a model. |
| `dummy_infill.py`      | A dummy infill strategy with a fixed output, for testing optimisers.                                                                                           |
| `logger_helper.py`     | `_make_logger()` for a test logger, and `_find_log_file()` to check log output.                                                                                |
| `template_e2e.py`      | `template_for_e2e_test(used_algorithms, settings, ...)`: runs a full optimisation on Forrester and checks the optimum within `RESULT_TOLERANCE` (`1e-2`).      |
| `surrogate_plotter.py` | Plots a surrogate against the true function, for inspecting a failing verification test (see `SBO_TEST_PLOT_DIR` above).                                       |

`tests/conftest.py` runs every test in its own temporary working directory, resets the `Config` singleton and the
loggers after every test, and adds the markers.

## Writing Tests for a New Component

- **Unit tests** (required): one file per class in the matching folder of `tests/tests_unittest/`. Use the dummy
    surrogates and infill strategies to isolate the class, and cover the error cases (`pytest.raises`) as well.
- **Verification test** (for the status "Verified"): compare the results with an independent reference, such as an
    existing library, an analytical result or a numerical integration, within a stated tolerance.
- **End-to-end test** (optional): add the new algorithm to the combinations in `tests/tests_e2e/`.

A new test must not leave files behind. Relative paths already end up in the temporary working directory, but write to
the `tmp_path` fixture when the test checks the files, and pass a logger with `to_file=False` or a `tmp_path` folder.
