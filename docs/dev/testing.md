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
