# Changelog

All notable changes to this project are listed in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The README and the documentation mark `CoKRG` and `FilteredGEI` as planned for 1.4.0 and `CEI` for 1.5.0, instead of
    "not implemented" or "not functional".

### Development

- Ruff replaces black, isort and flake8 (with its five plugins) for linting, import sorting and formatting, configured
    in `pyproject.toml`. The code is reformatted with Ruff, and the findings it reports that flake8 missed are fixed
    (`raise ... from`, `zip(..., strict=...)` and misplaced `noqa` comments).
- A workflow (`project-in-dev.yml`) moves the issues linked in a merged pull request into `dev` (`Part of #N`) to **In
    dev** on the project board.
- A `codespell` pre-commit hook checks the spelling: common typos, and British spelling with a project dictionary of
    American-to-British spellings (`.codespell-en-GB.txt`), configured in `pyproject.toml`.
- Ruff's security rules (`S`, flake8-bandit) are enabled; `assert` is allowed in the tests. Loading a pickle in
    `load_model_from_file` is documented as safe for trusted files only.
- A workflow (`audit.yml`) checks the locked dependencies (`uv.lock`) for known vulnerabilities with `uv audit`, when
    the dependencies change and weekly.
- Dependabot opens a monthly pull request into `dev` that updates the Python packages in `uv.lock` (grouped), within the
    version ranges in `pyproject.toml`. `v1.0.0` keeps the environment used for the thesis results. The quarterly GitHub
    Actions updates now also go into `dev` instead of `main`.
- A `noxfile.py` with a nox session for every check: the unit tests on Python 3.11 to 3.14 (`tests-3.11` ...
    `tests-3.14`), the MPI unit tests, the verification and end-to-end tests (with and without MPI), coverage, `lint`
    (pre-commit), `docs`, `deptry` and `audit`. Each session installs its locked dependency group from `uv.lock` into
    its own environment. The workflows run these sessions, so the same commands run locally and in CI. nox is in the new
    `nox` dependency group: `uv run --only-group nox nox -s <session>`. This replaces the manual `.venv-*` environments
    for testing other Python versions.

## [1.1.0] - 2026-11-10

### Added

- `sbo.benchmarks`: the eleven benchmark problems of the thesis (Forrester, Bohachevsky, Booth, Branin, Currin,
    Himmelblau, six-hump camelback, Park91a, Park91b, Hartmann6 and borehole) as part of the package. Each is a
    `MultiFidelityFunction` with its functions per fidelity, bounds and known optimum (`x_opt`, `f_opt`); also
    `ALL_BENCHMARKS`, `get_benchmark` and the `FunctionSet` protocol. They give the same values as mf2 2022.6.0.
- `Config.from_json` accepts `"module": "sbo.benchmarks"`, and an object (such as a `MultiFidelityFunction`) as well as
    a class as the attribute.
- `BENCHMARK_FUNCTION_DELAY` in `sbo.constants`.
- Formulas in the API documentation (LaTeX, rendered with MathJax) for the benchmark problems, the infill strategies
    `EI`, `GEI`, `TwoStep` and `FilteredGEI`, the surrogates `HK`, `KRG`, `MFK` and `RBF`, the optimisation base class
    and the default settings.

### Changed

- Supports Python 3.11 to 3.14 (`requires-python = ">=3.11,<3.15"`), and the CI tests all four versions. Python 3.15
    follows once smt publishes wheels for it.
- The locked dependencies (`uv.lock`) are updated, among others NumPy 2.5 and SciPy 1.18 (NumPy 2.4 and SciPy 1.17 on
    Python 3.11, which the newest versions no longer support), matplotlib 3.11 and smt 2.15.
- The HPC container uses Python 3.14.

### Deprecated

- `"module": "mf2"` in `Config.from_json`: it now uses the problem of the same name from `sbo.benchmarks`, so mf2 is no
    longer needed, and gives a `DeprecationWarning`.
- `MF2_KNOWN_OPTIMA` (use the `f_opt` attribute of the benchmark problems) and `MF2_FUNCTION_DELAY` (use
    `BENCHMARK_FUNCTION_DELAY`).
- The `benchmarks` extra, which is now empty.

All of these will be removed in version 2.0.0.

### Removed

- The dependency on mf2 (GPLv3): the `benchmarks` extra and the `test` group no longer install it, so the package and
    its documentation no longer need GPL notes.
- Support for Python 3.10. Use version 1.0.0, the version of the thesis results, on Python 3.10.

### Fixed

- NumPy `DeprecationWarning`s "Conversion of an array with ndim > 0 to a scalar", which newer NumPy versions turn into
    errors: `TwoStep`, the surrogate base class and the 1D surrogate plots of `SBO` now convert the 2D predictions of
    SMT to scalars explicitly.
- `SBO.optimum` and `sbo()` now return an empty dict on MPI worker ranks, like `SBO.run()`. They returned `None`, and
    every later call to `optimum` ran the optimisation again.

### Security

- The updated locked dependencies fix the open Dependabot alerts, except those for setuptools (see below). Pillow 13.0.0
    (pulled in by matplotlib) is the only one that affects the runtime; certifi, urllib3, virtualenv, filelock, black,
    pymdown-extensions and mkdocs-material are only used for development. setuptools stays below 81 while
    `"module": "mf2"` is supported, because mf2 imports `pkg_resources`; it is only needed to build and install the
    package.

## [1.0.0] - 2026-11-07

The first public release: the code of the MSc thesis
[*Combining an Asynchronous Multi-Point and a Multi-Fidelity Infill Strategy for Surrogate-Based Optimisation*](https://resolver.tudelft.nl/uuid:baf55719-5cc1-42ae-b9f7-55af0f2292d8).
This is the code exactly as used for the thesis results; it is not actively developed.

### Added

- Surrogate-based optimisation with `SBO` and `Config`: bounds, objective functions per fidelity, initial samples (Latin
    hypercube or user-supplied), validated settings and stopping criteria, and reproducible runs with a seeded random
    number generator.
- Surrogates: `RBF`, `KRG` (Kriging), `HK` (hierarchical Kriging) and `MFK` (SMT multi-fidelity Kriging, not verified).
- Infill strategies: `EI`, `GEI`, `TwoStep` and `Random`; `FilteredGEI` is included but needs a Co-Kriging surrogate,
    which isn't implemented yet.
- Optimisation algorithms: `EgoSequential`, and the multi-point `EgoMPI` (synchronous) and `EgoAsync` (asynchronous)
    with MPI.
- Results: optimisation data per iteration, plots and a log file.
- `Config.from_json` to define a problem with the mf2 benchmark functions (optional `benchmarks` extra).
- Example scripts in `examples/`.
- Unit, verification, end-to-end and MPI tests.
- Apptainer container and an example SLURM job script for running on HPC clusters.
- Documentation with MkDocs: user guide, developer guide and API reference.

[1.0.0]: https://github.com/ioostmeijer/multi-fidelity-multi-point-sbo/releases/tag/v1.0.0
[1.1.0]: https://github.com/ioostmeijer/multi-fidelity-multi-point-sbo/releases/tag/v1.1.0
[unreleased]: https://github.com/ioostmeijer/multi-fidelity-multi-point-sbo/compare/v1.1.0...HEAD
