# Changelog

All notable changes to this project are listed in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Development

- Ruff replaces black, isort and flake8 (with its five plugins) for linting, import sorting and formatting, configured
    in `pyproject.toml`. The code is reformatted with Ruff, and the findings it reports that flake8 missed are fixed
    (`raise ... from`, `zip(..., strict=...)` and misplaced `noqa` comments).

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
