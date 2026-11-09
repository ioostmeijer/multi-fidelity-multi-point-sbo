# Changelog

All notable changes to this project are listed in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `sbo.benchmarks`: the eleven benchmark problems of the thesis (Forrester, Bohachevsky, Booth, Branin, Currin,
    Himmelblau, six-hump camelback, Park91a, Park91b, Hartmann6 and borehole) as part of the package. Each is a
    `MultiFidelityFunction` with its functions per fidelity, bounds and known optimum (`x_opt`, `f_opt`); also
    `ALL_BENCHMARKS`, `get_benchmark` and the `FunctionSet` protocol. They give the same values as mf2 2022.6.0.
- `Config.from_json` accepts `"module": "sbo.benchmarks"`, and an object (such as a `MultiFidelityFunction`) as well as
    a class as the attribute.
- `BENCHMARK_FUNCTION_DELAY` in `sbo.constants`.

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
[unreleased]: https://github.com/ioostmeijer/multi-fidelity-multi-point-sbo/compare/v1.0.0...HEAD
