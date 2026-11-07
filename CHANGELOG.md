# Changelog

All notable changes to this project are listed in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
