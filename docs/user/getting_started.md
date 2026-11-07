# Getting Started

## Requirements

- **Python 3.10**
- **An MPI implementation** (OpenMPI or MPICH on Linux and macOS, MS-MPI on Windows) for multi-point optimisation.
    Without MPI the package still installs and runs, but multi-point runs fall back to single-point `EgoSequential` with
    a warning.

## Installation

```bash
pip install -e .                  # the package and its runtime dependencies
pip install -e ".[benchmarks]"    # also the mf2 benchmark functions (for Config.from_json and example 05)
```

The benchmark functions from [mf2](https://mf2.readthedocs.io/) are optional.

To reproduce the thesis results, install the exact, locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --no-dev --extra benchmarks   # runtime dependencies and mf2, at the locked versions
```

To run on a cluster, see [Running on HPC](hpc.md).

## First Optimisation

`examples/01_basic_optimisation.py` minimises `f(x) = x²` on `[-5, 5]` with sequential EGO, Expected Improvement and a
Kriging surrogate:

```python
--8<-- "examples/01_basic_optimisation.py"
```

Run it with `python examples/01_basic_optimisation.py`. `sbo.run()` returns the best point found, as a dict:

```python
{"coord": array([0.]), "value": 0.0, "fid": 0, ...}
```

`coord` is the location, `value` the objective value and `fid` the fidelity level (0 is the high-fidelity function).
[Results, Plots and Logs](results.md) explains the other keys, the data of every iteration, and how to plot it.

The three entries of `used_algorithms` choose the components:

- `"surrogate"`: the model fitted to the samples, e.g. `KRG` (Kriging) or `HK` (hierarchical Kriging, multi-fidelity);
- `"infill"`: how the next point is chosen, e.g. `EI` (Expected Improvement);
- `"optimisation"`: the loop, e.g. `EgoSequential` (one point at a time) or `EgoAsync` (parallel, with MPI).

All options and their status are listed on the [home page](../index.md#included-algorithms).

## Examples

The `examples/` folder contains a runnable script for each common use case. Each one is shown and explained on the page
of its topic:

| Script                              | Shows                                                                    | Explained in                                                    |
| ----------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `01_basic_optimisation.py`          | The minimal single-fidelity setup                                        | This page                                                       |
| `02_custom_settings.py`             | Changing settings (GEI with `g = 5`, a budget of 50 calls, a fixed seed) | [Settings and Stopping Criteria](settings.md#example)           |
| `03_multi_fidelity_optimisation.py` | Three fidelity levels with `HK` and `TwoStep`                            | [Multi-fidelity Optimisation](multi_fidelity.md#example)        |
| `04_multi_point_optimisation.py`    | Asynchronous multi-point optimisation with `EgoAsync` and MPI            | [Multi-point Optimisation](multi_point.md#example)              |
| `05_run_from_config/`               | Loading a problem from a JSON file (needs the `benchmarks` extra)        | [Defining a Problem](configuration.md#loading-from-a-json-file) |

## Next Steps

- Write your own objective function and problem: [Defining a Problem](configuration.md).
- Change the budget or the stopping criteria: [Settings and Stopping Criteria](settings.md).
- Look at the results: [Results, Plots and Logs](results.md).
