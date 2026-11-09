# Troubleshooting

Most problems are reported in the log; see [Logging](results.md#logging) for where it is.

## My Multi-point Run Is Sequential

The log says that `EgoMPI` or `EgoAsync` continues with `EgoSequential`. Check that:

- an MPI implementation and `mpi4py` are installed (`python -c "from mpi4py import MPI; print(MPI.Get_version())"`);
- the script is started with the MPI launcher and at least two ranks: `mpiexec -n 4 python script.py`;
- on a cluster, the container is started once per rank by the host MPI (see [Running on HPC](hpc.md)).

## A Setting Has No Effect

Settings with an unknown key, a wrong type or an invalid value are ignored or replaced, with a warning (see
[Validation](settings.md#validation)). Common causes:

- the setting belongs to another algorithm than the selected one (e.g. `g` without `"infill": "GEI"`);
- an algorithm setting is at the top level instead of under `"surrogate"`, `"infill"` or `"optimisation"`.

Set `"strict_validation": True` to get an error instead of a warning.

## The Optimisation Stops Too Early or Too Late

Check which stopping criterion was met (logged at INFO level) and the
[Stopping Criteria](settings.md#stopping-criteria). The convergence criteria with their defaults of `1e-6` can stop a
run early on a flat function; set them to `None` to run until the budget is used.

## The Algorithm Is Not the One I Selected

`HK` with one function becomes `KRG`, `FilteredGEI` with one function becomes `GEI`, and `EgoMPI`/`EgoAsync` without MPI
become `EgoSequential`; see [Algorithms](configuration.md#algorithms). `sbo.config.used_algorithms` shows the algorithms
in use.

## A Second Problem Uses the First Problem's Settings

`Config` is a singleton: a second `Config(...)` re-initialises the same instance, and every existing `SBO` sees the new
problem. Create a new `SBO` after the new `Config`, or call `Config.destroy_instance()` first; see
[Config Is a Singleton](configuration.md#config-is-a-singleton).

## `ModuleNotFoundError: No module named 'mf2'`

The benchmark problems are part of the package (`sbo.benchmarks`), so mf2 is no longer needed. Code that imports mf2
itself still needs it: replace, for example, `from mf2 import forrester` with `from sbo.benchmarks import FORRESTER`
(the problem names are lower case, e.g. `FORRESTER.name == "forrester"`). In a JSON file for `Config.from_json`, use
`"module": "sbo.benchmarks"`; `"module": "mf2"` still works without mf2, with a `DeprecationWarning`.

## `ValueError: Not all ub > lb.`

Every upper bound must be strictly larger than its lower bound, and `lb` and `ub` must have the same length. To move
both bounds of an existing config, use `config.set_bounds(lb, ub)`.

## `NotImplementedError` When Creating the Optimiser

`CEI` and `CoKRG` are not implemented yet, and `FilteredGEI` needs `CoKRG` for more than one fidelity. See the status of
every algorithm on the [home page](../index.md#included-algorithms).
