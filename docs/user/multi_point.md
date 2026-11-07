# Multi-point Optimisation

When one function evaluation takes long, evaluating several points at the same time shortens the wall-clock time. The
multi-point optimisers use MPI: every MPI rank is a process that can evaluate the objective function.

!!! warning "MPI required"

    `EgoMPI` and `EgoAsync` need an MPI implementation (OpenMPI, MPICH or MS-MPI), `mpi4py`, and at least two ranks.
    Otherwise they fall back to `EgoSequential` with a warning in the log.

## Synchronous or Asynchronous

|                  | `EgoMPI` (synchronous)                                  | `EgoAsync` (asynchronous)                                                |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| Points per round | `number_new_points` per iteration, at most one per rank | One new point for every rank that becomes free                           |
| Who evaluates    | All ranks, including rank 0                             | Ranks 1 to N-1; rank 0 only runs the optimiser                           |
| Waits for        | All points of the iteration                             | Nothing: a free rank gets a new point immediately                        |
| Iterations       | Yes (`max_iterations` applies)                          | No (results are numbered in arrival order; `max_iterations` is not used) |
| Best when        | Evaluation times are similar                            | Evaluation times differ (e.g. multi-fidelity)                            |

## How Several Points Are Chosen

An infill strategy proposes one point from the current surrogate. To propose more points before the results are known,
the optimiser temporarily adds the pending points to the surrogate with an assumed value, and asks the infill strategy
again. The assumed value is set by `multipoint_algorithm` (under `"optimisation"`):

- `"KB"` (Kriging Believer, default): the surrogate's own prediction at the pending point, at the point's fidelity when
    the surrogate supports that (`HK`, `MFK`), and the HF prediction otherwise.
- `"CL"` (Constant Liar): one constant value for all pending points, the minimum, mean or maximum (`constant_liar_lie`)
    of the sample values of the highest fidelity that has samples.

Proposed points that duplicate a sample, a pending point, or another point of the same batch are replaced by the same
location at a higher fidelity, or by a random point if no higher fidelity is available.

## Running

Start the script with the MPI launcher, with one process per rank. Every rank runs the same script:

```bash
mpiexec -n 4 python examples/04_multi_point_optimisation.py
```

`sbo.run()` must be called on every rank. Only rank 0 returns the optimum; the other ranks return an empty dict, so save
or print the results on rank 0 only:

```python
from mpi4py import MPI

optimum = sbo.run()
if MPI.COMM_WORLD.Get_rank() == 0:
    print(optimum)
```

On a cluster, see [Running on HPC](hpc.md).

## Example

`examples/04_multi_point_optimisation.py` minimises `x²` with `EgoAsync`:

```python
--8<-- "examples/04_multi_point_optimisation.py"
```
