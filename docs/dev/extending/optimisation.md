# Optimisation Algorithm

An optimisation algorithm runs the loop: it asks the infill strategy for points, evaluates them, updates the surrogate
and decides when to stop. Add it as `src/sbo/optimisation/<Name>.py`, derived from `OptimisationBase`
(`src/sbo/optimisation/Base.py`), and follow the [checklist](index.md#checklist). `EgoSequential` is the simplest
complete example to start from.

## Method to Implement

```python
def solve(self, surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs) -> dict:
    ...
```

`solve` receives the surrogate (already holding the initial samples) and the infill strategy, and returns the optimum: a
point dict with at least `coord`, `value` and `fid`. `SBO.run()` calls it through `OptimisationBase.__call__`.

## Helpers from OptimisationBase

| Helper                                                                                                                                                                                      | Use                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `infill(surrogate)`                                                                                                                                                                         | New tasks and the infill value: `tuple[list[list[dict]], float]`. The tasks are already nested if `nested` is on.                                                                                                                                                   |
| `self._remove_duplicates(tasks, surrogate, pending_points=None)`                                                                                                                            | Replaces points that duplicate a sample, a pending point or an earlier point of the same batch (by a higher fidelity at the same location, or a random point). Call it before evaluating.                                                                           |
| `self.evaluate_point(point, inplace=False, bypass_delay=False)`                                                                                                                             | Evaluates the point's fidelity function and adds `value`, `start_time`, `end_time` and `mpi_rank`; applies `function_delay` unless `bypass_delay` is `True`. With `inplace=False` it returns a copy.                                                                |
| `self._kriging_believer(surrogate, infill, sample_locations)`, `self._constant_liar(surrogate, infill, sample_locations)`                                                                   | Propose new tasks while `sample_locations` (pending points) are not evaluated yet; choose with `self.config.get("optimisation.multipoint_algorithm")`. Both return `(tasks, inf)` and restore the surrogate afterwards.                                             |
| `self._run_end_iteration(iteration, function_calls, best_val, prev_best_val, infill_value, best_coord, surrogate, start_time_iteration, current_surrogate_optimum, prev_surrogate_optimum)` | Stores the data of the iteration in `self._optimisation_data` and returns `True` if a stopping criterion is met. Use an `int` iteration counter starting at 0; a `str` key means there are no iterations (as in `EgoAsync`), and `max_iterations` is then not used. |
| `self._count_final_evaluation(function_calls)`                                                                                                                                              | Counts the final HF evaluation of the surrogate optimum; the function-call criteria reserve one HF call for it.                                                                                                                                                     |
| `self.update_number_of_points(n)`                                                                                                                                                           | Changes `number_new_points`.                                                                                                                                                                                                                                        |

The data of the run is in `self._optimisation_data`, which `SBO.optimisation_data` returns; its structure is described
in [Results, Plots and Logs](../../user/results.md#optimisation-data).

## Structure of `solve`

```python
def solve(self, surrogate, infill, *args, **kwargs) -> dict:
    self._optimisation_data["init"] = {"sample_points": copy.deepcopy(surrogate.samples), ...}
    function_calls = dict.fromkeys(range(self.config.num_fidelities), 0)
    iteration = 0
    start_time = time.time()

    while True:
        iteration_start = time.time()
        tasks, infill_value = infill(surrogate)
        tasks = self._remove_duplicates(tasks, surrogate)
        points = [point for task in tasks for point in task]
        for point in points:
            self.evaluate_point(point, inplace=True)
            function_calls[point["fid"]] += 1
        surrogate.add_training_data(points)
        surrogate.train()

        best_point = min(surrogate.samples[0], key=lambda p: p["value"])
        surrogate_optimum = {**surrogate.get_optimum(fidelity=0), "fid": 0}
        stop = self._run_end_iteration(
            iteration=iteration, function_calls=function_calls, best_val=best_point["value"], ...
        )
        if stop:
            break
        iteration += 1

    # Evaluate the surrogate optimum on HF (no delay) and count it
    final = self.evaluate_point({**surrogate.get_optimum(), "fid": 0}, bypass_delay=True)
    self._count_final_evaluation(function_calls)
    best_point = min([best_point, final], key=lambda p: p["value"])

    self._optimisation_data["solution"] = best_point
    self._optimisation_data["total"]["time_elapsed"] = time.time() - start_time
    return best_point
```

Track the best point and the surrogate optimum of the previous iteration as well; `_run_end_iteration` needs both for
the convergence criteria.

## Parallel Algorithms

For an algorithm that uses MPI:

- Return `EgoSequential` from `__new__` when MPI is not available or there are fewer ranks than needed (see
    [Conditional Creation](index.md#conditional-creation) and `EgoMPI.__new__`), and accept an `mpi_comm` keyword
    argument for testing with a mock communicator.
- `SBO.run()` calls `solve` on every rank. Only rank 0 returns the optimum; the other ranks return `None` or `{}`.
- Count every evaluated point in `function_calls`, including results that arrive after the loop has stopped.
- Use `self.config.rng` for random choices, so all ranks and runs stay reproducible.

## Settings

The stopping criteria and multi-point settings are shared by all optimisers (`OPTIMISATION_ALL_DEFAULT_SETTINGS`). Reuse
them with `MYOPTIMISER_DEFAULT_SETTINGS = OPTIMISATION_ALL_DEFAULT_SETTINGS`, or extend them with
`{**OPTIMISATION_ALL_DEFAULT_SETTINGS, ...}`.
