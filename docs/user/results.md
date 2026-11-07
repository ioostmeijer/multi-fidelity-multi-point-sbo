# Results, Plots and Logs

## The Optimum

`sbo.run()` (or `sbo()`, or the `sbo.optimum` property) returns the best point found:

| Key                      | Meaning                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| `coord`                  | Location of the optimum, `np.ndarray` of shape `(n_dimensions,)`.                                |
| `value`                  | High-fidelity objective value at `coord`.                                                        |
| `fid`                    | Fidelity level; always `0` (high fidelity) for the optimum.                                      |
| `start_time`, `end_time` | Wall-clock times (`time.time()`) of the evaluation, if the point was evaluated by the optimiser. |
| `mpi_rank`               | The MPI rank that evaluated the point.                                                           |

The optimum is the best of the high-fidelity samples and of the optimum of the final surrogate, which is evaluated once
more on the high-fidelity function at the end of the run. In multi-point runs only rank 0 returns it; the other ranks
return `{}`.

## Optimisation Data

`sbo.optimisation_data` holds the data of the whole run, as a dict with one entry per iteration plus a few summary
entries:

| Key                | Contents                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `"config"`         | The problem: bounds, function names, number of fidelities and dimensions, settings and used algorithms.                 |
| `"init"`           | The initial samples (`sample_points`) before the first iteration.                                                       |
| `0`, `1`, `2`, ... | One entry per iteration (see below). For `EgoAsync`: `"000000"`, `"000001"`, ... per received result, in arrival order. |
| `"final"`          | `EgoAsync` only: the state after all workers stopped.                                                                   |
| `"total"`          | The latest values, the total `function_calls` and the total `time_elapsed` of the run.                                  |
| `"solution"`       | The optimum (the same dict as returned by `run()`).                                                                     |

Each iteration entry contains:

| Key                                   | Contents                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `best_value`, `best_coord`            | Best high-fidelity sample so far.                                              |
| `surrogate_optimum`                   | Optimum of the surrogate (`coord`, `value`), used by the convergence criteria. |
| `infill_value`                        | Infill value of the selected point(s), e.g. the expected improvement.          |
| `function_calls`, `function_calls_HF` | Number of calls so far, per fidelity (`{fid: count}`) and for the HF function. |
| `sample_points`                       | All samples so far, per fidelity (`{fid: [points]}`).                          |
| `time_elapsed`                        | Duration of the iteration in seconds.                                          |
| `time_since_start`                    | `EgoAsync` only: seconds since the start of the run.                           |

For example, the convergence history:

```python
data = sbo.optimisation_data
iterations = [k for k in data if isinstance(k, int)]
history = [data[k]["best_value"] for k in iterations]
```

## Saving and Loading

`save_optimisation_data` writes the optimisation data as a pickle file to the `save_folder` setting (default
`"results"`, created if needed):

```python
sbo.save_optimisation_data("my_run")     # -> results/my_run.pkl
```

Load it again with `pickle`:

```python
import pickle

with open("results/my_run.pkl", "rb") as f:
    data = pickle.load(f)
```

## Plotting

For 1D and 2D problems, `plot_surrogate` draws the surrogate, the samples and the next sample location for each
iteration, one subplot per iteration:

```python
sbo.plot_surrogate(
    "My run",
    filename="my_run.png",          # relative paths are saved in save_folder; None: don't save
    show=False,                     # True: open a window
    up_to=5,                        # only iterations before 5 (optional)
    iterations=[0, 2, 4],           # or exactly these iterations (optional)
    true_functions_plot=[0],        # also draw the true function(s) of these fidelities (1D only)
    fid_samples_plot=[0, 1],        # only show samples of these fidelities (default: all)
)
```

In 1D the plot also shows the 95% confidence band of the surrogate (if it has a variance) and the infill function on a
second axis. Plotting temporarily re-trains the surrogate on the samples of each iteration and restores it afterwards.
For problems with more than two dimensions, `plot_surrogate` only logs a warning.

## Logging

By default, `Config` uses the logger `default_logger`, which writes to `logs/default_logger_<timestamp>.log` in the
working directory (one file per rank in MPI runs) and also prints warnings and errors to the screen.

To log somewhere else, pass your own logger. `setup_logger` creates one:

```python
import logging
from sbo.utils.Logger import setup_logger

logger = setup_logger("my_run", level=logging.INFO, to_file=False)   # print to stdout
config = Config(..., logger=logger)
```

| `setup_logger` argument | Default            | Meaning                                              |
| ----------------------- | ------------------ | ---------------------------------------------------- |
| `name`                  | `"default_logger"` | Name of the logger and of the log file.              |
| `level`                 | `logging.INFO`     | Lowest level that is logged.                         |
| `save_dir`              | `"logs"`           | Folder for the log file.                             |
| `to_file`               | `True`             | Write to a file (`True`) or to stdout (`False`).     |
| `overwrite`             | `False`            | Overwrite an existing log file instead of appending. |

At `logging.DEBUG` level every call of the objective function and its result are logged as well.

Warnings worth checking in the log: ignored or replaced settings, algorithms that were replaced (e.g. `HK` → `KRG`,
`EgoAsync` → `EgoSequential`), and a short `function_delay` list.
