# Settings and Stopping Criteria

Every algorithm has default settings, so `settings` only needs the values you want to change. The
[Settings Reference](../api/settings_reference.md) lists every setting with its description, type, default and allowed
values.

## Passing Settings

General settings go at the top level; settings for the selected algorithms go in a nested dict under `"surrogate"`,
`"infill"` or `"optimisation"`:

```python
settings = {
    "random_seed": 0,                              # general setting
    "number_initial_samples": 10,
    "infill": {"g": 5},                            # setting of the selected infill strategy (GEI)
    "optimisation": {"max_function_calls": 50},    # setting of the selected optimisation algorithm
}
config = Config(..., settings=settings)
```

Only the settings of the selected algorithms are available: with `"infill": "EI"`, the GEI setting `g` is unknown.

### Example

`examples/02_custom_settings.py` runs GEI with `g = 5`, a budget of 50 function calls and a fixed seed:

```python
--8<-- "examples/02_custom_settings.py"
```

### Reading Settings

```python
config.get("random_seed")                   # general setting
config.get("optimisation.max_iterations")   # nested setting, dot notation
config["infill"]["g"]                        # dict access also works
```

## Validation

All settings are checked when `Config` is created:

- an **unknown key** is ignored;
- a value of the **wrong type** is replaced by the default (an `int` is accepted, and converted, where a `float` is
    expected; a JSON list is converted to a tuple where a tuple is expected);
- a value **below the minimum or above the maximum** is replaced by that minimum or maximum;
- a value that is **not one of the allowed values** is replaced by the default.

By default each of these only logs a warning, which the default logger also prints to the screen. To get an error
instead, set `"strict_validation": True`:

```python
config = Config(..., settings={"strict_validation": True, "number_initial_samples": 1})
# ValueError: Value for setting 'number_initial_samples' (1) is below the minimum of 2 ...
```

## Commonly Used Settings

| Setting                           | Default     | Description                                                                                                |
| --------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------- |
| `random_seed`                     | `42`        | Seed for the initial samples, the internal optimisers and all random choices.                              |
| `number_initial_samples`          | `5`         | Number of initial samples if none are passed to `SBO` (at least 2).                                        |
| `number_new_points`               | `2`         | Points proposed per iteration in a multi-point run; see [Multi-point](multi_point.md).                     |
| `nested`                          | `False`     | Nested initial samples for multi-fidelity: every lower-fidelity level also has the higher-fidelity points. |
| `function_delay`                  | `None`      | Artificial delay per fidelity in seconds, e.g. to emulate expensive functions in tests.                    |
| `save_folder`                     | `"results"` | Folder for `save_optimisation_data` and plots; see [Results](results.md).                                  |
| `strict_validation`               | `False`     | Raise a `ValueError` for invalid settings instead of warning.                                              |
| `infill_opt_*`, `surrogate_opt_*` |             | Settings of the differential evolution that optimises the infill function and finds the surrogate optimum. |

## Stopping Criteria

The optimisation stops as soon as **any** criterion is met. Each one is a setting under `"optimisation"`; set it to
`None` to switch it off.

| Setting                          | Default | Stops when                                                                                     |
| -------------------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `max_iterations`                 | `300`   | `max_iterations` iterations are completed. Not used by `EgoAsync`, which has no iterations.    |
| `max_function_calls`             | `1000`  | The total number of function calls (all fidelities) reaches the budget.                        |
| `max_function_calls_HF`          | `150`   | The number of high-fidelity calls reaches the budget.                                          |
| `convergence_threshold`          | `1e-6`  | The surrogate optimum changed by less than this over the last `convergence_window` iterations. |
| `relative_convergence_threshold` | `1e-6`  | The same, relative to the earlier surrogate optimum.                                           |
| `convergence_window`             | `5`     | Number of iterations used by the two convergence criteria.                                     |
| `min_infill_value`               | `1e-6`  | The infill value (e.g. the expected improvement) of the new point drops below this.            |

!!! note "Function-call budgets include the final evaluation"

    After the loop, the optimum of the surrogate is evaluated once more on the high-fidelity function. The function-call
    budgets reserve one HF call for this, so `max_function_calls_HF = 50` gives at most 50 HF calls in total.

The convergence criteria compare the **surrogate optimum** (the minimum of the high-fidelity surrogate), not the best
sample found: the surrogate optimum also changes when new samples change the model.

## Reproducibility

`random_seed` seeds the initial samples, the differential evolution of the infill and surrogate optimisation, the
surrogate models (their own `seed` setting is set to `random_seed`), and `config.rng`, which is used for every other
random choice. Two runs with the same problem, settings and seed therefore give the same result.

The thesis results were produced with the locked environment (`uv sync --locked --no-dev --extra benchmarks`); other
package versions can give slightly different numbers.
