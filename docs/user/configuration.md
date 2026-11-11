# Defining a Problem

An optimisation problem is described by one `Config` object, which is passed to `SBO`:

```python
from sbo import SBO, Config

config = Config(
    used_algorithms={"optimisation": "EgoSequential", "infill": "EI", "surrogate": "KRG"},
    lb=[-5],
    ub=[5],
    functions=[lambda x: float(x[0] ** 2)],
    settings={"random_seed": 0},
)
optimum = SBO(config).run()
```

## Arguments

| Argument                  | Required | Description                                                                                 |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------- |
| `used_algorithms`         | Yes      | The surrogate, infill strategy and optimisation algorithm to use, by name.                  |
| `lb`, `ub`                | Yes      | Lower and upper bounds of the design space, one value per dimension.                        |
| `functions`               | Yes      | The objective function(s), ordered from high fidelity (HF) to low fidelity (LF).            |
| `settings`                | No       | Settings that override the defaults; see [Settings](settings.md).                           |
| `inexpensive_constraints` | No       | Constraints that are cheap to evaluate (under development).                                 |
| `expensive_constraints`   | No       | Constraints that are modelled with a surrogate (under development).                         |
| `logger`                  | No       | A `logging.Logger` to use instead of the default logger; see [Logging](results.md#logging). |
| `name`                    | No       | A name for the problem.                                                                     |

## Objective Functions

`functions` is a list of callables. The optimiser **minimises** the first function. Each function:

- takes the coordinates as a NumPy array of shape `(n_dimensions,)`;
- returns the objective value, either as a plain number or as an array-like whose first element is the value.

```python
def objective(x: np.ndarray) -> float:
    return float((x[0] - 1.0) ** 2 + (x[1] + 0.5) ** 2)


def objective_array(x: np.ndarray) -> np.ndarray:
    return np.atleast_1d(np.sum(x**2))      # also fine: the first element is used
```

`Config` wraps every function, so the rest of the framework always receives a 1D float array; `config.functions` returns
these wrapped functions. To maximise a function, minimise its negative.

For a multi-fidelity problem, add the cheaper approximations after the high-fidelity function, ordered from high to low
fidelity. The number of functions sets the number of fidelity levels:

```python
functions = [objective_hf, objective_lf1, objective_lf2]   # HF -> LF
```

See [Multi-fidelity Optimisation](multi_fidelity.md) for the algorithms that use them.

## Bounds

`lb` and `ub` are lists, tuples or NumPy arrays with one entry per design variable, so their length sets the dimension
of the problem. Every upper bound must be larger than the lower bound:

```python
lb = [-5, 0]    # x1 in [-5, 5], x2 in [0, 10]
ub = [5, 10]
```

To change both bounds later, use `config.set_bounds(lb, ub)`. The `config.lb` and `config.ub` setters check the new
bound against the current other bound, so moving the whole domain (e.g. from `[0, 1]` to `[2, 3]`) needs `set_bounds`.

## Algorithms

`used_algorithms` is a dict with the keys `"surrogate"`, `"infill"` and `"optimisation"`; any other key raises a
`ValueError`. The values are the class names listed with their status on the
[home page](../index.md#included-algorithms):

```python
used_algorithms = {"surrogate": "HK", "infill": "TwoStep", "optimisation": "EgoAsync"}
```

Not every combination makes sense:

- A multi-fidelity run needs a multi-fidelity surrogate (`HK` or `MFK`) and an infill strategy that chooses the fidelity
    (`TwoStep`); see [Multi-fidelity Optimisation](multi_fidelity.md).
- `EgoMPI` and `EgoAsync` need MPI and at least two ranks; see [Multi-point Optimisation](multi_point.md).

Some algorithms replace themselves when they don't apply, and log this:

| Selected             | Condition                       | Used instead    |
| -------------------- | ------------------------------- | --------------- |
| `HK`                 | One function (single fidelity)  | `KRG`           |
| `FilteredGEI`        | One function (single fidelity)  | `GEI`           |
| `EgoMPI`, `EgoAsync` | No MPI, or fewer than two ranks | `EgoSequential` |

## Initial Samples

By default, `SBO` evaluates `number_initial_samples` initial samples (default 5), placed with Latin hypercube sampling
(`NestedLHS` for nested multi-fidelity sampling). To start from samples you already have, pass them to `SBO`:

```python
initial_samples = [
    {"coord": np.array([1.0]), "value": 1.0, "fid": 0},
    {"coord": np.array([-2.0]), "value": 4.0, "fid": 0},
]
optimum = SBO(config, initial_samples=initial_samples).run()
```

Each sample is a dict with its coordinates (`coord`), objective value (`value`) and fidelity level (`fid`, 0 is the
highest fidelity). If a fidelity level has fewer than two samples, random extra samples are added (seeded with
`random_seed`). The list you pass is not changed.

## Constraints

!!! warning "Under development"

    Constraint handling is not finished yet. No infill strategy currently supports constraints, and `CEI`, the constrained
    infill strategy, is planned for 1.5.0. Passing `expensive_constraints` raises a `ValueError`. `inexpensive_constraints`
    are accepted but ignored during the optimisation. Empty dicts or lists count as no constraints.

Constraints are passed as dicts with the keys `"inequality"` and `"equality"`. Inequality constraints are satisfied when
`g(x) <= 0`, equality constraints when `h(x) = 0`:

```python
inexpensive_constraints = {"inequality": [g1, g2], "equality": [h1]}
```

For `expensive_constraints`, each constraint is itself a list of callables ordered from HF to LF, from which a surrogate
model is built during the optimisation.

## Loading from a JSON File

A problem can also be defined in a JSON file and loaded with `Config.from_json`, which accepts a path or an already
loaded dict. The functions are given by module and attribute name, and the bounds are taken from the function set. This
is mainly used with the benchmark problems in `sbo.benchmarks`, which are part of the package: Forrester, Bohachevsky,
Booth, Branin, Currin, Himmelblau, six-hump camelback, Park91a, Park91b, Hartmann6 and borehole.

```json
--8<-- "examples/05_run_from_config/05_run_from_config.json"
```

```python
--8<-- "examples/05_run_from_config/05_run_from_config.py"
```

| Key                                       | Meaning                                                                                                                                                                                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `used_algorithms`                         | As in `Config`.                                                                                                                                                                                                                         |
| `functions.module`, `functions.attribute` | `sbo.benchmarks` and the name of a benchmark problem (case and spaces don't matter, e.g. `Forrester`), or another module and a class or object in it. The class or object must provide `functions` (HF to LF), `l_bound` and `u_bound`. |
| `functions.n_fidelities`                  | Optional: use only the first `n` fidelity levels. All available fidelities are used, if a larger number of fidelities is specified and a warning is logged.                                                                             |
| `settings`                                | As in `Config`.                                                                                                                                                                                                                         |
| `name`                                    | Optional problem name.                                                                                                                                                                                                                  |
| `logger_name`, `logger`                   | Optional: the name of the logger to create, and a dict of `setup_logger` arguments (e.g. `{"to_file": false}`).                                                                                                                         |

`from_json` imports only the module named in `functions.module`. If the module can't be imported, `from_json` raises a
`ModuleNotFoundError`. An unknown benchmark name gives a `ValueError` that lists the available problems.

!!! note "Deprecated: the mf2 module"

    JSON files written for version 1.0.0 use `"module": "mf2"`. They still work, also without mf2 installed: `from_json`
    uses the problem of the same name from `sbo.benchmarks` and gives a `DeprecationWarning`. Change the module to
    `"sbo.benchmarks"`; `"mf2"` will be removed in version 2.0.0.

## Config Is a Singleton

!!! note

    There is only one `Config` instance per process. Creating a second `Config` re-initialises that instance with the new
    arguments and logs a warning. Every object that holds the config, such as an existing `SBO`, then sees the new problem.

To make the switch explicit, destroy the instance first:

```python
Config.destroy_instance()
config = Config(...)   # new problem definition
```

`Config.get_instance()` returns the current instance from anywhere in the code.
