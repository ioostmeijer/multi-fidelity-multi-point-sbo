# Infill Strategy

An infill strategy decides where (and at which fidelity) to sample next. Add it as `src/sbo/infill/<Name>.py`, derived
from `InfillBase` (`src/sbo/infill/Base.py`), and follow the [checklist](index.md#checklist).

## Methods to Implement

| Method                                                   | Returns                                  | Purpose                                                                                                      |
| -------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `get_new_points(self, surrogate, *args, **kwargs)`       | `tuple[list[list[dict]], float]`         | The new tasks and the infill value; called by the optimisation algorithm via `infill(surrogate)`.            |
| `calc_infill_value(self, x, surrogate, *args, **kwargs)` | `float` (or an array for several points) | The infill value at `x`; used for plotting. Raise `NotImplementedError` if there is none (as `Random` does). |

`get_new_points` returns a list of **tasks**, each a list of points `{"coord": np.ndarray, "fid": int}` that one worker
evaluates together, and the **infill value** of the selection. The optimisation algorithm uses the infill value for the
`min_infill_value` stopping criterion, so return `float("inf")` if your strategy has no meaningful value.

## Helpers from InfillBase

| Helper                                                                 | Use                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `self._select_sample_point_optimisation(optim_func, constraints=None)` | Minimises `optim_func(x)` over the bounds with differential evolution, configured by the `infill_opt_*` settings. Returns `(x, optim_func(x))`. If DE does not report convergence, its best point is still used when valid; only an invalid result falls back to the best of 1000 points (a grid in 1D, a Latin hypercube in more dimensions). To maximise an infill value, pass its negative. |
| `self._differential_evolution_settings()`                              | The keyword arguments for `scipy.optimize.differential_evolution` from the `infill_opt_*` settings and `random_seed`, if you need your own DE call.                                                                                                                                                                                                                                            |
| `self._fallback_points(n_points=1000)`                                 | The fallback points described above.                                                                                                                                                                                                                                                                                                                                                           |
| `self._return_nested_samples(task)`                                    | Adds the lower-fidelity points at the same location if `nested` sampling is on; always pass the task through it.                                                                                                                                                                                                                                                                               |
| `self.config`                                                          | The [config](../architecture.md#config): settings, bounds, `num_fidelities`, `rng`.                                                                                                                                                                                                                                                                                                            |

## Surrogate Requirements

Check the type of the surrogate at the start of `get_new_points` and raise a `TypeError` if it cannot be used:

- `SurrogateBase`: predictions only;
- `SurrogateBaseWithVariance`: predictions and variance (needed for EI-like criteria);
- `SurrogateBaseWithVariancePerFidelity`: variance per fidelity (needed to choose a fidelity, as `TwoStep` does).

`surrogate.get_optimum(fidelity=0)` returns the optimum of the surrogate, `surrogate.samples` the samples per fidelity.

## Example

A minimal strategy that samples where the surrogate prediction is lowest:

```python
import numpy as np

from ..surrogate import SurrogateBase
from .Base import InfillBase


class LowestPrediction(InfillBase):
    """Sample at the minimum of the surrogate prediction."""

    def get_new_points(
        self, surrogate: SurrogateBase, *args, **kwargs
    ) -> tuple[list[list[dict]], float]:
        """Return the minimum of the surrogate as the new point."""
        coord, _ = self._select_sample_point_optimisation(
            lambda x: self.calc_infill_value(x, surrogate)
        )
        task = [{"coord": coord, "fid": 0}]
        return [self._return_nested_samples(task)], float("inf")

    def calc_infill_value(
        self, x: np.ndarray, surrogate: SurrogateBase, *args, **kwargs
    ) -> float:
        """Return the surrogate prediction at x."""
        prediction = surrogate.predict(np.atleast_2d(x), return_variance=False)
        return float(prediction[0][0])
```

With `LOWESTPREDICTION_DEFAULT_SETTINGS: dict = {}` in `src/sbo/DefaultSettings/infill/__init__.py`, it can be used with
`"infill": "LowestPrediction"`.

## Multi-point Runs

An infill strategy proposes points for the current surrogate. In multi-point runs, the optimisation algorithm calls it
again after adding the pending points with Kriging Believer or Constant Liar values (see
[Multi-point Optimisation](../../user/multi_point.md#how-several-points-are-chosen)), so the strategy itself does not
need to handle pending points. A strategy may return several tasks at once; `number_new_points` limits how many are used
per iteration.
