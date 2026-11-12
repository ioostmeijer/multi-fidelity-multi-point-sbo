# Surrogate Model

A surrogate model is fitted to the samples and predicts the objective elsewhere. Add it as `src/sbo/surrogate/<Name>.py`
and follow the [checklist](index.md#checklist).

## Choosing the Base Class

All base classes are in `src/sbo/surrogate/Base.py`; each one extends the previous:

| Base class                             | Use when the model provides                          | Example      |
| -------------------------------------- | ---------------------------------------------------- | ------------ |
| `SurrogateBase`                        | Predictions only                                     | `RBF`        |
| `SurrogateBaseWithVariance`            | Predictions and the variance (of the HF level)       | `KRG`, `MFK` |
| `SurrogateBaseWithVariancePerFidelity` | Predictions and the variance of every fidelity level | `HK`         |

Infill strategies check the base class: EI-like strategies need a variance, and `TwoStep` needs the variance per
fidelity.

## Methods to Implement

| Method                                                            | Base class        | Returns                             | Purpose                                                                        |
| ----------------------------------------------------------------- | ----------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| `_initialise_model(self, *args, **kwargs)`                        | all               | the model object                    | Create the model; called in `__init__`, the result is stored in `self._model`. |
| `train(self)`                                                     | all               | `None`                              | Fit the model to `self.samples`, then set `self._trained = True`.              |
| `predict_values(self, x, *args, **kwargs)`                        | all               | `np.ndarray`, shape `(n_points, 1)` | Predicted values at `x` (shape `(n_points, n_dimensions)`).                    |
| `predict_variance(self, x, *args, **kwargs)`                      | `...WithVariance` | `np.ndarray`, shape `(n_points, 1)` | Predicted variance at `x`.                                                     |
| `predict_variance_fidelity(self, x, *args, fidelity=0, **kwargs)` | `...PerFidelity`  | `np.ndarray`, shape `(n_points, 1)` | Variance at the given fidelity; `predict_variance(x, fidelity=f)` calls it.    |

## What the Base Class Provides

| Member                                                                                       | Meaning                                                                                                                                                    |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `self.samples`                                                                               | The training data: `{fid: [{"coord", "value", "fid"}, ...]}`, a dict with a key per fidelity.                                                              |
| `self._trained`                                                                              | `False` after initialisation and whenever the samples change. Train only when it is `False`, and set it to `True` at the end of `train`.                   |
| `set_training_data(points)`, `add_training_data(points)`                                     | Replace or extend the samples (a list of points, or a dict per fidelity); duplicates are skipped and the input is not changed.                             |
| `predict(x, return_variance=True, **kwargs)`                                                 | Corrects the shape of `x`, trains if needed, and returns the values, or `(values, variances)` if the class has a variance and `return_variance` is `True`. |
| `get_optimum(fidelity=0)`                                                                    | The minimum of the prediction (differential evolution with the `surrogate_opt_*` settings), cached until the samples change.                               |
| `self._correct_x_shape(x)`                                                                   | Returns `x` with shape `(n_points, n_dimensions)`; a 1D array is one point (or, in 1D, a column of points).                                                |
| `save_model()`, `load_model(data)`, `save_model_to_file(path)`, `load_model_from_file(path)` | Save and restore the model and its samples.                                                                                                                |

`save_model_to_file` and `load_model_from_file` use pickle, which can run arbitrary code while loading. In version 2.0.0
they will save only data (samples, optimum and fitted hyperparameters) and rebuild the model on load, so each surrogate
will export and rebuild its own model ([#54](https://github.com/ioostmeijer/multi-fidelity-multi-point-sbo/issues/54)).

Seed every random choice of the model with `self.config.get("random_seed")` (or use `self.config.rng`) so runs are
reproducible.

## Multi-fidelity Surrogates

A surrogate with more than one fidelity level should:

- read the samples of every level from `self.samples` (level 0 is HF) and train on all of them;
- accept `fidelity` in `predict_values` (and `predict_variance`), and return the prediction of that level;
- set the class attribute **`supports_fidelity_prediction = True`**. Kriging Believer then predicts the assumed value of
    a pending lower-fidelity point at its own fidelity; without the flag it uses the HF prediction for every point (see
    [Multi-point Optimisation](../../user/multi_point.md#how-several-points-are-chosen)).

If the model needs at least two samples per level (as `HK` does), raise a `ValueError` in `train` when they are missing.
If the surrogate only makes sense with several fidelities, return the single-fidelity alternative from `__new__` (see
[Conditional Creation](index.md#conditional-creation)), as `HK` does with `KRG`.

## Settings

The model parameters belong in its settings dict, e.g. `src/sbo/DefaultSettings/surrogate/<NAME>.py`. To reuse the
Kriging settings, unpack them: `{**KRG_DEFAULT_SETTINGS, "my_option": {...}}`. Read them in `_initialise_model` with
`self.config.settings["surrogate"]` or `self.config.get("surrogate.<key>")`.
