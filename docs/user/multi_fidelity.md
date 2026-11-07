# Multi-fidelity Optimisation

In a multi-fidelity optimisation, cheaper approximations of the objective (lower fidelities, e.g. a coarse mesh or a
simpler model) are combined with the expensive high-fidelity (HF) function. A multi-fidelity surrogate learns from all
of them, and the infill strategy decides per new point which fidelity is worth evaluating, so fewer expensive HF calls
are needed.

## Setting It Up

1. Pass the functions ordered from high to low fidelity: `functions=[f_hf, f_lf1, f_lf2]`. Fidelity level 0 is the HF
    function.
2. Choose a multi-fidelity surrogate and an infill strategy that chooses the fidelity:

| Surrogate                          | Infill strategies                         | Notes                                                                                                                   |
| ---------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `HK` (hierarchical Kriging)        | `TwoStep`, `Random`, `EI`/`GEI` (HF only) | Verified. Each fidelity level needs at least 2 samples.                                                                 |
| `MFK` (SMT multi-fidelity Kriging) | `EI`/`GEI` (HF only), `Random`            | Not verified. Needs nested samples (`nested` is switched on). Does not provide variance per fidelity, so not `TwoStep`. |

`TwoStep` works in two steps: a single-fidelity infill strategy (`single_fid_infill`, default `EI`) chooses the location
on the HF surrogate, and then the cheapest fidelity whose prediction at that location is close enough to the HF
prediction (Jensen-Shannon distance below `jsd_threshold`, default 0.7) is evaluated. `EI` and `GEI` only sample the HF
function in a multi-fidelity run.

With a single function, `HK` is replaced by `KRG` (see [Defining a Problem](configuration.md#algorithms)).

## Initial Samples

With the default settings, the `number_initial_samples` Latin hypercube points are spread over the fidelity levels at
random, and every level is topped up to at least two samples. With `"nested": True`, `NestedLHS` is used instead: every
lower-fidelity level contains the points of the higher levels, which some models (such as `MFK`) need.

## Emulating Expensive Functions

To test how a strategy behaves when the fidelities have different costs, `function_delay` adds an artificial delay per
fidelity, in seconds, ordered from HF to LF:

```python
settings = {"function_delay": [10.0, 1.0]}   # HF takes 10 s, LF 1 s
```

If the list is shorter than the number of fidelities, the last value is used for the remaining levels (with a warning).
The final evaluation of the surrogate optimum is never delayed.

## Example

`examples/03_multi_fidelity_optimisation.py` minimises `x²` with two lower-fidelity versions, using `HK` and `TwoStep`:

```python
--8<-- "examples/03_multi_fidelity_optimisation.py"
```
