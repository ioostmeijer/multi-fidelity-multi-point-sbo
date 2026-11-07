# Extending the Framework

New surrogate models, infill strategies and optimisation algorithms are found by name, so once a class is added it can
be selected in `used_algorithms` like the built-in ones. This page covers what every new component needs; the pages per
component type describe the methods to implement:

- [Infill Strategy](infill.md)
- [Surrogate Model](surrogate.md)
- [Optimisation Algorithm](optimisation.md)

Read the [Architecture](../architecture.md) first to see how the components work together.

## Checklist

| Step                                                                  | Where                                                                                              |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1. Add the class, in a file with the **same name as the class**       | `src/sbo/<type>/<Name>.py`                                                                         |
| 2. Derive it from the base class of its type                          | see the page of the component type                                                                 |
| 3. Export it                                                          | `src/sbo/<type>/__init__.py` (import and `__all__`)                                                |
| 4. Add its default settings (an empty dict if it has none)            | `src/sbo/DefaultSettings/<type>/<NAME>.py`                                                         |
| 5. Export the settings                                                | `src/sbo/DefaultSettings/<type>/__init__.py` and `src/sbo/DefaultSettings/__init__.py` (`__all__`) |
| 6. Add unit tests (and a verification test for the status "Verified") | `tests/`, see [Testing](../testing.md)                                                             |
| 7. Add it to the algorithm tables, with its status                    | `docs/index.md` and `README.md`                                                                    |

`<type>` is `surrogate`, `infill` or `optimisation`. `SBO` imports `sbo.<type>.<Name>` and takes the class `<Name>` from
it, which is why the file name must equal the class name.

## Default Settings

Every component has a settings dict named after the class in upper case with the suffix `_DEFAULT_SETTINGS`, e.g.
`MYINFILL_DEFAULT_SETTINGS` for `MyInfill`. Each key maps to a dict that describes the setting:

| Key              | Required | Meaning                                                                                                                                       |
| ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `"type"`         | Yes      | The type, e.g. `int`, `float`, `str`, `list`, or a union such as `float \| None`. An `int` is accepted (and converted) for a `float` setting. |
| `"default"`      | Yes      | The default value.                                                                                                                            |
| `"min"`, `"max"` | No       | Bounds for numerical settings; values outside are replaced by the bound.                                                                      |
| `"allowed"`      | No       | The list of allowed values (for any type).                                                                                                    |

Place a string block directly **above** the dict that describes every key; the
[Settings Reference](../../api/settings_reference.md) takes the descriptions from it. Use this format, one item per key,
with the type first and the defaults in parentheses at the end (those two parts are left out of the reference, which
shows them from the dict):

```python
"""Default settings for the MyInfill infill strategy.

Keys:
    - 'my_setting': float, what the setting does, in one or more sentences. Continuation
        lines are indented further than the dash.
        (default: 0.5, min: 0.0, max: 1.0)
"""
MYINFILL_DEFAULT_SETTINGS = {
    "my_setting": {"type": float, "default": 0.5, "min": 0.0, "max": 1.0},
}
```

To reuse the settings of another component, unpack its dict (`{**KRG_DEFAULT_SETTINGS, "extra": {...}}`) or assign it
(`X_DEFAULT_SETTINGS = Y_DEFAULT_SETTINGS`); the reference then shows only the added or changed keys. A build warning
`No description for setting '...'` means a key is missing from the block.

Inside the component, read the settings from the config (see
[Retrieving Settings](../architecture.md#retrieving-settings)), e.g. `self.config.get("infill.my_setting")`.

## Conditional Creation

A component can decide when it is created that another class fits better, by overriding `__new__` and returning an
instance of that class. `HK` does this for one fidelity:

```python
def __new__(cls, *args, **kwargs) -> "HK | KRG":
    config = kwargs.get("config")            # components are created with config=... as keyword
    if config.num_fidelities == 1:
        from .KRG import KRG

        config.update_used_algorithms({**config.used_algorithms, "surrogate": "KRG"})
        return KRG(*args, **kwargs)
    return super().__new__(cls)
```

Update `config.used_algorithms` when doing this, so the log, the saved data and `SBO` show the class that is actually
used. Raise a `ValueError` if no config is passed.

## Algorithms That Are Not Ready

A component that is not implemented or not functional yet should raise a `NotImplementedError` in `__init__` (as `CEI`,
`CoKRG` and multi-fidelity `FilteredGEI` do), be marked as such in the algorithm tables, and be excluded where
algorithms are listed automatically:

- the allowed single-fidelity infill strategies of `TwoStep` (`ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES` in
    `src/sbo/DefaultSettings/ALLOWED_ENTRIES.py`);
- the combinations in `test_sbo_initialisation` (`tests/tests_unittest/test_sbo.py`).

An infill strategy that supports expensive constraints must be added to `EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL` in
`src/sbo/constants/compatibility_algorithms.py`.
