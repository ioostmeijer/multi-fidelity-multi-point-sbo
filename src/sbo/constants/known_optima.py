"""The known optima of the benchmark problems (deprecated).

The known optimum of every benchmark problem is now its ``f_opt`` attribute, see
``sbo.benchmarks``. ``MF2_KNOWN_OPTIMA`` is kept until version 2.0.0, built from those.
"""

import warnings

from ..benchmarks import ALL_BENCHMARKS


def _deprecated(name: str, stacklevel: int) -> dict:
    """Return the deprecated constant ``name``, with a DeprecationWarning.

    Parameters
    ----------
    name
        The name of the attribute.
    stacklevel
        The stack level of the warning, so it points at the code that used the name.

    Returns
    -------
    dict
        The known optimum (``f_opt``) of every benchmark problem, keyed by its name,
        for the deprecated name MF2_KNOWN_OPTIMA.

    Raises
    ------
    AttributeError
        For any other name.
    """
    if name == "MF2_KNOWN_OPTIMA":
        warnings.warn(
            "MF2_KNOWN_OPTIMA is deprecated and will be removed in version 2.0.0; "
            "use the f_opt attribute of the problems in sbo.benchmarks instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        return {benchmark.name: benchmark.f_opt for benchmark in ALL_BENCHMARKS}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> dict:
    """Return a deprecated constant (module ``__getattr__``, PEP 562)."""
    return _deprecated(name, stacklevel=2)
