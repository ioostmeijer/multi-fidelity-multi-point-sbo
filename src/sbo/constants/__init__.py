"""Enable the import of the constants."""

from . import function_delays, known_optima
from .compatibility_algorithms import EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL
from .function_delays import ALL_FUNCTION_DELAYS, BENCHMARK_FUNCTION_DELAY

# Deprecated names (removed in version 2.0.0), with the module that defines them. That
# module gives the DeprecationWarning.
_DEPRECATED = {
    "MF2_FUNCTION_DELAY": function_delays,
    "MF2_KNOWN_OPTIMA": known_optima,
}


def __getattr__(name: str) -> dict:
    """Return a deprecated constant, with a DeprecationWarning.

    Parameters
    ----------
    name
        The name of the attribute.

    Returns
    -------
    dict
        The value of the deprecated constant.

    Raises
    ------
    AttributeError
        If there is no deprecated constant with this name.
    """
    if name in _DEPRECATED:
        return _DEPRECATED[name]._deprecated(name, stacklevel=2)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALL_FUNCTION_DELAYS",
    "BENCHMARK_FUNCTION_DELAY",
    "EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL",
]
