"""The file containing the constants used."""

import warnings

# The function delays (HF, LF) in seconds used for the benchmark problems in
# sbo.benchmarks, keyed by their name. They emulate expensive functions, as in the
# thesis experiments.
BENCHMARK_FUNCTION_DELAY = {
    "forrester": [12 * 10, 1.2 * 10],
    "bohachevsky": [12 * 17, 1.2 * 17],
    "booth": [12 * 16, 1.2 * 16],
    "branin": [12 * 19, 1.2 * 19],
    "currin": [12 * 24, 1.2 * 24],
    "himmelblau": [12 * 21, 1.2 * 21],
    "six_hump_camelback": [12 * 37, 1.2 * 37],
    "park91a": [12 * 50, 1.2 * 50],
    "park91b": [12 * 126, 1.2 * 126],
    "hartmann6": [12 * 190, 1.2 * 190],
    "borehole": [12 * 550, 1.2 * 550],
}

# Combine all function delays
ALL_FUNCTION_DELAYS = {**BENCHMARK_FUNCTION_DELAY}


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
        BENCHMARK_FUNCTION_DELAY, for the deprecated name MF2_FUNCTION_DELAY.

    Raises
    ------
    AttributeError
        For any other name.
    """
    if name == "MF2_FUNCTION_DELAY":
        warnings.warn(
            "MF2_FUNCTION_DELAY is deprecated and will be removed in version 2.0.0; "
            "use BENCHMARK_FUNCTION_DELAY instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        return BENCHMARK_FUNCTION_DELAY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> dict:
    """Return a deprecated constant (module ``__getattr__``, PEP 562)."""
    return _deprecated(name, stacklevel=2)
