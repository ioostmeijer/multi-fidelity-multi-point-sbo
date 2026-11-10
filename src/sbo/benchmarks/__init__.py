"""Multi-fidelity benchmark problems, available via sbo.benchmarks.

Every problem is a :class:`MultiFidelityFunction` with one function per fidelity (high
to low), the bounds and the known optimum. Get one by its constant (``FORRESTER``) or
by name (``get_benchmark("forrester")``).
"""

# The problem modules only import from .base, never from this package, so there are
# no circular imports.
from .base import FidelityFunction, FunctionSet, MultiFidelityFunction, as_2d
from .bohachevsky import BOHACHEVSKY
from .booth import BOOTH
from .borehole import BOREHOLE
from .branin import BRANIN
from .currin import CURRIN
from .forrester import FORRESTER
from .hartmann6 import HARTMANN6
from .himmelblau import HIMMELBLAU
from .park91a import PARK91A
from .park91b import PARK91B
from .six_hump_camelback import SIX_HUMP_CAMELBACK

ALL_BENCHMARKS: tuple[MultiFidelityFunction, ...] = (
    FORRESTER,
    BOHACHEVSKY,
    BOOTH,
    BRANIN,
    CURRIN,
    HIMMELBLAU,
    SIX_HUMP_CAMELBACK,
    PARK91A,
    PARK91B,
    HARTMANN6,
    BOREHOLE,
)
"""All benchmark problems, in a fixed order (the order of the thesis and of the
verification tests)."""

_BENCHMARKS_BY_NAME = {benchmark.name: benchmark for benchmark in ALL_BENCHMARKS}


def get_benchmark(name: str) -> MultiFidelityFunction:
    """Return the benchmark problem with the given name.

    Parameters
    ----------
    name
        The name of the problem. Case and spaces don't matter: ``"Six Hump Camelback"``
        finds ``"six_hump_camelback"``.

    Returns
    -------
    MultiFidelityFunction
        The benchmark problem.

    Raises
    ------
    ValueError
        If there is no benchmark problem with this name.
    """
    key = name.strip().lower().replace(" ", "_")
    try:
        return _BENCHMARKS_BY_NAME[key]
    except KeyError:
        available = ", ".join(_BENCHMARKS_BY_NAME)
        raise ValueError(
            f"Unknown benchmark '{name}'. Available benchmarks: {available}."
        ) from None


__all__ = [
    "ALL_BENCHMARKS",
    "BOHACHEVSKY",
    "BOOTH",
    "BOREHOLE",
    "BRANIN",
    "CURRIN",
    "FORRESTER",
    "HARTMANN6",
    "HIMMELBLAU",
    "PARK91A",
    "PARK91B",
    "SIX_HUMP_CAMELBACK",
    "FidelityFunction",
    "FunctionSet",
    "MultiFidelityFunction",
    "as_2d",
    "get_benchmark",
]
