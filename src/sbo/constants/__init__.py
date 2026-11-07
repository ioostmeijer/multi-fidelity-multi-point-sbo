"""Enable the import of the constants."""

from .compatibility_algorithms import EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL
from .function_delays import ALL_FUNCTION_DELAYS, MF2_FUNCTION_DELAY
from .known_optima import MF2_KNOWN_OPTIMA

__all__ = [
    "ALL_FUNCTION_DELAYS",
    "EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL",
    "MF2_FUNCTION_DELAY",
    "MF2_KNOWN_OPTIMA",
]
