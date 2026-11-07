"""Import all infill settings."""

from .FILTEREDGEI import FILTEREDGEI_DEFAULT_SETTINGS
from .GEI import GEI_DEFAULT_SETTINGS
from .TWOSTEP import TWOSTEP_DEFAULT_SETTINGS

# Create an empty default settings dictionary for infill methods which do not have
# specific settings. This allows for a consistent interface when validating settings
# across different infill strategies.
EI_DEFAULT_SETTINGS: dict = {}
RANDOM_DEFAULT_SETTINGS: dict = {}
CEI_DEFAULT_SETTINGS: dict = {}

__all__ = [
    "CEI_DEFAULT_SETTINGS",
    "EI_DEFAULT_SETTINGS",
    "FILTEREDGEI_DEFAULT_SETTINGS",
    "GEI_DEFAULT_SETTINGS",
    "RANDOM_DEFAULT_SETTINGS",
    "TWOSTEP_DEFAULT_SETTINGS",
]
