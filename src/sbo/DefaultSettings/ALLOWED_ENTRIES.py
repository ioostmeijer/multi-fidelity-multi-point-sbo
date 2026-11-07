import os
import warnings
from pathlib import Path

ALLOWED_DIFF_EVOLUTION_STRATEGIES = [
    "best1bin",
    "best1exp",
    "rand1bin",
    "rand1exp",
    "rand2bin",
    "rand2exp",
    "randtobest1bin",
    "randtobest1exp",
    "currenttobest1bin",
    "currenttobest1exp",
    "best2exp",
    "best2bin",
]

ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES = []
try:
    current_dir = Path(__file__).resolve().parent
    infill_dir = current_dir.joinpath("..", "..", "sbo", "infill").resolve()
    for file in os.listdir(infill_dir):
        if file.endswith(".py") and file not in [
            "__init__.py",
            "Base.py",
            "TwoStep.py",
            # Not implemented or not functional yet (raise NotImplementedError on
            # creation)
            "CEI.py",
            "FilteredGEI.py",
        ]:
            strategy_name = file[:-3]
            ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES.append(strategy_name)
except Exception as e:
    warnings.warn(
        f"Error while loading allowed Two-Step single-fidelity infill strategies: {e}",
        stacklevel=2,
    )
    ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES = ["EI"]
