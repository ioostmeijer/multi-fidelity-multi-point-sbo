"""
Basic multi-fidelity optimisation.

The minimal setup to perform a multi-fidelity optimisation.

Expected output:
    Optimum at x = [0], f = 0.

Run with:
    python examples/03_multi_fidelity_optimisation.py
"""

import numpy as np

from sbo import SBO, Config


# Define the objective functions
def optimisation_function_hf(x: np.ndarray) -> np.ndarray:
    """The optimisation function."""
    return x**2


def optimisation_function_lf1(x: np.ndarray) -> np.ndarray:
    """The first low-fidelity function."""
    return x**2 + 1


def optimisation_function_lf2(x: np.ndarray) -> np.ndarray:
    """The second low-fidelity function."""
    return x**2 + 2


# Set the list of functions. The functions should be ordered from HF -> LF
functions = [
    optimisation_function_hf,
    optimisation_function_lf1,
    optimisation_function_lf2,
]

# Generate the config file to define the optimisation problem
config = Config(
    used_algorithms={
        "optimisation": "EgoSequential",
        "infill": "TwoStep",
        "surrogate": "HK",
    },
    lb=[-5],
    ub=[5],
    functions=functions,
)

# Instantiate the sbo
sbo = SBO(config)

# Run the optimisation
# The optimum has the type dict: {'coord': np.ndarray, 'value': float, 'fid': int}
optimum = sbo.run()
