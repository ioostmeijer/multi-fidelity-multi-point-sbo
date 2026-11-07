"""
Basic single-fidelity optimisation.

The minimal setup to perform a single-fidelity optimisation.

Expected output:
    Optimum at x = [0], f = 0.

Run with:
    python examples/01_basic_optimisation.py
"""

import numpy as np

from sbo import SBO, Config


# Define the objective function
def optimisation_function(x: np.ndarray) -> np.ndarray:
    """The optimisation function."""
    return x**2


# Generate the config file to define the optimisation problem
config = Config(
    used_algorithms={
        "optimisation": "EgoSequential",
        "infill": "EI",
        "surrogate": "KRG",
    },
    lb=[-5],
    ub=[5],
    functions=[optimisation_function],
)

# Instantiate the sbo
sbo = SBO(config)

# Run the optimisation
# The optimum has the type dict: {'coord': np.ndarray, 'value': float, 'fid': int}
optimum = sbo.run()
