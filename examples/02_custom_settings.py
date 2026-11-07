"""
Single fidelity optimisation with custom settings.

The settings passed to the optimiser can be changed by the config instance.
The available settings depend on the used algorithms; all of them are listed in the
Settings Reference of the documentation.

Expected output:
    Optimum at x = [0], f = 0.

Run with:
    python examples/02_custom_settings.py
"""

import numpy as np

from sbo import SBO, Config


# Define the objective function
def optimisation_function(x: np.ndarray) -> np.ndarray:
    """The optimisation function."""
    return x**2


# Set the custom settings
settings = {
    "infill": {"g": 5},
    "optimisation": {"max_function_calls": 50},
    "random_seed": 0,
}

# Generate the config file to define the optimisation problem
config = Config(
    used_algorithms={
        "optimisation": "EgoSequential",
        "infill": "GEI",
        "surrogate": "KRG",
    },
    lb=[-5],
    ub=[5],
    functions=[optimisation_function],
    settings=settings,
)

# Instantiate the sbo
sbo = SBO(config)

# Run the optimisation
# The optimum has the type dict: {'coord': np.ndarray, 'value': float, 'fid': int}
optimum = sbo.run()
