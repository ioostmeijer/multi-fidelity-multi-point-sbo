"""
Basic multi-point optimisation.

The minimal setup to perform a multi-point optimisation.
Note that for running multi-point optimisations MPI should be installed. And at least
two ranks should be available. Otherwise the optimisation will be run using a
single-point optimisation (EgoSequential)

Expected output:
    Optimum at x = [0], f = 0.

Run with:
    mpiexec -n <N_CORES> python examples/04_multi_point_optimisation.py
With:
    - The mpiexec call can depend per used MPI.
    - <N_CORES> should be replaced with the number of cores to be used.
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
        "optimisation": "EgoAsync",
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
# Note: only rank 0 returns the optimum, all other ranks return an empty dict.
# The sbo.run() should be called on every rank.
optimum = sbo.run()
