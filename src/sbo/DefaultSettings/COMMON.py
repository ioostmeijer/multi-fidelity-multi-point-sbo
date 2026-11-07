import pathlib

from .ALLOWED_ENTRIES import ALLOWED_DIFF_EVOLUTION_STRATEGIES

"""The default settings used in the optimisation, infill and surrogate classes.

Including the settings used in the scipy differential evolution algorithm, used to find
the optimum point on the current surrogate model (used in every iteration of the SBO to
calculate the stopping criteria).

Keys:
    - 'function_delay': list or None, optional delays for the functions in seconds.
        Ordered from HF to LF. If None, no delays are used. If the list is shorter
        than the number of fidelities, the last value is used for the remaining
        fidelities (with a warning). (default: None)
    - 'infill': dict, optional settings for the infill strategy
        (default: {})
    - 'infill_opt_init': str, initialisation method of the differential evolution
        used to optimise the infill function
        (default: "latinhypercube", allowed:
            ['latinhypercube', 'random', 'sobol', 'halton'])
    - 'infill_opt_max_iterations': int, maximum number of generations of the
        infill optimisation (default: 1000, must be >= 1)
    - 'infill_opt_mutation': float or tuple[float], mutation factor of the infill
        optimisation (default: (0.5, 1.0))
    - 'infill_opt_polish': bool, whether to polish the result of the infill
        optimisation (default: True)
    - 'infill_opt_population_size': int, population size multiplier of the infill
        optimisation (default: 15, must be >= 1)
    - 'infill_opt_recombination': float, recombination factor of the infill
        optimisation (default: 0.7, min: 0.0, max: 1.0)
    - 'infill_opt_strategy': str, the differential evolution strategy of the infill
        optimisation (default: "best1bin", allowed: ALLOWED_DIFF_EVOLUTION_STRATEGIES)
    - 'infill_opt_tolerance': float, relative tolerance for convergence of the infill
        optimisation (default: 1e-6, min: 0.0)
    - 'nested': bool, whether to use nested sampling
        (default: False)
    - 'number_initial_samples': int, number of initial samples to generate if none are
        provided (default: 5, must be >= 2)
    - 'number_new_points': int, number of new points to generate in each iteration
        (default: 2, must be >= 1)
    - 'optimisation': dict, optional settings for the optimisation algorithm
        (default: {})
    - 'random_seed': int, to choose the random seed for reproducibility
        (default: 42, must be >= 0)
    - 'save_folder': str | pathlib.Path | None, directory in which
        save_optimisation_data saves the results (default: "results")
    - 'strict_validation': bool, if True an unknown setting or an invalid value
        raises a ValueError; if False it only logs a warning and the key is ignored or
        the value replaced by the default, minimum or maximum (default: False)
    - 'surrogate': dict, optional settings for the surrogate model
        (default: {})
    - 'surrogate_opt_atol': float, absolute tolerance for convergence
        (default: 0.01, min: 0.0)
    - 'surrogate_opt_init': str, initialisation method for the optimisation
        (default: "latinhypercube", allowed:
            ['latinhypercube', 'random', 'sobol', 'halton'])
    - 'surrogate_opt_max_iterations': int, maximum number of iterations
        (default: 1000, must be >= 1)
    - 'surrogate_opt_mutation': float or tuple[float], mutation factor
        (default: (0.5, 1.0))
    - 'surrogate_opt_polish': bool, whether to polish the result
        (default: True)
    - 'surrogate_opt_population_size': int, size of the population
        (default: 15, must be >= 1)
    - 'surrogate_opt_recombination': float, recombination factor
        (default: 0.7, min: 0.0, max: 1.0)
    - 'surrogate_opt_strategy': str, the strategy to use for the optimisation
        (default: "best1bin", allowed: ALLOWED_DIFF_EVOLUTION_STRATEGIES)
    - 'surrogate_opt_tolerance': float, tolerance for convergence
        (default: 0.01, min: 0.0)
    - 'surrogate_opt_verbose': bool, whether to print verbose output
        (default: False)
"""
DEFAULT_SETTINGS = {
    "function_delay": {"type": list | type(None), "default": None},
    "infill": {"type": dict, "default": {}},
    "infill_opt_init": {
        "type": str,
        "default": "latinhypercube",
        "allowed": ["latinhypercube", "random", "sobol", "halton"],
    },
    "infill_opt_max_iterations": {"type": int, "default": 1000, "min": 1},
    "infill_opt_mutation": {"type": float | tuple, "default": (0.5, 1.0)},
    "infill_opt_polish": {"type": bool, "default": True},
    "infill_opt_population_size": {"type": int, "default": 15, "min": 1},
    "infill_opt_recombination": {
        "type": float,
        "default": 0.7,
        "min": 0.0,
        "max": 1.0,
    },
    "infill_opt_strategy": {
        "type": str,
        "default": "best1bin",
        "allowed": ALLOWED_DIFF_EVOLUTION_STRATEGIES,
    },
    "infill_opt_tolerance": {"type": float, "default": 1e-6, "min": 0.0},
    "nested": {"type": bool, "default": False},
    "number_new_points": {
        "type": int,
        "default": 2,
        "min": 1,
    },
    "number_initial_samples": {"type": int, "default": 5, "min": 2},
    "optimisation": {"type": dict, "default": {}},
    "random_seed": {"type": int, "default": 42, "min": 0},
    "save_folder": {
        "type": str | type(None) | pathlib.Path,
        "default": "results",
    },
    "strict_validation": {"type": bool, "default": False},
    "surrogate": {"type": dict, "default": {}},
    "surrogate_opt_atol": {"type": float, "default": 0.01, "min": 0.0},
    "surrogate_opt_init": {
        "type": str,
        "default": "latinhypercube",
        "allowed": ["latinhypercube", "random", "sobol", "halton"],
    },
    "surrogate_opt_max_iterations": {"type": int, "default": 1000, "min": 1},
    "surrogate_opt_mutation": {"type": float | tuple, "default": (0.5, 1.0)},
    "surrogate_opt_polish": {"type": bool, "default": True},
    "surrogate_opt_population_size": {"type": int, "default": 15, "min": 1},
    "surrogate_opt_recombination": {
        "type": float,
        "default": 0.7,
        "min": 0.0,
        "max": 1.0,
    },
    "surrogate_opt_strategy": {
        "type": str,
        "default": "best1bin",
        "allowed": ALLOWED_DIFF_EVOLUTION_STRATEGIES,
    },
    "surrogate_opt_tolerance": {"type": float, "default": 0.01, "min": 0.0},
    "surrogate_opt_verbose": {"type": bool, "default": False},
}
