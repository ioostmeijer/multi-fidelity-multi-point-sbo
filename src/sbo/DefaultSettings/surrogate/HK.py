from .KRIGING import KRG_DEFAULT_SETTINGS

"""Default settings for the hierarchical Kriging surrogate model.

Keys (alphabetically ordered, spaces replaced with underscores):
    - all keys from KRG_DEFAULT_SETTINGS
    - corr: str, correlation function type
        (default: "squar_exp",
        allowed: ["pow_exp", "cubic_spline", "squar_exp"])
    - n_theta_optim_starts: int, number of differential evolution runs (each with a
        different seed) used to optimise the hyperparameters theta; the best is kept
        (default: 3, must be >= 1)
    - pow_exp_power: float, Power for the pow_exp kernel function.
        (default: 1.9, min: 1e-10, max: 2.0)
    - theta_bounds: list[float], bounds for hyperparameters
        (default: [1e-06, 20.0])
"""
HK_DEFAULT_SETTINGS = {
    **KRG_DEFAULT_SETTINGS,
    "corr": {
        "type": str,
        "default": "squar_exp",
        "allowed": ["pow_exp", "cubic_spline", "squar_exp"],
    },
    "pow_exp_power": {"default": 1.9, "min": 1e-10, "max": 2.0, "type": float},
    "theta_bounds": {"default": [1e-06, 20.0], "type": list},
    "n_theta_optim_starts": {"default": 3, "type": int, "min": 1},
}
