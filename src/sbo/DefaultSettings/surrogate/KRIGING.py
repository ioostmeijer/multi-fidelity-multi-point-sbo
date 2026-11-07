import numpy as np
from smt.surrogate_models import KRG as _SMT_KRG

from .SMT import SMT_SURROGATE_DEFAULT_SETTINGS

# SMT's default nugget (100 * machine epsilon), read from SMT so it is defined once
SMT_DEFAULT_NUGGET = float(_SMT_KRG().options["nugget"])

"""Default settings for the simple Kriging surrogate model.

Keys (alphabetically ordered, spaces replaced with underscores):
    - corr: str, correlation function type
        (default: "squar_exp",
        allowed: ["pow_exp", "abs_exp", "squar_exp", "matern52", "matern32"])
    - eval_noise: bool, whether to evaluate noise (default: False)
    - hyper_opt: str, hyperparameter optimisation method
                (default: "Cobyla", allowed: ["Cobyla", "TNC"])
    - is_ri: bool, activate reinterpolation for noisy cases
        (default: False)
    - noise_bounds: list[float], bounds for noise hyperparameters
        (default: [np.finfo(float).eps, 10000000000.0])
    - noise0: list[float], initial noise hyperparameters
        (default: [0.0])
    - nugget: float, a jitter for numerical stability
        (default: SMT's default, 100 * np.finfo(float).eps, min: 0.0)
    - poly: str, regression function type
        (default: "constant", allowed: ["constant", "linear", "quadratic"])
    - pow_exp_power: float, Power for the pow_exp kernel function. This option is set
        automatically when corr option is squar, abs, or matern.
        (default: 1.9, min: 1e-10, max: 2.0)
    - theta_bounds: list[float], bounds for hyperparameters
        (default: [1e-06, 20.0])
    - theta0: list[float], initial hyperparameters
        (default: [0.01])
    - use_het_noise: bool, heteroscedastic noise evaluation flag
        (default: False)
    - seed: int | None, random seed for reproducibility
        (default: None)
    - All keys from SMT_SURROGATE_DEFAULT_SETTINGS.
"""
KRG_DEFAULT_SETTINGS = {
    **SMT_SURROGATE_DEFAULT_SETTINGS,
    "corr": {
        "type": str,
        "default": "squar_exp",
        "allowed": ["pow_exp", "abs_exp", "squar_exp", "matern52", "matern32"],
    },
    "eval_noise": {"type": bool, "default": False},
    "hyper_opt": {"type": str, "default": "Cobyla", "allowed": ["Cobyla", "TNC"]},
    "is_ri": {"type": bool, "default": False},
    "noise_bounds": {
        "type": list,
        "default": [np.finfo(float).eps, 1e10],
    },
    "noise0": {"type": list, "default": [0.0]},
    "nugget": {"type": float, "default": SMT_DEFAULT_NUGGET, "min": 0.0},
    "poly": {
        "type": str,
        "default": "constant",
        "allowed": ["constant", "linear", "quadratic"],
    },
    "pow_exp_power": {"type": float, "default": 1.9, "min": 1e-10, "max": 2.0},
    "theta_bounds": {"type": list, "default": [1e-06, 20.0]},
    "theta0": {"type": list, "default": [0.01]},
    "use_het_noise": {"type": bool, "default": False},
    "seed": {"type": int | type(None), "default": None},
}

"""Default settings for the SMT MFK (Multi-Fidelity Kriging) surrogate model.

Keys (alphabetically ordered, spaces replaced with underscores):
    - optim_var: bool, whether the variance at HF samples is forced to zero
        (default: True)
    - propagate_uncertainty: bool, if True, the variance contribution of LF levels are
        considered
        (default: False)
    - rho_regr: str, regression type for rho
        (default: "constant", allowed: ["constant", "linear", "quadratic"])
    - All keys from KRG_DEFAULT_SETTINGS.
"""
MFK_DEFAULT_SETTINGS = {
    **KRG_DEFAULT_SETTINGS,
    "optim_var": {"type": bool, "default": True},
    "propagate_uncertainty": {"type": bool, "default": False},
    "rho_regr": {
        "type": str,
        "default": "constant",
        "allowed": ["constant", "linear", "quadratic"],
    },
}
