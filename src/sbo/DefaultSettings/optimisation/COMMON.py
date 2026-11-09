r"""Default settings used in all optimisation algorithms.

The convergence criteria compare the optimum of the surrogate model (the minimum of the
HF surrogate prediction, 'surrogate_optimum' in the optimisation data), not the best
observed sample value. This is intended: the surrogate optimum also changes when new
samples change the model, while the best observed value only changes when a better
sample is found.

Keys:
    - 'convergence_threshold': float | None, threshold $\varepsilon$ for convergence if
        None this will not be considered (default: 1e-6, must be >= 0.0). Converged
        when $\lvert \hat{y}^*_{t-w} - \hat{y}^*_t \rvert < \varepsilon$, with
        $\hat{y}^*_t$ the surrogate optimum in iteration $t$ and $w$ the
        'convergence_window'
    - 'convergence_window': int | None, number of iterations to consider for
        convergence. Calculated as the difference between the surrogate optimum of the
        current iteration and the surrogate optimum 'convergence_window' iterations
        ago. Used to determine if the optimisation has converged, based on the
        convergence_threshold and relative_convergence_threshold.
        (default: 5, must be >= 1)
    - 'constant_liar_lie': str, the type of lie to use in the Constant Liar algorithm.
        (default: "min", allowed: ["min", "max", "mean"])
    - 'max_function_calls': int | None, maximum number of function calls (all
        fidelities), if None this stopping criterion will not be used. One HF call is
        reserved for the final evaluation of the surrogate optimum, which is included.
        (default: 1000, must be >= 1)
    - 'max_function_calls_HF': int | None, maximum number of function calls for
        high fidelity, if None this stopping criterion will not be used. One call is
        reserved for the final evaluation of the surrogate optimum, which is included.
        (default: 150, must be >= 1)
    - 'max_iterations': int | None, maximum number of iterations, if None this stopping
        criterion will not be used. Not used by EgoAsync, which has no iterations.
        (default: 300, must be >= 1)
    - 'min_infill_value': float | None, minimum value for the infill criterion, if None
        this stopping criterion will not be used.
        (default: 1e-6, must be >= 0.0)
    - 'multipoint_algorithm': str, the algorithm to use for multipoint selection if the
        surrogate has not yet received the new points.
            1. "KB" for Kriging Believer (default)
            2. "CL" for Constant Liar
            (default: "KB", allowed: ["KB", "CL"])
    - 'relative_convergence_threshold': float | None, relative threshold
        $\varepsilon_r$ for convergence, if None this stopping criterion will not be
        used. Converged when
        $\lvert \hat{y}^*_{t-w} - \hat{y}^*_t \rvert / \lvert \hat{y}^*_{t-w} \rvert
        < \varepsilon_r$, with the notation of 'convergence_threshold' (the
        denominator is at least the machine epsilon)
        (default: 1e-6, must be >= 0.0)
"""

OPTIMISATION_ALL_DEFAULT_SETTINGS = {
    "convergence_threshold": {"type": float | type(None), "default": 1e-6, "min": 0.0},
    "convergence_window": {"type": int | type(None), "default": 5, "min": 1},
    "constant_liar_lie": {
        "type": str,
        "default": "min",
        "allowed": ["min", "max", "mean"],
    },
    "max_function_calls": {"type": int | type(None), "default": 1000, "min": 1},
    "max_function_calls_HF": {"type": int | type(None), "default": 150, "min": 1},
    "max_iterations": {"type": int | type(None), "default": 300, "min": 1},
    "min_infill_value": {"type": float | type(None), "default": 1e-6, "min": 0.0},
    "multipoint_algorithm": {"type": str, "default": "KB", "allowed": ["KB", "CL"]},
    "relative_convergence_threshold": {
        "type": float | type(None),
        "default": 1e-6,
        "min": 0.0,
    },
}

OPTIMISATIONBASE_DEFAULT_SETTINGS = OPTIMISATION_ALL_DEFAULT_SETTINGS
