from types import NoneType

from .SMT import SMT_SURROGATE_DEFAULT_SETTINGS

r"""Default settings for the Radial Basis Function (RBF) surrogate model.

Keys:
    - 'data_dir': str, Directory for loading / saving cached data; None means do not
        save or load (default: None)
    - 'd0': float, scaling $d_0$ of the basis functions $\exp(-d^2 / d_0^2)$, with $d$
        the distance to a sample
        (default: 1.0, min: 0.0)
    - 'max_print_depth': int, maximum depth for printing
        (default: 5, min: 1)
    - 'poly_degree': int, -1 means no global polynomial, 0 means constant, 1 means
            linear_trend
        (default: -1, allowed: [-1, 0, 1])
    - 'reg': float, regularisation parameter
        (default: 1e-10, min: 0.0)
    In addition all keys from SMT_SURROGATE_DEFAULT_SETTINGS.
"""
RBF_DEFAULT_SETTINGS = {
    **SMT_SURROGATE_DEFAULT_SETTINGS,
    "data_dir": {"type": str | NoneType, "default": None},
    "d0": {"type": float, "default": 1.0, "min": 0.0},
    "max_print_depth": {"type": int, "default": 5, "min": 1},
    "poly_degree": {"type": int, "default": -1, "allowed": [-1, 0, 1]},
    "reg": {"type": float, "default": 1e-10, "min": 0.0},
}
