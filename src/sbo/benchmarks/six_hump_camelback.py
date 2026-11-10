r"""The six-hump camelback function: a 2D benchmark with two fidelities.

References
----------
Dong, H., Song, B., Wang, P. et al. Multi-fidelity information fusion based on
prediction of kriging. Struct Multidisc Optim 51, 1267–1280 (2015)
doi:10.1007/s00158-014-1213-9
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 2


def six_hump_camelback_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the six-hump camelback function.

    $$
    f_h(x_1, x_2) = 4x_1^2 - 2.1x_1^4 + \frac{x_1^6}{3} + x_1 x_2 - 4x_2^2 + 4x_2^4
    $$

    Parameters
    ----------
    x
        One point with shape (2,), or several points with shape (n, 2).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 4 * x1**2 - 2.1 * x1**4 + x1**6 / 3 + x1 * x2 - 4 * x2**2 + 4 * x2**4


def six_hump_camelback_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the six-hump camelback function.

    $$
    f_l(x_1, x_2) = f_h(0.7x_1, 0.7x_2) + x_1 x_2 - 15
    $$

    Parameters
    ----------
    x
        One point with shape (2,), or several points with shape (n, 2).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return six_hump_camelback_high(0.7 * x) + x1 * x2 - 15


SIX_HUMP_CAMELBACK = MultiFidelityFunction(
    name="six_hump_camelback",
    functions=(six_hump_camelback_high, six_hump_camelback_low),
    l_bound=np.array([-2.0, -2.0]),
    u_bound=np.array([2.0, 2.0]),
    x_opt=np.array([0.0898, -0.7126]),
    f_opt=-1.0316284229280819,
)
"""The six-hump camelback function (2D, two fidelities)."""
