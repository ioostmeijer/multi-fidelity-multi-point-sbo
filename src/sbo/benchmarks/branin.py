r"""The Branin function: a 2D benchmark with two fidelities.

References
----------
Dong, H., Song, B., Wang, P. et al. Multi-fidelity information fusion based on
prediction of kriging. Struct Multidisc Optim 51, 1267–1280 (2015)
doi:10.1007/s00158-014-1213-9
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 2


def branin_base(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the base function of the Branin function.

    $$
    f_b(x_1, x_2) = \left( x_2 - \frac{5.1 x_1^2}{4\pi^2} + \frac{5x_1}{\pi} - 6
    \right)^2 + 10 \left( 1 - \frac{1}{8\pi} \right) \cos(x_1) + 10
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
    term1 = x2 - 5.1 * x1**2 / (4 * np.pi**2) + 5 * x1 / np.pi - 6
    term2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
    return term1**2 + term2 + 10


def branin_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Branin function.

    $$
    f_h(x_1, x_2) = f_b(x_1, x_2) - 22.5x_2
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
    return branin_base(x) - 22.5 * x[:, 1]


def branin_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Branin function.

    $$
    f_l(x_1, x_2) = f_b(0.7x_1, 0.7x_2) - 15.75x_2 + 20(0.9 + x_1)^2 - 50
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
    return branin_base(0.7 * x) - 15.75 * x2 + 20 * (0.9 + x1) ** 2 - 50


BRANIN = MultiFidelityFunction(
    name="branin",
    functions=(branin_high, branin_low),
    l_bound=np.array([-5.0, 0.0]),
    u_bound=np.array([10.0, 15.0]),
    x_opt=np.array([-3.786088705282203, 15.0]),
    f_opt=-333.91603435227887,
)
"""The Branin function (2D, two fidelities)."""
