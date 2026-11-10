r"""The Bohachevsky function: a 2D benchmark with two fidelities.

References
----------
Dong, H., Song, B., Wang, P. et al. Multi-fidelity information fusion based on
prediction of kriging. Struct Multidisc Optim 51, 1267–1280 (2015)
doi:10.1007/s00158-014-1213-9
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 2


def bohachevsky_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Bohachevsky function.

    $$
    f_h(x_1, x_2) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1) - 0.4\cos(4\pi x_2) + 0.7
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
    return (
        x1**2
        + 2 * x2**2
        - 0.3 * np.cos(3 * np.pi * x1)
        - 0.4 * np.cos(4 * np.pi * x2)
        + 0.7
    )


def bohachevsky_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Bohachevsky function.

    $$
    f_l(x_1, x_2) = f_h(0.7x_1, x_2) + x_1 x_2 - 12
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
    return bohachevsky_high(np.column_stack((0.7 * x1, x2))) + x1 * x2 - 12


BOHACHEVSKY = MultiFidelityFunction(
    name="bohachevsky",
    functions=(bohachevsky_high, bohachevsky_low),
    l_bound=np.array([-5.0, -5.0]),
    u_bound=np.array([5.0, 5.0]),
    x_opt=np.array([0.0, 0.0]),
    f_opt=0.0,
)
"""The Bohachevsky function (2D, two fidelities)."""
