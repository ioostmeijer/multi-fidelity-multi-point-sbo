r"""The Booth function: a 2D benchmark with two fidelities.

References
----------
Dong, H., Song, B., Wang, P. et al. Multi-fidelity information fusion based on
prediction of kriging. Struct Multidisc Optim 51, 1267–1280 (2015)
doi:10.1007/s00158-014-1213-9
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 2


def booth_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Booth function.

    $$
    f_h(x_1, x_2) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2
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
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def booth_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Booth function.

    $$
    f_l(x_1, x_2) = f_h(0.4x_1, x_2) + 1.7 x_1 x_2 - x_1 + 2 x_2
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
    return booth_high(np.column_stack((0.4 * x1, x2))) + 1.7 * x1 * x2 - x1 + 2 * x2


BOOTH = MultiFidelityFunction(
    name="booth",
    functions=(booth_high, booth_low),
    l_bound=np.array([-10.0, -10.0]),
    u_bound=np.array([10.0, 10.0]),
    x_opt=np.array([1.0, 3.0]),
    f_opt=0.0,
)
"""The Booth function (2D, two fidelities)."""
