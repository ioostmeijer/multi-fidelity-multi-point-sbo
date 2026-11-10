r"""The Himmelblau function: a 2D benchmark with two fidelities.

References
----------
Dong, H., Song, B., Wang, P. et al. Multi-fidelity information fusion based on
prediction of kriging. Struct Multidisc Optim 51, 1267–1280 (2015)
doi:10.1007/s00158-014-1213-9
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 2


def himmelblau_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Himmelblau function.

    $$
    f_h(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_2^2 + x_1 - 7)^2
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
    return (x1**2 + x2 - 11) ** 2 + (x2**2 + x1 - 7) ** 2


def himmelblau_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Himmelblau function.

    $$
    f_l(x_1, x_2) = f_h(0.5x_1, 0.8x_2) + x_2^3 - (x_1 + 1)^2
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
        himmelblau_high(np.column_stack((0.5 * x1, 0.8 * x2))) + x2**3 - (x1 + 1) ** 2
    )


HIMMELBLAU = MultiFidelityFunction(
    name="himmelblau",
    functions=(himmelblau_high, himmelblau_low),
    l_bound=np.array([-4.0, -4.0]),
    u_bound=np.array([4.0, 4.0]),
    x_opt=np.array([3.0, 2.0]),
    f_opt=0.0,
)
"""The Himmelblau function (2D, two fidelities)."""
