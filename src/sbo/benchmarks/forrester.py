"""The Forrester function: a 1D benchmark with two fidelities.

References
----------
Forrester, A. I. J., Sóbester, A. and Keane, A. J. (2007). Multi-fidelity optimization
via surrogate modelling. Proceedings of the Royal Society A, 463(2088), 3251-3269.
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 1


def forrester_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Forrester function.

    $$
    f_h(x) = (6x - 2)^2 \sin(12x - 4)
    $$

    Parameters
    ----------
    x
        One point with shape (1,), or several points with shape (n, 1).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)[:, 0]
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def forrester_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Forrester function.

    $$
    f_l(x) = A f_h(x) + B (x - 0.5) + C
    $$

    Parameters
    ----------
    x
        One point with shape (1,), or several points with shape (n, 1).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x_2d = as_2d(x, N_DIM)
    return 0.5 * forrester_high(x_2d) + 10 * (x_2d[:, 0] - 0.5) - 5


FORRESTER = MultiFidelityFunction(
    name="forrester",
    functions=(forrester_high, forrester_low),
    l_bound=np.array([0.0]),
    u_bound=np.array([1.0]),
    x_opt=np.array([0.7572487585]),
    f_opt=-6.0207400557670825,
)
"""The Forrester function (1D, two fidelities)."""
