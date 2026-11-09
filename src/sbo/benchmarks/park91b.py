r"""The Park91b function: a 4D benchmark with two fidelities.

References
----------
Shifeng Xiong, Peter Z. G. Qian & C. F. Jeff Wu (2013) Sequential Design and Analysis
of High-Accuracy and Low-Accuracy Computer Codes, Technometrics, 55:1, 37-46,
DOI: 10.1080/00401706.2012.723572
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 4


def park91b_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Park91b function.

    $$
    f_h(x) = \frac{2}{3} \exp(x_1 + x_2) - x_4 \sin(x_3) + x_3
    $$

    Parameters
    ----------
    x
        One point with shape (4,), or several points with shape (n, 4).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    return 2 / 3 * np.exp(x1 + x2) - x4 * np.sin(x3) + x3


def park91b_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Park91b function.

    $$
    f_l(x) = 1.2 f_h(x) - 1
    $$

    Parameters
    ----------
    x
        One point with shape (4,), or several points with shape (n, 4).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    return 1.2 * park91b_high(x) - 1


PARK91B = MultiFidelityFunction(
    name="park91b",
    functions=(park91b_high, park91b_low),
    l_bound=np.array([0.0, 0.0, 0.0, 0.0]),
    u_bound=np.array([1.0, 1.0, 1.0, 1.0]),
    x_opt=np.array([0.0, 0.0, 0.0, 0.0]),
    f_opt=0.6666666666666666,
)
"""The Park91b function (4D, two fidelities)."""
