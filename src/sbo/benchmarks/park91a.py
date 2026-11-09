r"""The Park91a function: a 4D benchmark with two fidelities.

References
----------
Shifeng Xiong, Peter Z. G. Qian & C. F. Jeff Wu (2013) Sequential Design and Analysis
of High-Accuracy and Low-Accuracy Computer Codes, Technometrics, 55:1, 37-46,
DOI: 10.1080/00401706.2012.723572
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 4


def park91a_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Park91a function.

    $$
    f_h(x) = \frac{x_1}{2} \left( \sqrt{1 + (x_2 + x_3^2) \frac{x_4}{x_1^2}} - 1
    \right) + (x_1 + 3x_4) \exp(1 + \sin(x_3))
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
    term1 = x1 / 2 * (np.sqrt(1 + (x2 + x3**2) * x4 / x1**2) - 1)
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    return term1 + term2


def park91a_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Park91a function.

    $$
    f_l(x) = \left( 1 + \frac{\sin(x_1)}{10} \right) f_h(x) - 2x_1 + x_2^2 + x_3^2
    + 0.5
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
    return (1 + np.sin(x1) / 10) * park91a_high(x) - 2 * x1 + x2**2 + x3**2 + 0.5


PARK91A = MultiFidelityFunction(
    name="park91a",
    functions=(park91a_high, park91a_low),
    l_bound=np.array([1e-8, 0.0, 0.0, 0.0]),
    u_bound=np.array([1.0, 1.0, 1.0, 1.0]),
    x_opt=np.array([1e-8, 0.0, 0.0, 0.0]),
    f_opt=2.718281828459045e-08,
)
"""The Park91a function (4D, two fidelities)."""
