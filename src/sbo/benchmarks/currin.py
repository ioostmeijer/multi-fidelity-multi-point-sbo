r"""The Currin function: a 2D benchmark with two fidelities.

References
----------
Shifeng Xiong, Peter Z. G. Qian & C. F. Jeff Wu (2013) Sequential Design and Analysis
of High-Accuracy and Low-Accuracy Computer Codes, Technometrics, 55:1, 37-46,
DOI: 10.1080/00401706.2012.723572
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 2


def currin_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Currin function.

    $$
    f_h(x_1, x_2) = \left( 1 - \exp\left( -\frac{1}{2x_2} \right) \right)
    \frac{2300x_1^3 + 1900x_1^2 + 2092x_1 + 60}{100x_1^3 + 500x_1^2 + 4x_1 + 20}
    $$

    At $x_2 = 0$ the exponential is 0, so the first factor is 1.

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
    # At x2 = 0 (the optimum), -1 / 0 gives -inf and exp(-inf) gives 0, which is the
    # correct limit, so the division warning is not useful.
    with np.errstate(divide="ignore"):
        factor = 1 - np.exp(-1 / (2 * x2))
    return (
        factor
        * (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60)
        / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)
    )


def currin_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Currin function.

    $$
    f_l(x_1, x_2) = \frac{1}{4} \big(
        f_h(x_1 + 0.05, x_2 + 0.05) + f_h(x_1 + 0.05, x_2 - 0.05)
        + f_h(x_1 - 0.05, x_2 + 0.05) + f_h(x_1 - 0.05, x_2 - 0.05)
    \big)
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
    return (
        currin_high(x + np.array([0.05, 0.05]))
        + currin_high(x + np.array([0.05, -0.05]))
        + currin_high(x + np.array([-0.05, 0.05]))
        + currin_high(x + np.array([-0.05, -0.05]))
    ) / 4


CURRIN = MultiFidelityFunction(
    name="currin",
    functions=(currin_high, currin_low),
    l_bound=np.array([0.0, 0.0]),
    u_bound=np.array([1.0, 1.0]),
    x_opt=np.array([0.21666666666666, 0.0]),
    f_opt=13.798722044728434,
)
"""The Currin function (2D, two fidelities)."""
