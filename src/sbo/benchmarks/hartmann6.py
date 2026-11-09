r"""The Hartmann6 function: a 6D benchmark with two fidelities.

References
----------
Park, C., Haftka, R.T., Kim, N.H. Remarks on multi-fidelity surrogates. (2016)
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 6

A_MATRIX = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
"""The matrix $A$, with shape (4, 6)."""

P_MATRIX = np.array(
    [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
)
"""The matrix $P$, with shape (4, 6)."""

ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
r"""The weights $\alpha$ of the high fidelity."""

ALPHA_PRIME = np.array([0.5, 0.5, 2.0, 4.0])
r"""The weights $\alpha'$ of the low fidelity."""


def _exponents(x: np.ndarray) -> np.ndarray:
    r"""Return $-\sum_{j=1}^{6} A_{ij} (x_j - P_{ij})^2$ for every $i$.

    Parameters
    ----------
    x
        Points with shape (n, 6).

    Returns
    -------
    np.ndarray
        The exponents, with shape (n, 4).
    """
    return -np.sum(A_MATRIX * (x[:, np.newaxis, :] - P_MATRIX) ** 2, axis=2)


def _hartmann6_exp(z: np.ndarray) -> np.ndarray:
    r"""Evaluate the approximation of the exponential used by the low fidelity.

    $$
    f_{\exp}(z) = \left( e^{-4/9} + e^{-4/9} \frac{z + 4}{9} \right)^9
    $$

    Parameters
    ----------
    z
        The exponents, with any shape.

    Returns
    -------
    np.ndarray
        The approximated exponentials, with the same shape as ``z``.
    """
    return (np.exp(-4 / 9) + np.exp(-4 / 9) * (z + 4) / 9) ** 9


def hartmann6_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the Hartmann6 function.

    $$
    f_h(x) = -\frac{1}{1.94} \left( 2.58 + \sum_{i=1}^{4} \alpha_i
    \exp\left( -\sum_{j=1}^{6} A_{ij} (x_j - P_{ij})^2 \right) \right)
    $$

    Parameters
    ----------
    x
        One point with shape (6,), or several points with shape (n, 6).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)
    return -(2.58 + np.exp(_exponents(x)) @ ALPHA) / 1.94


def hartmann6_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the Hartmann6 function.

    $$
    f_l(x) = -\frac{1}{1.94} \left( 2.58 + \sum_{i=1}^{4} \alpha'_i
    f_{\exp}\left( -\sum_{j=1}^{6} A_{ij} (x_j - P_{ij})^2 \right) \right)
    $$

    with $f_{\exp}(z) = \left( e^{-4/9} + e^{-4/9} \frac{z + 4}{9} \right)^9$.

    Parameters
    ----------
    x
        One point with shape (6,), or several points with shape (n, 6).

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)
    return -(2.58 + _hartmann6_exp(_exponents(x)) @ ALPHA_PRIME) / 1.94


HARTMANN6 = MultiFidelityFunction(
    name="hartmann6",
    functions=(hartmann6_high, hartmann6_low),
    l_bound=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    u_bound=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    x_opt=np.array([0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573]),
    f_opt=-3.0424576600692044,
)
"""The Hartmann6 function (6D, two fidelities)."""
