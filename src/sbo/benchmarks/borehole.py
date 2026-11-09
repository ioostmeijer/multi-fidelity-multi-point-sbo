r"""The borehole function: an 8D benchmark with two fidelities.

The design variables are $x = [r_w, r, T_u, H_u, T_l, H_l, L, K_w]$.

References
----------
Shifeng Xiong, Peter Z. G. Qian & C. F. Jeff Wu (2013) Sequential Design and Analysis
of High-Accuracy and Low-Accuracy Computer Codes, Technometrics, 55:1, 37-46,
DOI: 10.1080/00401706.2012.723572
"""

import numpy as np

from .base import MultiFidelityFunction, as_2d

N_DIM = 8


def borehole_base(x: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""Evaluate the base function of the borehole function.

    $$
    f_b(x, a, b) = \frac{a T_u (H_u - H_l)}{\ln\left( \frac{r}{r_w} \right) \left( b
    + \frac{2 L T_u}{\ln\left( \frac{r}{r_w} \right) r_w^2 K_w} + \frac{T_u}{T_l}
    \right)}
    $$

    Parameters
    ----------
    x
        One point with shape (8,), or several points with shape (n, 8), with
        $x = [r_w, r, T_u, H_u, T_l, H_l, L, K_w]$.
    a
        The factor $a$ in the numerator.
    b
        The term $b$ in the denominator.

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    x = as_2d(x, N_DIM)
    r_w, r, t_u, h_u, t_l, h_l, length, k_w = x.T
    log_ratio = np.log(r / r_w)
    numerator = a * t_u * (h_u - h_l)
    denominator = log_ratio * (
        b + 2 * length * t_u / (log_ratio * r_w**2 * k_w) + t_u / t_l
    )
    return numerator / denominator


def borehole_high(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the high fidelity of the borehole function.

    $$
    f_h(x) = f_b(x, 2\pi, 1)
    $$

    Parameters
    ----------
    x
        One point with shape (8,), or several points with shape (n, 8), with
        $x = [r_w, r, T_u, H_u, T_l, H_l, L, K_w]$.

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    return borehole_base(x, 2 * np.pi, 1)


def borehole_low(x: np.ndarray) -> np.ndarray:
    r"""Evaluate the low fidelity of the borehole function.

    $$
    f_l(x) = f_b(x, 5, 1.5)
    $$

    Parameters
    ----------
    x
        One point with shape (8,), or several points with shape (n, 8), with
        $x = [r_w, r, T_u, H_u, T_l, H_l, L, K_w]$.

    Returns
    -------
    np.ndarray
        The function values, with shape (n,).
    """
    return borehole_base(x, 5, 1.5)


BOREHOLE = MultiFidelityFunction(
    name="borehole",
    functions=(borehole_high, borehole_low),
    l_bound=np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]),
    u_bound=np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]),
    x_opt=np.array([0.05, 50000, 63070, 990, 63.1, 820, 1680, 9855]),
    f_opt=7.819676328755232,
)
"""The borehole function (8D, two fidelities)."""
