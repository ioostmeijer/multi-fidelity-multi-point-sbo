import numpy as np
import pytest

from sbo.benchmarks.bohachevsky import BOHACHEVSKY, bohachevsky_high, bohachevsky_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [-4.0, -1.04],
        [-2.16, 0.8],
        [-0.32, 2.64],
    ]
)
HIGH = np.array([18.21267732798245, 6.950369641891186, 15.114186936228634])
LOW = np.array([2.4199722296699697, -9.104392286787975, 2.073860385968745])


@pytest.mark.unit
def test_bohachevsky_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(bohachevsky_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_bohachevsky_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(bohachevsky_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_bohachevsky_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert BOHACHEVSKY.name == "bohachevsky"
    assert BOHACHEVSKY.functions == (bohachevsky_high, bohachevsky_low)
    np.testing.assert_array_equal(BOHACHEVSKY.l_bound, [-5.0, -5.0])
    np.testing.assert_array_equal(BOHACHEVSKY.u_bound, [5.0, 5.0])
    assert bohachevsky_high(BOHACHEVSKY.x_opt)[0] == pytest.approx(BOHACHEVSKY.f_opt)
