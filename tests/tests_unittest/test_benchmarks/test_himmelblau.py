import numpy as np
import pytest

from sbo.benchmarks.himmelblau import HIMMELBLAU, himmelblau_high, himmelblau_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [-3.2, -0.832],
        [-1.728, 0.64],
        [-0.256, 2.112],
    ]
)
HIGH = np.array([92.932268466176, 123.571890528256, 85.650445279232])
LOW = np.array([144.03228889750565, 152.41727482675196, 113.50654762431937])


@pytest.mark.unit
def test_himmelblau_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(himmelblau_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_himmelblau_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(himmelblau_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_himmelblau_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert HIMMELBLAU.name == "himmelblau"
    assert HIMMELBLAU.functions == (himmelblau_high, himmelblau_low)
    np.testing.assert_array_equal(HIMMELBLAU.l_bound, [-4.0, -4.0])
    np.testing.assert_array_equal(HIMMELBLAU.u_bound, [4.0, 4.0])
    assert himmelblau_high(HIMMELBLAU.x_opt)[0] == pytest.approx(HIMMELBLAU.f_opt)
