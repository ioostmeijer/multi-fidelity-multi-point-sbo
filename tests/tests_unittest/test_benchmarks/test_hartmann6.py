import numpy as np
import pytest

from sbo.benchmarks.hartmann6 import HARTMANN6, hartmann6_high, hartmann6_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [0.19, 0.4564, 0.7228, 0.2692, 0.5356, 0.802],
        [0.3556, 0.622, 0.8884, 0.4348, 0.7012, 0.2476],
        [0.5212, 0.7876, 0.334, 0.6004, 0.8668, 0.4132],
    ]
)
HIGH = np.array([-1.8098921195981634, -1.7359503285631537, -1.4908047918401255])
LOW = np.array([-1.5496429544678552, -1.6980964663340203, -1.5031700138450221])


@pytest.mark.unit
def test_hartmann6_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(hartmann6_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_hartmann6_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(hartmann6_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_hartmann6_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert HARTMANN6.name == "hartmann6"
    assert HARTMANN6.functions == (hartmann6_high, hartmann6_low)
    np.testing.assert_array_equal(HARTMANN6.l_bound, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    np.testing.assert_array_equal(HARTMANN6.u_bound, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert hartmann6_high(HARTMANN6.x_opt)[0] == pytest.approx(HARTMANN6.f_opt)
