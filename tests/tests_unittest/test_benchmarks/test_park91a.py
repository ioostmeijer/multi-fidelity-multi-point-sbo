import numpy as np
import pytest

from sbo.benchmarks.park91a import PARK91A, park91a_high, park91a_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [0.1, 0.396, 0.692, 0.188],
        [0.284, 0.58, 0.876, 0.372],
        [0.468, 0.764, 0.26, 0.556],
    ]
)
HIGH = np.array([3.575311856586606, 8.443688392561317, 7.687080879268848])
LOW = np.array([4.546685416408701, 9.716054556398769, 8.24914282288209])


@pytest.mark.unit
def test_park91a_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(park91a_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_park91a_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(park91a_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_park91a_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert PARK91A.name == "park91a"
    assert PARK91A.functions == (park91a_high, park91a_low)
    np.testing.assert_array_equal(PARK91A.l_bound, [1e-08, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(PARK91A.u_bound, [1.0, 1.0, 1.0, 1.0])
    assert park91a_high(PARK91A.x_opt)[0] == pytest.approx(PARK91A.f_opt)
