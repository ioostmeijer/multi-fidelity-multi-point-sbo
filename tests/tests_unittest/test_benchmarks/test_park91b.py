import numpy as np
import pytest

from sbo.benchmarks.park91b import PARK91B, park91b_high, park91b_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [0.1, 0.396, 0.692, 0.188],
        [0.284, 0.58, 0.876, 0.372],
        [0.468, 0.764, 0.26, 0.556],
    ]
)
HIGH = np.array([1.6668009659843275, 2.171990353805411, 2.402449108071459])
LOW = np.array([1.000161159181193, 1.606388424566493, 1.8829389296857508])


@pytest.mark.unit
def test_park91b_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(park91b_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_park91b_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(park91b_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_park91b_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert PARK91B.name == "park91b"
    assert PARK91B.functions == (park91b_high, park91b_low)
    np.testing.assert_array_equal(PARK91B.l_bound, [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(PARK91B.u_bound, [1.0, 1.0, 1.0, 1.0])
    assert park91b_high(PARK91B.x_opt)[0] == pytest.approx(PARK91B.f_opt)
