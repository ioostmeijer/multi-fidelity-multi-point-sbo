import numpy as np
import pytest

from sbo.benchmarks.booth import BOOTH, booth_high, booth_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [-8.0, -2.08],
        [-4.32, 1.6],
        [-0.64, 5.28],
    ]
)
HIGH = np.array([899.7919999999999, 210.89600000000004, 9.526399999999999])
LOW = np.array([420.048, 73.33312000000001, 16.425600000000003])


@pytest.mark.unit
def test_booth_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(booth_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_booth_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(booth_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_booth_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert BOOTH.name == "booth"
    assert BOOTH.functions == (booth_high, booth_low)
    np.testing.assert_array_equal(BOOTH.l_bound, [-10.0, -10.0])
    np.testing.assert_array_equal(BOOTH.u_bound, [10.0, 10.0])
    assert booth_high(BOOTH.x_opt)[0] == pytest.approx(BOOTH.f_opt)
