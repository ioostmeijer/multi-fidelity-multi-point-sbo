import numpy as np
import pytest

from sbo.benchmarks import FORRESTER
from sbo.benchmarks.forrester import forrester_high, forrester_low

# Reference values at x = 0, 0.5 and 1 (the same as mf2 2022.6.0, used for the thesis)
X = np.array([[0.0], [0.5], [1.0]])
HIGH = np.array([3.027209981231713, 0.9092974268256817, 15.829731945974109])
LOW = np.array([-8.486395009384143, -4.5453512865871595, 7.914865972987055])


@pytest.mark.unit
def test_forrester_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(forrester_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_forrester_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(forrester_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_forrester_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert FORRESTER.name == "forrester"
    assert FORRESTER.functions == (forrester_high, forrester_low)
    np.testing.assert_array_equal(FORRESTER.l_bound, [0.0])
    np.testing.assert_array_equal(FORRESTER.u_bound, [1.0])
    assert forrester_high(FORRESTER.x_opt)[0] == pytest.approx(FORRESTER.f_opt)
