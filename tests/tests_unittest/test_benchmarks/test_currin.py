import numpy as np
import pytest

from sbo.benchmarks.currin import CURRIN, currin_high, currin_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [0.1, 0.396],
        [0.284, 0.58],
        [0.468, 0.764],
    ]
)
HIGH = np.array([8.169201150753773, 7.793422865165145, 5.733926416198895])
LOW = np.array([7.582391812332353, 7.773492682441669, 5.757585656585726])


@pytest.mark.unit
def test_currin_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(currin_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_currin_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(currin_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_currin_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert CURRIN.name == "currin"
    assert CURRIN.functions == (currin_high, currin_low)
    np.testing.assert_array_equal(CURRIN.l_bound, [0.0, 0.0])
    np.testing.assert_array_equal(CURRIN.u_bound, [1.0, 1.0])
    assert currin_high(CURRIN.x_opt)[0] == pytest.approx(CURRIN.f_opt)
