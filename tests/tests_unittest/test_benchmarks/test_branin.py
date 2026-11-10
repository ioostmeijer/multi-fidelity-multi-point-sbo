import numpy as np
import pytest

from sbo.benchmarks.branin import BRANIN, branin_high, branin_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [-3.5, 5.94],
        [-0.74, 8.7],
        [2.02, 11.46],
    ]
)
HIGH = np.array([-80.61555659582787, -176.55225467686122, -185.63296903446113])
LOW = np.array([36.71687208574042, -167.5790838215462, -32.35410581027898])


@pytest.mark.unit
def test_branin_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(branin_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_branin_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(branin_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_branin_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert BRANIN.name == "branin"
    assert BRANIN.functions == (branin_high, branin_low)
    np.testing.assert_array_equal(BRANIN.l_bound, [-5.0, 0.0])
    np.testing.assert_array_equal(BRANIN.u_bound, [10.0, 15.0])
    assert branin_high(BRANIN.x_opt)[0] == pytest.approx(BRANIN.f_opt)
