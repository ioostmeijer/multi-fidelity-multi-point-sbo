import numpy as np
import pytest

from sbo.benchmarks.six_hump_camelback import (
    SIX_HUMP_CAMELBACK,
    six_hump_camelback_high,
    six_hump_camelback_low,
)

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [-1.6, -0.416],
        [-0.864, 0.32],
        [-0.128, 1.056],
    ]
)
HIGH = np.array([2.1630148498773334, 1.310272050337677, 0.44337494554190426])
LOW = np.array([-11.947533097616454, -14.40411711790339, -16.160804837490453])


@pytest.mark.unit
def test_six_hump_camelback_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(six_hump_camelback_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_six_hump_camelback_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(six_hump_camelback_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_six_hump_camelback_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert SIX_HUMP_CAMELBACK.name == "six_hump_camelback"
    assert SIX_HUMP_CAMELBACK.functions == (
        six_hump_camelback_high,
        six_hump_camelback_low,
    )
    np.testing.assert_array_equal(SIX_HUMP_CAMELBACK.l_bound, [-2.0, -2.0])
    np.testing.assert_array_equal(SIX_HUMP_CAMELBACK.u_bound, [2.0, 2.0])
    assert six_hump_camelback_high(SIX_HUMP_CAMELBACK.x_opt)[0] == pytest.approx(
        SIX_HUMP_CAMELBACK.f_opt
    )
