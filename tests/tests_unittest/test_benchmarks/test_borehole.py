import numpy as np
import pytest

from sbo.benchmarks.borehole import BOREHOLE, borehole_high, borehole_low

# Reference values at three points inside the bounds (from mf2 2022.6.0, used for the
# thesis)
X = np.array(
    [
        [0.06, 19860.4, 99420.76, 1012.56, 88.7036, 793.6, 1274.56, 11107.68],
        [0.0784, 29042.0, 109086.28, 1034.64, 98.4372, 719.68, 1377.6, 11510.64],
        [0.0968, 38223.6, 76727.8, 1056.72, 108.1708, 741.76, 1480.64, 10161.6],
    ]
)
HIGH = np.array([21.532988094015046, 50.64798382046952, 63.38783170949964])
LOW = np.array([17.135390330441147, 40.30432428875143, 50.44229806355933])


@pytest.mark.unit
def test_borehole_high_values() -> None:
    """Test the high fidelity against reference values."""
    np.testing.assert_allclose(borehole_high(X), HIGH, rtol=1e-12)


@pytest.mark.unit
def test_borehole_low_values() -> None:
    """Test the low fidelity against reference values."""
    np.testing.assert_allclose(borehole_low(X), LOW, rtol=1e-12)


@pytest.mark.unit
def test_borehole_definition() -> None:
    """Test the fidelity order, bounds and optimum."""
    assert BOREHOLE.name == "borehole"
    assert BOREHOLE.functions == (borehole_high, borehole_low)
    np.testing.assert_array_equal(
        BOREHOLE.l_bound, [0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0]
    )
    np.testing.assert_array_equal(
        BOREHOLE.u_bound,
        [0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0],
    )
    assert borehole_high(BOREHOLE.x_opt)[0] == pytest.approx(BOREHOLE.f_opt)
