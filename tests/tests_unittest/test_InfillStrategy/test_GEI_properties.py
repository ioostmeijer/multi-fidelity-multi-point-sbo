from types import SimpleNamespace

import numpy as np
import pytest
from dummy_surrogate import DummySurrogateWithVariance
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from logger_helper import _make_logger

from sbo import Config
from sbo.infill.EI import EI
from sbo.infill.GEI import GEI

dummy_config = Config(
    used_algorithms={"infill": "GEI"},
    lb=[0.0],
    ub=[1.0],
    functions=[lambda x: x**2],
    settings=None,
    logger=_make_logger(),
)

_VALUES = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)
_VARIANCES = st.one_of(
    st.just(0.0),
    st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_subnormal=True),
)
_EXPONENTS = st.integers(min_value=0, max_value=4)


class _FixedPredictions(DummySurrogateWithVariance):
    """A surrogate that returns the given mean and variance, one row per point."""

    def __init__(self, mu: np.ndarray, var: np.ndarray) -> None:
        super().__init__(dummy_config)
        self._mu_values = np.asarray(mu, dtype=float).reshape(-1, 1)
        self._var_values = np.asarray(var, dtype=float).reshape(-1, 1)

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._mu_values

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._var_values


def _calc_gei(mu: np.ndarray, var: np.ndarray, best_value: float, g: int) -> np.ndarray:
    """Return the GEI with exponent g of every prediction as a 1D array."""
    gei = GEI.__new__(GEI)
    gei.config = SimpleNamespace(
        logger=_make_logger(), get=lambda key: {"infill.g": g}[key]
    )
    value = gei.calc_infill_value(
        np.zeros((len(mu), 1)), _FixedPredictions(mu, var), best_value=best_value
    )
    return np.atleast_1d(np.asarray(value, dtype=float))


def _tolerance(
    mu: np.ndarray, var: np.ndarray, best_value: float, g: int
) -> np.ndarray:
    """Return an absolute tolerance for rounding, relative to the scale of I^g."""
    scale = np.abs(best_value - mu) + np.sqrt(var)
    return 1e-9 * scale**g + 1e-12


@st.composite
def _predictions(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, float]:
    """Draw the mean and variance of n points and the best value.

    Parameters
    ----------
    draw
        The Hypothesis draw function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        The mean and variance, each with shape (n,), and the best value.
    """
    n_points = draw(st.integers(min_value=1, max_value=5))
    mu = draw(arrays(float, n_points, elements=_VALUES))
    var = draw(arrays(float, n_points, elements=_VARIANCES))
    best_value = draw(_VALUES)
    return mu, var, best_value


@pytest.mark.unit
@given(predictions=_predictions(), g=_EXPONENTS)
def test_gei_is_finite_and_not_negative(
    predictions: tuple[np.ndarray, np.ndarray, float], g: int
) -> None:
    """Test that GEI is finite and not negative, one value per point."""
    mu, var, best_value = predictions
    gei = _calc_gei(mu, var, best_value, g)
    assert gei.shape == mu.shape
    assert np.all(np.isfinite(gei))
    assert np.all(gei >= -_tolerance(mu, var, best_value, g))


@pytest.mark.unit
@given(predictions=_predictions(), g=st.integers(min_value=1, max_value=4))
def test_gei_is_at_least_improvement_to_power_g(
    predictions: tuple[np.ndarray, np.ndarray, float], g: int
) -> None:
    r"""Test $\mathrm{GEI}_g \ge \max(y_{\min} - \hat{y}, 0)^g$ for $g \ge 1$."""
    mu, var, best_value = predictions
    gei = _calc_gei(mu, var, best_value, g)
    lower = np.maximum(best_value - mu, 0.0) ** g
    assert np.all(gei >= lower - _tolerance(mu, var, best_value, g))


@pytest.mark.unit
@given(predictions=_predictions())
def test_gei_with_g0_is_probability(
    predictions: tuple[np.ndarray, np.ndarray, float],
) -> None:
    """Test that GEI with g = 0 (probability of improvement) is between 0 and 1."""
    mu, var, best_value = predictions
    gei = _calc_gei(mu, var, best_value, 0)
    assert np.all((gei >= 0.0) & (gei <= 1.0))


@pytest.mark.unit
@given(predictions=_predictions())
def test_gei_with_g1_is_ei(predictions: tuple[np.ndarray, np.ndarray, float]) -> None:
    """Test that GEI with g = 1 is the Expected Improvement."""
    mu, var, best_value = predictions
    ei = EI.__new__(EI)
    ei.config = SimpleNamespace(logger=_make_logger(), num_fidelities=1)
    expected = np.atleast_1d(
        ei.calc_infill_value(
            np.zeros((len(mu), 1)), _FixedPredictions(mu, var), best_value=best_value
        )
    )
    np.testing.assert_allclose(
        _calc_gei(mu, var, best_value, 1),
        expected,
        rtol=1e-9,
        atol=_tolerance(mu, var, best_value, 1).max(),
    )


@pytest.mark.unit
@given(
    mu=arrays(float, st.integers(min_value=1, max_value=5), elements=_VALUES),
    best_value=_VALUES,
    g=_EXPONENTS,
)
def test_gei_without_variance_is_certain_improvement(
    mu: np.ndarray, best_value: float, g: int
) -> None:
    """Test that GEI is I^g where the variance is 0 (for g = 0: 1 or 0)."""
    gei = _calc_gei(mu, np.zeros_like(mu), best_value, g)
    improvement = best_value - mu
    expected = np.where(improvement > 0.0, np.maximum(improvement, 0.0) ** g, 0.0)
    np.testing.assert_allclose(gei, expected, rtol=1e-12)


@pytest.mark.unit
@given(predictions=_predictions(), g=_EXPONENTS)
def test_gei_of_point_does_not_depend_on_batch(
    predictions: tuple[np.ndarray, np.ndarray, float], g: int
) -> None:
    """Test that a point gets the same GEI alone as in a batch of points."""
    mu, var, best_value = predictions
    batch = _calc_gei(mu, var, best_value, g)
    for i in range(len(mu)):
        alone = _calc_gei(mu[i : i + 1], var[i : i + 1], best_value, g)
        np.testing.assert_allclose(alone, batch[i : i + 1], rtol=1e-12, atol=1e-12)
