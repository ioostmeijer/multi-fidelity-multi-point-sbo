import numpy as np
import pytest
from dummy_surrogate import DummySurrogateWithVariance
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from logger_helper import _make_logger
from scipy.stats import norm

from sbo import Config
from sbo.infill.EI import EI

dummy_config = Config(
    used_algorithms={"infill": "EI"},
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


def _make_ei() -> EI:
    """Create an EI infill strategy without a full Config."""
    ei = EI.__new__(EI)
    ei.config = type("Cfg", (), {"logger": _make_logger(), "num_fidelities": 1})()
    return ei


def _calc_ei(mu: np.ndarray, var: np.ndarray, best_value: float) -> np.ndarray:
    """Return the EI of every prediction as a 1D array."""
    surrogate = _FixedPredictions(mu, var)
    value = _make_ei().calc_infill_value(
        np.zeros((len(mu), 1)), surrogate, best_value=best_value
    )
    return np.atleast_1d(np.asarray(value, dtype=float))


@st.composite
def _predictions(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, float]:
    """Draw the mean and variance of n points and the best sampled value.

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
@given(predictions=_predictions())
def test_ei_is_finite_and_not_negative(
    predictions: tuple[np.ndarray, np.ndarray, float],
) -> None:
    """Test that EI is finite and never negative, one value per point."""
    mu, var, best_value = predictions
    ei = _calc_ei(mu, var, best_value)
    assert ei.shape == mu.shape
    assert np.all(np.isfinite(ei))
    assert np.all(ei >= 0.0)


@pytest.mark.unit
@given(predictions=_predictions())
def test_ei_is_between_improvement_and_upper_bound(
    predictions: tuple[np.ndarray, np.ndarray, float],
) -> None:
    r"""Test $\max(I, 0) \le \mathrm{EI} \le \max(I, 0) + \hat{s}\,\phi(0)$."""
    mu, var, best_value = predictions
    ei = _calc_ei(mu, var, best_value)
    improvement = np.maximum(best_value - mu, 0.0)
    upper = improvement + np.sqrt(var) * norm.pdf(0.0)
    tol = 1e-9
    assert np.all(ei >= improvement - tol)
    assert np.all(ei <= upper + tol)


@pytest.mark.unit
@given(
    mu=arrays(float, st.integers(min_value=1, max_value=5), elements=_VALUES),
    best_value=_VALUES,
)
def test_ei_without_variance_is_positive_improvement(
    mu: np.ndarray, best_value: float
) -> None:
    r"""Test that EI is $\max(y_{\min} - \hat{y}, 0)$ where the variance is 0."""
    ei = _calc_ei(mu, np.zeros_like(mu), best_value)
    np.testing.assert_allclose(ei, np.maximum(best_value - mu, 0.0), atol=1e-12)


@pytest.mark.unit
@given(predictions=_predictions())
def test_ei_of_point_does_not_depend_on_batch(
    predictions: tuple[np.ndarray, np.ndarray, float],
) -> None:
    """Test that a point gets the same EI alone as in a batch of points."""
    mu, var, best_value = predictions
    batch = _calc_ei(mu, var, best_value)
    for i in range(len(mu)):
        alone = _calc_ei(mu[i : i + 1], var[i : i + 1], best_value)
        np.testing.assert_allclose(alone, batch[i : i + 1], rtol=1e-12, atol=1e-12)
