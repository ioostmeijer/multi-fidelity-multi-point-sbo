import numpy as np
import pytest
from dummy_surrogate import DummySurrogateWithVariance  # noqa: F401
from logger_helper import _find_log_file, _make_logger  # noqa: F401
from scipy.stats import norm

from sbo import Config
from sbo.infill.EI import EI

dummy_config = Config(
    used_algorithms={"infill": "EI"},
    lb=[0.0],
    ub=[1.0],
    functions=[lambda x: x**2],
    settings=None,
    # Created at import, before the test runs in its temporary directory: use the
    # test logger, so no log file is written to the current directory
    logger=_make_logger(),
)


@pytest.mark.unit
def test_calc_infill_value_basic() -> None:
    """Test basic Expected Improvement calculation."""
    ei = EI.__new__(EI)
    ei.config = type("Cfg", (), {"logger": _make_logger(), "num_fidelities": 1})()

    surrogate = DummySurrogateWithVariance(dummy_config, mu=0.0, var=1.0)
    surrogate.add_training_data([{**surrogate.get_optimum(), "fid": 0}])
    x = np.array([0.3])

    # expected EI calculation
    best = 1.0
    mu, var = surrogate(x, return_variance=True)
    sigma = np.sqrt(var)
    improvement = best - mu
    z = improvement / sigma
    expected = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    val = ei.calc_infill_value(x, surrogate, best_value=None)
    assert pytest.approx(expected, rel=1e-9) == val


@pytest.mark.unit
def test_calc_infill_value_zero_variance_returns_zero() -> None:
    """Test that zero variance from surrogate returns EI of 0.0."""
    ei = EI.__new__(EI)
    ei.config = type("Cfg", (), {"logger": _make_logger(), "num_fidelities": 1})()

    surrogate = DummySurrogateWithVariance(dummy_config, mu=0.2, var=0.0)
    x = np.array([0.0])

    assert ei.calc_infill_value(x, surrogate, best_value=1.0) == 0.0


@pytest.mark.unit
def test_calc_infill_value_requires_surrogate_with_variance() -> None:
    """Test wrong surrogate type raises TypeError."""
    ei = EI.__new__(EI)
    ei.config = type("Cfg", (), {"logger": _make_logger()})()

    bad_surrogate = object()
    with pytest.raises(TypeError):
        ei.calc_infill_value(np.array([0.0]), bad_surrogate, best_value=1.0)


@pytest.mark.unit
def test_get_new_points_rejects_non_variance_surrogate() -> None:
    """Test that get_new_points raises TypeError for surrogate without variance."""
    ei = EI.__new__(EI)
    ei.config = type("Cfg", (), {"logger": _make_logger()})()

    with pytest.raises(TypeError):
        ei.get_new_points(object())


@pytest.mark.unit
def test_calc_infill_value_mixed_zero_variance_no_nan() -> None:
    """Test that points without variance get max(improvement, 0), without nan."""
    import warnings

    class _MixedVariance(DummySurrogateWithVariance):
        def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
            return np.array([[0.5], [0.5], [1.0]])

        def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
            return np.array([[0.0], [1.0], [0.0]])

    ei = EI.__new__(EI)
    ei.config = type("Cfg", (), {"logger": _make_logger(), "num_fidelities": 1})()
    surrogate = _MixedVariance(dummy_config, mu=0.0, var=1.0)
    surrogate.add_training_data([{**surrogate.get_optimum(), "fid": 0}])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        val = ei.calc_infill_value(np.zeros((3, 1)), surrogate, best_value=1.0)

    expected_mid = 0.5 * norm.cdf(0.5) + norm.pdf(0.5)
    assert not np.any(np.isnan(val))
    np.testing.assert_allclose(val, [0.5, expected_mid, 0.0])
