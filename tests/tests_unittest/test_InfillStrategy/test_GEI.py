import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from dummy_surrogate import DummySurrogateWithVariance  # noqa: F401
from logger_helper import _find_log_file, _make_logger  # noqa: F401
from scipy.stats import norm

from sbo import Config
from sbo.infill.GEI import GEI


def dummy_config() -> Config:
    """Create a dummy Config object for testing.

    Returns
    -------
    A dummy Config object with predefined settings for testing purposes.
    """
    return Config(
        used_algorithms={"infill": "GEI"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings=None,
        logger=_make_logger(),
    )


@pytest.mark.unit
def test_calc_infill_value_reduces_to_ei_for_g1() -> None:
    """Test for g=1, the standard EI value is returned."""
    gei = GEI(dummy_config())
    gei.config.g = 1

    surrogate = DummySurrogateWithVariance(
        config=dummy_config(),
        mu=0.0,
        var=1.0,
        optimum_value={"coord": np.array([0.0, 1.0]), "value": [1.0]},
    )
    x = np.array([0.3])

    best = 1.0
    mu, var = surrogate(x, return_variance=True)
    sigma = np.sqrt(var)
    improvement = best - mu
    z = improvement / sigma
    expected = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    val = GEI.calc_infill_value(gei, x, surrogate, best_value=best)
    assert pytest.approx(expected, rel=1e-9) == val


@pytest.mark.unit
def test_calc_infill_value_zero_variance_returns_zero() -> None:
    """Test if 0 is returned if variance is 0."""
    gei = object.__new__(GEI)
    gei.logger = _make_logger()
    gei.config = SimpleNamespace(
        logger=_make_logger(),
        bounds=[(0.0, 1.0)],
        random_seed=1,
        g=2,
    )

    surrogate = DummySurrogateWithVariance(
        config=dummy_config(),
        mu=0.2,
        var=0.0,
        optimum_value={"coord": np.array([0.0, 1.0]), "value": [1.0]},
    )
    x = np.array([0.0])

    assert GEI.calc_infill_value(gei, x, surrogate, best_value=1.0) == 0.0


@pytest.mark.unit
def test_calc_infill_value_requires_surrogate_with_variance() -> None:
    """Test that TypeError is raised if surrogate does not provide variance."""
    gei = object.__new__(GEI)
    gei.logger = _make_logger()
    gei.config = SimpleNamespace(
        logger=_make_logger(),
        bounds=[(0.0, 1.0)],
        random_seed=1,
        g=2,
    )

    with pytest.raises(TypeError):
        GEI.calc_infill_value(gei, np.array([0.0]), object(), best_value=1.0)


@pytest.mark.unit
def test_get_new_points_uses_differential_evolution_and_returns_task_and_value(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test correct return values from GEI.get_new_points.

    Monkeypatch differential_evolution to return a deterministic result and verify
    GEI.get_new_points returns the expected nested tasks and GEI value.
    """
    # Patch differential_evolution used inside GEI.get_new_points
    gei_mod = importlib.import_module("sbo.infill.Base")
    monkeypatch.setattr(
        gei_mod,
        "differential_evolution",
        lambda func, bounds, *args, **kwargs: SimpleNamespace(
            x=np.array([0.123]), fun=-0.3, success=True
        ),
        raising=True,
    )

    surrogate = DummySurrogateWithVariance(
        config=dummy_config(),
        mu=0.0,
        var=1.0,
        optimum_value={"coord": np.array([0.0, 1.0]), "value": [1.0]},
    )

    gei = GEI(config=dummy_config())

    # Overwrite _return_nested_samples to just return input for testing
    gei._return_nested_samples = lambda tasks: tasks

    tasks_nested, gei_value = GEI.get_new_points(gei, surrogate)

    # test return GEI value
    assert pytest.approx(0.3, rel=1e-12) == gei_value

    # Validate structure
    assert isinstance(tasks_nested, list)
    assert len(tasks_nested) == 1
    inner = tasks_nested[0]
    assert isinstance(inner, list)
    assert len(inner) == 1
    task = inner[0]
    assert "coord" in task and "fid" in task
    assert isinstance(task["coord"], np.ndarray)
    assert task["fid"] == 0


@pytest.mark.unit
def test_calc_infill_value_rejects_surrogate_without_variance() -> None:
    """calc_infill_value raises a TypeError for a surrogate without variance."""
    from dummy_surrogate import DummySurrogate

    from sbo import Config
    from sbo.infill import GEI

    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "GEI"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
    )
    gei = GEI(config=config)
    surrogate = DummySurrogate(config, mu=0.5)

    with pytest.raises(TypeError):
        gei.calc_infill_value(np.array([0.3]), surrogate, best_value=1.0)
