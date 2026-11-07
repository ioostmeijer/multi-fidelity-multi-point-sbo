import logging
from typing import Generator

import numpy as np
import pytest

from sbo.infill import InfillBase
from sbo.surrogate import SurrogateBase
from sbo.utils import Config


@pytest.fixture(autouse=True)
def basic_config() -> Config:
    """Fixture to provide a basic Config instance for testing.

    Returns
    -------
    Config
        A basic Config object with default settings for testing InfillBase.
    """
    return Config(
        used_algorithms={},
        lb=[0.0],
        ub=[5.0],
        functions=[lambda x: x**1, lambda x: x**2, lambda x: x**3, lambda x: x**4],
        settings=None,
    )


@pytest.mark.unit
def test_get_new_points_raises_and_logs(
    basic_config: Config, caplog: Generator
) -> None:
    """Test that calling get_new_points on InfillBase raises NotImplementedError."""
    caplog.set_level(logging.ERROR)

    infill = InfillBase(config=basic_config)
    infill.config.logger.propagate = True

    with pytest.raises(NotImplementedError):
        infill.get_new_points(None)

    # ensure an error message was emitted
    assert any(
        "get_new_points method not implemented" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.unit
def test_call_delegates_to_overridden_get_new_points(basic_config: Config) -> None:
    """Test that __call__ delegates to an overridden get_new_points method."""
    called = {"flag": False}
    expected = ("ok", 1.23)

    class Impl(InfillBase):
        def get_new_points(self, surrogate: SurrogateBase) -> tuple:
            called["flag"] = True
            return expected

    impl = Impl(config=basic_config)
    result = impl(surrogate=None)
    assert called["flag"] is True
    assert result == expected


@pytest.mark.unit
def test_return_nested_samples_when_not_nested_returns_original(
    basic_config: Config,
) -> None:
    """Test _return_nested_samples returns original samples when nested is False."""
    infill = InfillBase(config=basic_config)

    p = np.array([0.0, 1.0])
    samples = [(p, 0), (p, 2)]
    out = infill._return_nested_samples(samples.copy())
    assert out == samples


@pytest.mark.unit
def test_return_nested_samples_when_nested_extends_with_lower_fidelities(
    basic_config: Config,
) -> None:
    """Test _return_nested_samples extends samples when nested is True."""
    config = basic_config
    config["nested"] = True
    infill = InfillBase(config=config)

    p = np.array([0.0])
    samples = [{"coord": p, "fid": 1}]  # fidelity 1 -> should add fidelity 2 and 3
    out = infill._return_nested_samples(samples.copy())
    # original sample remains and additional fidelities appended
    assert {"coord": p, "fid": 1} in out
    assert {"coord": p, "fid": 2} in out
    assert {"coord": p, "fid": 3} in out
    # final length should be original + number of added fidelities (2)
    assert len(out) == 3


@pytest.mark.unit
def test_fallback_search_in_bounds_for_2d() -> None:
    """The fallback search (DE not successful) samples the whole 2D box."""
    from unittest.mock import patch

    from sbo.utils import Config

    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "EI"},
        lb=[-5.0, 10.0],
        ub=[5.0, 20.0],
        functions=[lambda x: x],
    )
    infill = InfillBase(config=config)

    class _Failed:
        success = False
        x = None
        fun = 0.0

    evaluated = []

    def optim_func(x: np.ndarray) -> float:
        evaluated.append(np.asarray(x))
        return float(np.sum(x))

    with patch("sbo.infill.Base.differential_evolution", return_value=_Failed()):
        x, _ = infill._select_sample_point_optimisation(optim_func)

    pts = np.array(evaluated)
    assert np.all(pts >= [-5.0, 10.0]) and np.all(pts <= [5.0, 20.0])
    # Not only the diagonal: the two coordinates are not perfectly correlated
    assert abs(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1]) < 0.5
    assert np.all(x >= [-5.0, 10.0]) and np.all(x <= [5.0, 20.0])


@pytest.mark.unit
def test_unsuccessful_de_point_is_used() -> None:
    """A valid DE point is used even when DE does not report convergence."""
    from unittest.mock import patch

    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "EI"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x],
    )
    infill = InfillBase(config=config)

    class _NotConverged:
        success = False
        x = np.array([0.42])
        fun = -1.5
        message = "Maximum number of iterations has been exceeded."

    with patch("sbo.infill.Base.differential_evolution", return_value=_NotConverged()):
        x, value = infill._select_sample_point_optimisation(lambda x: 0.0)

    np.testing.assert_array_equal(x, [0.42])
    assert value == -1.5


@pytest.mark.unit
def test_differential_evolution_settings_from_config() -> None:
    """The infill_opt_* settings are passed to differential evolution."""
    from unittest.mock import patch

    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "EI"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x],
        settings={
            "infill_opt_max_iterations": 7,
            "infill_opt_population_size": 5,
            "infill_opt_tolerance": 0.01,
            "random_seed": 3,
        },
    )
    infill = InfillBase(config=config)

    class _Converged:
        success = True
        x = np.array([0.1])
        fun = 0.0

    with patch(
        "sbo.infill.Base.differential_evolution", return_value=_Converged()
    ) as mock_de:
        infill._select_sample_point_optimisation(lambda x: 0.0)

    kwargs = mock_de.call_args.kwargs
    assert kwargs["maxiter"] == 7
    assert kwargs["popsize"] == 5
    assert kwargs["tol"] == 0.01
    assert kwargs["seed"] == 3
    assert kwargs["mutation"] == (0.5, 1.0)
    assert kwargs["strategy"] == "best1bin"
