from types import SimpleNamespace
from types import SimpleNamespace as NS

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from logger_helper import _find_log_file, _make_logger  # noqa: F401

from sbo.infill.FilteredGEI import FilteredGEI
from sbo.infill.GEI import GEI
from sbo.surrogate import CoKRG
from sbo.utils.Config import Config


class DummyCoKRG(CoKRG):
    """A dummy CoKRG surrogate model for testing purposes."""

    def __init__(self, num_fidelities: int = 2) -> None:
        """Initialise the DummyCoKRG with necessary attributes."""
        self.config = SimpleNamespace(num_fidelities=num_fidelities)
        self.samples = {0: [{"value": 1.0}, {"value": 1.2}]}

    def get_optimum(self) -> dict:
        """Get the optimum value and coordinates.

        Returns
        -------
        A fixed optimum for testing purposes.
        """
        return {"value": 1.0, "coord": np.array([0.0])}

    def predict(
        self, x: np.ndarray, *args, return_variance: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict the mean and optionally standard deviation at point x.

        Parameters
        ----------
        x
            The input point for prediction.
        return_variance
            If True, return both mean and variance; otherwise, return only mean.

        Returns
        -------
        The mean and optionally variance at point x.
        """
        if not return_variance:
            return np.array([0.05])
        return np.array([0.05]), np.array([0.2])

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the variance at point x.

        Parameters
        ----------
        x
            The input point for prediction.

        Returns
        -------
        The variance at point x.
        """
        return np.array([0.2])

    def predict_variance_fidelity(self, x: np.ndarray, fidelity: int) -> np.ndarray:
        """Predict the variance at point x for a given fidelity level.

        Parameters
        ----------
        x
            The input point for prediction.
        fidelity
            The fidelity level for which to predict variance.

        Returns
        -------
        The variance at point x for the specified fidelity level.
        """
        return np.array([0.05])


def dummy_config() -> Config:
    """Create a dummy Config object for testing.

    Returns
    -------
    A dummy Config object with predefined settings for testing purposes.
    """
    return Config(
        used_algorithms={"infill": "FilteredGEI", "surrogate": "CoKRG"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={"infill": {"single_fid_infill": "EI"}},
        logger=_make_logger(),
    )


@pytest.mark.unit
def test_new_returns_gei_when_one_fidelity() -> None:
    """Test that FilteredGEI.__new__ returns GEI instance when num_fidelities is 1."""
    config = dummy_config()
    inst = FilteredGEI.__new__(FilteredGEI, config=config)
    assert isinstance(inst, GEI)
    assert not isinstance(inst, FilteredGEI)


@pytest.mark.unit
def test_new_requires_config_raises_value_error() -> None:
    """Test that FilteredGEI.__new__ raises ValueError when no config is provided."""
    with pytest.raises(ValueError):
        FilteredGEI.__new__(FilteredGEI)  # no config provided


@pytest.mark.unit
def test_get_new_points_rejects_non_cokrg() -> None:
    """Test that get_new_points raises TypeError when surrogate is not CoKRG."""
    f_gei = object.__new__(FilteredGEI)
    f_gei.config = dummy_config()
    with pytest.raises(TypeError):
        FilteredGEI.get_new_points(f_gei, object())


@pytest.mark.unit
def test_get_new_points_rejects_too_many_fidelities(monkeypatch: MonkeyPatch) -> None:
    """Test ValueError is raised when surrogate has more than 2 fidelities."""
    monkeypatch.setattr("sbo.infill.FilteredGEI.CoKRG", DummyCoKRG, raising=False)

    f_gei = object.__new__(FilteredGEI)
    f_gei.config = dummy_config()

    bad_surrogate = DummyCoKRG(num_fidelities=3)
    with pytest.raises(ValueError):
        FilteredGEI.get_new_points(f_gei, bad_surrogate)


@pytest.mark.unit
def test_get_new_points_flow_and_fidelity_selection(monkeypatch: MonkeyPatch) -> None:
    """Test the flow of get_new_points and fidelity selection logic."""
    # Monkeypatch CoKRG to use our DummyCoKRG
    monkeypatch.setattr("sbo.infill.FilteredGEI.CoKRG", DummyCoKRG, raising=False)

    # Monkeypatch differential_evolution (used by the base class) for consistent output
    monkeypatch.setattr(
        "sbo.infill.Base.differential_evolution",
        lambda func, bounds, **kwargs: NS(x=np.array([0.123]), fun=-0.42, success=True),
    )

    # Create an instance and set necessary attributes/methods used by get_new_points
    f_gei = object.__new__(FilteredGEI)
    f_gei.config = dummy_config()

    # # Provide deterministic calc_infill_value used by differential_evolution objective
    f_gei.calc_infill_value = lambda x, surrogate, best_value=None: 0.42

    # Overwrite _return_nested_samples to return the tasks for testing
    f_gei._return_nested_samples = lambda tasks: tasks

    surrogate = DummyCoKRG()

    tasks_nested, gei_value = FilteredGEI.get_new_points(f_gei, surrogate)

    # Test returned GEI value
    assert pytest.approx(0.42, rel=1e-12) == gei_value

    # Test structure of the returned tasks
    assert isinstance(tasks_nested, list)
    assert len(tasks_nested) == 1
    inner = tasks_nested[0]
    assert isinstance(inner, list)
    assert len(inner) == 1
    task = inner[0]
    assert "coord" in task and "fid" in task
    assert isinstance(task["coord"], np.ndarray)
    assert task["fid"] in (0, 1)


@pytest.mark.unit
def test_init_multi_fidelity_not_functional() -> None:
    """Test that FilteredGEI for more than one fidelity raises NotImplementedError."""
    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "FilteredGEI", "surrogate": "HK"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 0.1],
        logger=_make_logger(),
    )
    with pytest.raises(NotImplementedError):
        FilteredGEI(config=config)
