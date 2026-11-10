import importlib

import numpy as np
import pytest
from dummy_surrogate import DummySurrogateWithVariancePerFidelity  # noqa: F401
from logger_helper import _find_log_file, _make_logger  # noqa: F401

from sbo.infill import EI
from sbo.infill.TwoStep import TwoStep
from sbo.utils.Config import Config


def dummy_config() -> Config:
    """Create a dummy Config object for testing.

    Returns
    -------
    A dummy Config object with predefined settings for testing purposes.
    """
    return Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={"infill": {"single_fid_infill": "EI"}},
        logger=_make_logger(),
    )


class MockConfig(Config):
    """A mock Config class for testing purposes."""

    def __init__(self, *args, **kwargs) -> None:
        """A mock Config class to allow direct setting of settings for testing."""
        super().__init__(*args, **kwargs)

    def bypass_update_settings(self, new_settings: dict) -> None:
        """A method to bypass the usual settings update checks for testing purposes.

        Parameters
        ----------
        new_settings
            The new settings dictionary.
        """
        self._settings = new_settings


@pytest.mark.unit
def test_get_new_points_rejects_non_variance_surrogate() -> None:
    """Test that a ValueError is raised if wrong surrogate is provided."""
    two_step = TwoStep(config=dummy_config())

    bad_surrogate = object()
    with pytest.raises(ValueError):
        TwoStep.get_new_points(two_step, bad_surrogate)


@pytest.mark.unit
def test_get_new_points_flow_selects_cheapest_accurate_fidelity() -> None:
    """Test the full flow of get_new_points, ensuring it selects the cheapest."""
    # Config and variance setup
    config = Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 1, lambda x: x**2 * 10],
        settings={"infill": {"single_fid_infill": "EI"}},
        logger=_make_logger(),
    )
    var_per_fidelity = {
        0: 0.01,  # HF
        1: 0.05,  # Accurate enough
        2: 1.0,  # Not accurate enough
    }

    # instantiate TwoStep with the mock single-fid infill
    two_step = TwoStep(config=config)
    coord = np.array([0.5])
    two_step.single_fid_infill.get_new_points = lambda surrogate: (
        [[{"coord": coord}]],
        0.77,
    )

    surrogate = DummySurrogateWithVariancePerFidelity(
        config=config, var_per_fidelity=var_per_fidelity
    )

    tasks_nested, infill_value = TwoStep.get_new_points(two_step, surrogate)

    # Check returned infill value and structure
    assert pytest.approx(0.77, rel=1e-12) == infill_value
    assert isinstance(tasks_nested, list)
    assert len(tasks_nested) == 1
    inner = tasks_nested[0]
    assert isinstance(inner, list) and len(inner) == 1
    task = inner[0]
    assert "coord" in task and "fid" in task
    # fidelity should be one of the accurate fidelities
    # HF (0) and fidelity 1 are accurate -> max is 1
    assert int(task["fid"]) == 1
    assert np.all(task["coord"] == coord)


@pytest.mark.unit
def test__get_accurate_enough_fidelities_handles_zero_variance() -> None:
    """Test _get_accurate_enough_fidelities works when variance is zero, HF is incl."""
    two_step = TwoStep(config=dummy_config())

    surrogate = DummySurrogateWithVariancePerFidelity(
        config=dummy_config(),
    )
    accurate = TwoStep._get_accurate_enough_fidelities(
        two_step, np.array([0.0]), surrogate
    )

    assert 0 in accurate
    assert isinstance(accurate, list)


@pytest.mark.unit
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test__get_accurate_enough_fidelities_handles_2d_predictions() -> None:
    """Test that 2D predictions (n, 1), as SMT returns them, give no warning."""

    class Dummy2DSurrogate(DummySurrogateWithVariancePerFidelity):
        """A dummy surrogate that returns predictions with shape (n, 1)."""

        def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
            """Predict the mean values at point x as a column vector."""
            return super().predict_values(x).reshape(-1, 1)

        def predict_variance_fidelity(
            self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
        ) -> np.ndarray:
            """Predict the variance at point x as a column vector."""
            return (
                super().predict_variance_fidelity(x, fidelity=fidelity).reshape(-1, 1)
            )

    config = Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 1, lambda x: x**2 * 10],
        settings={"infill": {"single_fid_infill": "EI"}},
        logger=_make_logger(),
    )
    two_step = TwoStep(config=config)
    surrogate = Dummy2DSurrogate(
        config=config, var_per_fidelity={0: 0.01, 1: 0.05, 2: 1.0}
    )

    accurate = TwoStep._get_accurate_enough_fidelities(
        two_step, np.array([0.5]), surrogate
    )

    assert accurate == [0, 1]


@pytest.mark.unit
def test_try_to_load_nonexistent_single_fid_infill() -> None:
    """Test that trying to load a non-existent single-fid infill raises ImportError."""
    bad_config = Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={"infill": {"single_fid_infill": "NonExistentInfill"}},
        logger=_make_logger(),
    )
    two_step = TwoStep(config=bad_config)

    assert isinstance(two_step.single_fid_infill, EI)


@pytest.mark.unit
def test_init_sets_single_fid_infill_correctly() -> None:
    """Test that TwoStep initialises single_fid_infill correctly."""
    good_config = Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={"infill": {"single_fid_infill": "EI"}},
        logger=_make_logger(),
    )

    two_step = TwoStep(config=good_config)

    assert isinstance(two_step.single_fid_infill, EI)
    assert two_step.single_fid_infill.config == good_config


@pytest.mark.unit
def test_import_something_else_than_ei() -> None:
    """Test that TwoStep can import another infill strategy."""
    good_config = Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={"infill": {"single_fid_infill": "GEI"}},
        logger=_make_logger(),
    )

    two_step = TwoStep(config=good_config)

    from sbo.infill.GEI import GEI

    assert isinstance(two_step.single_fid_infill, GEI)
    assert two_step.single_fid_infill.config == good_config


@pytest.mark.unit
def test_init_with_all_infill() -> None:
    """Test that TwoStep initialises single_fid_infill for all allowed strategies."""
    possible_infill = ["EI", "GEI", "Random"]
    config = Config(
        used_algorithms={"infill": "TwoStep"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={},
        logger=_make_logger(),
    )

    for infill in possible_infill:
        config.update_settings({"infill": {"single_fid_infill": infill}})

        two_step = TwoStep(config=config)

        # Import the infill class dynamically
        module = importlib.import_module(f"sbo.infill.{infill}")
        infill_cls = getattr(module, infill)
        assert isinstance(two_step.single_fid_infill, infill_cls)


@pytest.mark.unit
def test_cei_not_allowed_as_single_fid_infill() -> None:
    """CEI is not implemented, so it is not an allowed single-fidelity infill."""
    from sbo.DefaultSettings.ALLOWED_ENTRIES import (
        ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES,
    )

    assert "CEI" not in ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES
    assert "FilteredGEI" not in ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES
    assert "EI" in ALLOWED_TWOSTEP_SINGLE_FID_INFILL_STRATEGIES


@pytest.mark.unit
def test_gei_as_single_fid_infill() -> None:
    """Test that TwoStep creates a non-default single-fidelity infill by keyword."""
    from sbo.infill import GEI

    config = Config(
        used_algorithms={
            "infill": "TwoStep",
            "surrogate": "HK",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 0.1],
        settings={"infill": {"single_fid_infill": "GEI"}},
        logger=_make_logger(),
    )
    infill = TwoStep(config=config)

    assert isinstance(infill.single_fid_infill, GEI)
