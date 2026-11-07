from unittest.mock import patch

import numpy as np
import pytest
from logger_helper import _find_log_file, _make_logger  # noqa: F401

from sbo.infill.Random import Random
from sbo.utils import Config


class DummyConfig(dict):
    """Config class for testing."""

    def __init__(
        self,
        number_new_points: int,
        lb: list[float],
        ub: list[float],
        dimension: int,
        num_fidelities: int,
        random_seed: int = 0,
    ) -> None:
        """Initialise DummyConfig."""
        super().__init__()
        self["number_new_points"] = number_new_points
        self["random_seed"] = random_seed
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.dimension = dimension
        self.num_fidelities = num_fidelities
        self.rng = np.random.default_rng(random_seed)


@pytest.mark.unit
def test_get_new_points_single_fidelity_structure_and_inf() -> None:
    """Test Random.get_new_points with single fidelity."""
    np.random.seed(1)
    infill = object.__new__(Random)
    cfg = DummyConfig(
        number_new_points=3, lb=[0.0, 0.0], ub=[1.0, 1.0], dimension=2, num_fidelities=1
    )
    infill.config = cfg
    infill.logger = _make_logger()
    infill._return_nested_samples = lambda tasks: tasks  # pass-through for inspection

    new_points, infty = Random.get_new_points(infill)

    assert infty == float("inf")
    assert isinstance(new_points, list)
    assert len(new_points) == 3
    for item in new_points:
        assert isinstance(item, list) and len(item) == 1
        task = item[0]
        assert "coord" in task and "fid" in task
        assert isinstance(task["coord"], np.ndarray)
        assert task["coord"].shape == (cfg.dimension,)
        assert task["fid"] == 0
        assert np.all(task["coord"] >= cfg.lb)
        assert np.all(task["coord"] <= cfg.ub)


@pytest.mark.unit
def test_get_new_points_multi_fidelity_fid_range() -> None:
    """Test Random.get_new_points with multiple fidelities."""
    np.random.seed(2)
    infill = object.__new__(Random)
    cfg = DummyConfig(
        number_new_points=5, lb=[-1.0], ub=[2.0], dimension=1, num_fidelities=4
    )
    infill.config = cfg
    infill.logger = _make_logger()
    infill._return_nested_samples = lambda tasks: tasks

    new_points, infty = Random.get_new_points(infill)

    assert infty == float("inf")
    assert len(new_points) == 5
    fids = [inner[0]["fid"] for inner in new_points]
    assert all(isinstance(f, (int, np.integer)) for f in fids)
    assert all(0 <= int(f) < cfg.num_fidelities for f in fids)


@pytest.mark.unit
def test_all_fidelities_included() -> None:
    """Test that all fidelities are included in the generated points."""
    np.random.seed(42)
    cfg = DummyConfig(
        number_new_points=100,
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        dimension=2,
        num_fidelities=3,
    )
    infill = object.__new__(Random)
    infill.config = cfg
    infill.logger = _make_logger()

    new_samples, _ = infill.get_new_points()

    # Assert 100 points
    assert len(new_samples) == 100

    # Assert all fidelities are included in the samples.
    # n_fidelities = 3, therefore fidelities, 0, 1 and 2 should be included
    sampled_fidelities = {
        sample_point["fid"] for task in new_samples for sample_point in task
    }
    assert sampled_fidelities == {np.int32(0), np.int32(1), np.int32(2)}


def _random_config(**kwargs) -> Config:
    """Create a single-fidelity config that uses the Random infill strategy."""
    return Config(
        used_algorithms={
            "infill": "Random",
            "optimisation": "EgoSequential",
            "surrogate": "KRG",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        **kwargs,
    )


@pytest.mark.unit
def test_init_no_constraint_warning_without_constraints() -> None:
    """Test that Random does not warn about constraints when there are none."""
    config = _random_config()
    with patch.object(config.logger, "warning") as mock_warning:
        Random(config)

    assert not any(
        "Constraints" in str(call.args[0]) for call in mock_warning.call_args_list
    )


@pytest.mark.unit
def test_init_warns_with_constraints() -> None:
    """Test that Random warns that inexpensive constraints are ignored."""

    def g(x: np.ndarray) -> float:
        return float(x[0]) - 0.5

    config = _random_config(inexpensive_constraints={"inequality": [g]})
    with patch.object(config.logger, "warning") as mock_warning:
        Random(config)

    assert any(
        "will thus be ignored" in str(call.args[0])
        for call in mock_warning.call_args_list
    )
