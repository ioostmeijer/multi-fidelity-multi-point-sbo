import pathlib

import numpy as np
import pytest
from dummy_surrogate import DummySurrogate

from sbo import Config
from sbo.optimisation import EgoSequential


@pytest.mark.unit
def test_ego_sequential_initialisation() -> None:
    """Test the initialisation of the EgoSequential class."""
    config = Config(
        used_algorithms={"optimisation": "EgoSequential"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
    )

    ego_seq = EgoSequential(config)
    assert isinstance(ego_seq, EgoSequential)


@pytest.mark.unit
def test_ego_sequential_solve_method(tmp_path: pathlib.Path) -> None:
    """Test the solve method of the EgoSequential class."""
    save_path = tmp_path / "save_dir"
    config = Config(
        used_algorithms={"optimisation": "EgoSequential"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        settings={"save_folder": save_path},
    )

    ego_seq = EgoSequential(config=config)
    surrogate = DummySurrogate(
        config=config, mu=0.5, optimum_value={"coord": np.array([0.0]), "value": [0.0]}
    )
    results = ego_seq.solve(
        surrogate=surrogate,
        infill=lambda s: ([[{"coord": np.array([0.0]), "fid": 0}]], 0),
    )

    expected_results = {"coord": [0.0], "value": [0.0], "fid": 0}
    assert "end_time" in results
    assert "start_time" in results
    assert expected_results["coord"] == results["coord"]
    assert expected_results["value"] == results["value"]
    assert expected_results["fid"] == results["fid"]
