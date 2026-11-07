import pathlib

import numpy as np
import pytest

from sbo import Config
from sbo.surrogate import (
    SurrogateBase,
    SurrogateBaseWithVariance,
    SurrogateBaseWithVariancePerFidelity,
)


def mock_config() -> Config:
    """Create a mock Config object for testing.

    Returns
    -------
    A mock Config object.
    """
    return Config(
        used_algorithms={}, lb=[0.0], ub=[1.0], functions=[lambda x: x, lambda x: x**2]
    )


def _overwrite_initialise_model(*args, **kwargs) -> bool:
    """Mock initialise model method.

    Returns
    -------
    True, indicating successful model initialisation.
    """
    return True


@pytest.mark.unit
def test_raise_not_implemented() -> None:
    """Test that NotImplementedError is raised for abstract methods."""
    with pytest.raises(NotImplementedError):
        surrogate_base = SurrogateBase(config=mock_config())

    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(config=mock_config())
    surrogate_base_w_variance = SurrogateBaseWithVariance(config=mock_config())
    surrogate_base_w_variance_per_fidelity = SurrogateBaseWithVariancePerFidelity(
        config=mock_config()
    )

    with pytest.raises(NotImplementedError):
        surrogate_base.predict_values(np.array([[0.5]]))

    with pytest.raises(NotImplementedError):
        surrogate_base.train()

    with pytest.raises(NotImplementedError):
        surrogate_base_w_variance.predict_values(np.array([[0.5]]))

    with pytest.raises(NotImplementedError):
        surrogate_base_w_variance.predict_variance(np.array([[0.5]]))

    with pytest.raises(NotImplementedError):
        surrogate_base_w_variance_per_fidelity.predict_variance(np.array([[0.5]]))

    with pytest.raises(NotImplementedError):
        surrogate_base_w_variance_per_fidelity.predict_variance(
            np.array([[0.5]]), fidelity_level=0
        )


@pytest.mark.unit
def test_get_optimum() -> None:
    """Test get_optimum method."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(config=mock_config())
    surrogate_base._trained = True
    surrogate_base.predict_values = lambda x, fidelity: (x - 1) ** 2

    opt = surrogate_base.get_optimum()
    opt_coord = opt["coord"]
    opt_value = opt["value"]

    assert np.allclose(opt_coord, np.array([[1.0]]))
    assert np.isclose(opt_value, 0.0)


@pytest.mark.unit
def test_add_training_data() -> None:
    """Test add_training_data method."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(
        config=mock_config(), initial_samples=[{"coord": [0.0], "value": 0.0, "fid": 1}]
    )

    new_data = [{"coord": [1.0], "value": 1.0, "fid": 0}]
    expected_data = {
        0: [
            {"coord": [1.0], "value": 1.0, "fid": 0},
        ],
        1: [
            {"coord": [0.0], "value": 0.0, "fid": 1},
        ],
    }

    surrogate_base.add_training_data(new_data)
    assert surrogate_base.samples == expected_data
    assert surrogate_base._trained is False
    assert list(surrogate_base._optimum.values()) == [None, None]
    assert list(surrogate_base._optimum.keys()) == [0, 1]


@pytest.mark.unit
def test_set_training_data() -> None:
    """Test set_training_data method."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(
        config=mock_config(), initial_samples=[{"coord": [0.0], "value": 0.0, "fid": 1}]
    )
    surrogate_base._trained = True
    surrogate_base._optimum = {}

    training_data = [
        {"coord": [1.0], "value": 1.0, "fid": 0},
        {"coord": [2.0], "value": 4.0, "fid": 0},
    ]
    expected_data = {
        0: [
            {"coord": [1.0], "value": 1.0, "fid": 0},
            {"coord": [2.0], "value": 4.0, "fid": 0},
        ],
        1: [],
    }

    surrogate_base.set_training_data(training_data)

    assert surrogate_base.samples == expected_data
    assert list(surrogate_base._optimum.values()) == [None, None]
    assert list(surrogate_base._optimum.keys()) == [0, 1]
    assert surrogate_base._trained is False


@pytest.mark.unit
def test_save_model() -> None:
    """Test save_model method."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(config=mock_config())

    saved_model = surrogate_base.save_model()
    assert isinstance(saved_model, dict)
    assert ["model", "trained", "optimum", "samples"] == list(saved_model.keys())


@pytest.mark.unit
def test_load_model() -> None:
    """Test load_model method."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(config=mock_config())

    surrogate_base2 = SurrogateBase(config=mock_config())

    # Change some internal state to verify loading works
    surrogate_base2._model = "mock_model"
    surrogate_base2._trained = True

    saved_model = surrogate_base2.save_model()
    surrogate_base.load_model(saved_model)
    for attr in ["_model", "_trained", "_optimum", "samples"]:
        assert getattr(surrogate_base, attr) == getattr(surrogate_base2, attr)


@pytest.mark.unit
def test_save_model_from_file(tmp_path: pathlib.Path) -> None:
    """Test save_model_to_file method."""
    config = mock_config()
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(config=config)
    surrogate_base._trained = True
    surrogate_base._model = "mock_model"

    save_path = tmp_path / "model_save.pkl"
    save_path2 = tmp_path / "model_save2.txt"
    save_path3 = tmp_path / "model_save3"
    for path in [save_path, save_path2, save_path3]:
        surrogate_base.save_model_to_file(path)

    for filename in ["model_save.pkl", "model_save2.pkl", "model_save3.pkl"]:
        assert (tmp_path / filename).exists(), f"File {filename} was not created."


@pytest.mark.unit
def test_load_model_from_file(tmp_path: pathlib.Path) -> None:
    """Test load_model_from_file method."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate_base = SurrogateBase(config=mock_config())
    surrogate_base._trained = True
    surrogate_base._model = "mock_model"
    save_path = tmp_path / "model_save.pkl"
    surrogate_base.save_model_to_file(save_path)

    surrogate_base2 = SurrogateBase(config=mock_config())
    surrogate_base2.load_model_from_file(save_path)
    assert surrogate_base2._trained is True
    assert surrogate_base2._model == "mock_model"

    invalid_path = "non_existent_model.pkl"
    surrogate_base3 = SurrogateBase(config=mock_config())
    with pytest.raises(FileNotFoundError):
        surrogate_base3.load_model_from_file(invalid_path)


@pytest.mark.unit
def test_correct_predict_var_w_fid_called() -> None:
    """Test that the correct predict_variance method is called."""
    SurrogateBase._initialise_model = _overwrite_initialise_model

    surrogate = SurrogateBaseWithVariancePerFidelity(config=mock_config())
    surrogate._trained = True

    surrogate.predict_variance_fidelity = lambda x, fidelity: "Per fidelity called"

    result_per_fid = surrogate.predict_variance(np.array([[0.5]]), fidelity=1)
    assert result_per_fid == "Per fidelity called"


@pytest.mark.unit
@pytest.mark.parametrize(
    "lb, x, expected_shape",
    [
        ([0.0, 0.0], np.array([0.2, 0.3]), (1, 2)),  # single point as 1D array
        ([0.0, 0.0], np.array([[0.2, 0.3]]), (1, 2)),
        ([0.0, 0.0], np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]), (3, 2)),
        ([0.0, 0.0], np.array([[0.2, 0.3], [0.4, 0.5]]), (2, 2)),  # n == n_dim
        ([0.0], np.array([0.2, 0.3, 0.4]), (3, 1)),  # 1D problem: column of points
        ([0.0], np.array([0.2]), (1, 1)),
    ],
)
def test_correct_x_shape(lb: list, x: np.ndarray, expected_shape: tuple) -> None:
    """Test that _correct_x_shape returns (n_points, n_dimensions)."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    Config(used_algorithms={}, lb=lb, ub=[1.0] * len(lb), functions=[lambda x: x])
    surrogate = SurrogateBase(config=Config.get_instance())

    result = surrogate._correct_x_shape(x)

    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.ravel(), x.ravel())


@pytest.mark.unit
def test_samples_lists_are_not_shared() -> None:
    """Test that every fidelity starts with its own samples list."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate = SurrogateBase(config=mock_config())

    surrogate.samples[0].append({"coord": [0.1], "value": 0.1, "fid": 0})

    assert surrogate.samples[1] == []


@pytest.mark.unit
def test_add_training_data_does_not_mutate_input() -> None:
    """Test that add_training_data leaves the caller's points unchanged."""
    SurrogateBase._initialise_model = _overwrite_initialise_model
    surrogate = SurrogateBase(config=mock_config())
    point = {"coord": np.array([0.1]), "value": np.array([0.5]), "fid": 0}

    surrogate.add_training_data([point])

    assert isinstance(point["value"], np.ndarray)
    assert surrogate.samples[0][0]["value"] == 0.5
    assert surrogate.samples[0][0] is not point
