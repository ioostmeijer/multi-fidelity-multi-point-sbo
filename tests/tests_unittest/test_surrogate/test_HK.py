import re

import numpy as np
import pytest
from smt.surrogate_models.krg import KRG as SMT_KRG

from sbo.surrogate import HK, KRG
from sbo.utils import Config, validate_settings


def base_config() -> Config:
    """Create a base configuration for testing.

    Returns
    -------
    Config object with default settings for HK surrogate model.
    """
    config = Config(
        used_algorithms={"surrogate": "HK"},
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 0.01 * x],
    )
    return config


def base_samples() -> list[dict]:
    """Create a base set of samples for testing.

    Returns
    -------
    List of sample dictionaries with coordinates, values, and fidelities.
    """
    samples = [
        {"coord": [0.0, 0.25], "value": 0.0, "fid": 0},
        {"coord": [0.5, 0.25], "value": 0.25, "fid": 0},
        {"coord": [1.0, 0.25], "value": 1.0, "fid": 0},
        {"coord": [0.0, 0.25], "value": 0.0, "fid": 1},
        {"coord": [0.5, 0.25], "value": 0.2525, "fid": 1},
        {"coord": [1.0, 0.25], "value": 1.01, "fid": 1},
    ]
    return samples


@pytest.mark.unit
def test_hk_krg_settings() -> None:
    """Test HK surrogate model settings for KRG surrogate.

    This test verifies that the HK surrogate model correctly sets its settings
    to match those of the KRG surrogate model when initialised.
    """
    settings = {"nugget": 1.99, "corr": "cubic_spline", "some random key": False}
    config = Config(
        used_algorithms={"surrogate": "HK"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 0.01 * x],
        settings={"surrogate": settings},
    )
    actual_settings = HK._get_krg_settings(config)

    expected_settings = {
        **validate_settings(used_algorithms={"surrogate": "KRG"})["surrogate"],
        "nugget": 1.99,
        "corr": "squar_exp",
    }

    assert expected_settings == actual_settings


@pytest.mark.unit
def test_hk_init_single_fid() -> None:
    """Test initialisation of HK surrogate with single fidelity function.

    This test verifies that when the HK surrogate model is initialised with a single
    fidelity function, it correctly defaults to using the KRG surrogate model.
    """
    config = Config(
        used_algorithms={"surrogate": "HK"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
    )
    surrogate = HK(config=config)
    expected_settings = validate_settings(used_algorithms={"surrogate": "KRG"})[
        "surrogate"
    ]
    expected_settings["corr"] = "squar_exp"

    assert isinstance(surrogate, KRG)
    assert config.settings["surrogate"] == expected_settings


@pytest.mark.unit
def test_hk_init() -> None:
    """Test initialisation of HK surrogate model."""
    surr = HK(config=base_config())
    assert isinstance(surr, HK)
    assert surr._trained is False
    assert surr.config == base_config()


@pytest.mark.unit
def test_hk_log_likelihood() -> None:
    """Test the log likelihood calculation of the HK surrogate model."""
    surr = HK(config=base_config())
    y_samples = np.array([0.1, 0.5, 0.3]).T
    f = np.array([0.2, 0.3, 0.8]).T
    n = 3
    fid = 0
    theta = np.array([1.0, 0.05])
    samples = {
        fid: [
            {"coord": [0.0, 0.0]},
            {"coord": [1.0, 0.0]},
            {"coord": [0.0, 1.0]},
        ]
    }
    x_samples = np.array([sample["coord"] for sample in samples[fid]])
    surr.samples = samples

    surr._model.setdefault(fid, {})
    surr._model[fid]["y_samples"] = y_samples
    surr._model[fid]["f"] = f
    surr._model[fid]["x"] = x_samples
    log_likelihood = surr._log_likelihood_hk(theta, n, fid)
    assert log_likelihood is not None, "log likelihood is None"
    log_likelihood = float(log_likelihood)

    # Expected value
    corr_matrix = surr._calculate_corr_matrix(x_samples, theta)
    corr_matrix_inv = np.linalg.inv(corr_matrix)
    beta_0 = (f.T @ corr_matrix_inv @ y_samples) / (f.T @ corr_matrix_inv @ f)
    residual = y_samples - beta_0 * f
    sigma_sq = (residual.T @ corr_matrix_inv @ residual) / n

    expected_log_likelihood = float(
        -n * np.log(sigma_sq) - np.log(np.linalg.det(corr_matrix))
    )

    if np.isfinite(expected_log_likelihood):
        assert np.isclose(log_likelihood, expected_log_likelihood)
    else:
        assert np.isinf(expected_log_likelihood)
        assert np.isinf(log_likelihood)
        assert np.sign(log_likelihood) == np.sign(expected_log_likelihood)


@pytest.mark.unit
def test_hk_train() -> None:
    """Test training HK surrogate with valid samples."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    expected_model_keys = [
        "y_samples",
        "f",
        "theta",
        "beta_0",
        "sigma_sq",
        "x",
        "chol",
        "ft",
        "q",
        "g",
        "gamma",
    ]

    assert surr._trained is True
    assert [0, 1] == sorted(surr._model.keys())
    assert isinstance(surr._model[1], SMT_KRG)
    assert all(key in surr._model[0] for key in expected_model_keys)


@pytest.mark.unit
def test_hk_train_no_samples() -> None:
    """Test training HK surrogate with no samples."""
    surr = HK(config=base_config())
    fidelities_with_insufficient_samples = [0, 1]
    msg = (
        "Not enough samples to train HK surrogate at fidelities: "
        f"{fidelities_with_insufficient_samples}. "
        "At least 2 samples are required per fidelity."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        surr.train()


@pytest.mark.unit
def test_hk_missing_samples() -> None:
    """Test training HK surrogate with insufficient samples at a fidelity."""
    surr = HK(config=base_config())
    samples = [
        {"coord": [0.0], "value": 1, "fid": 0},
        {"coord": [0.5], "value": 0.5, "fid": 0},
    ]
    surr.set_training_data(samples)
    fidelities_with_insufficient_samples = [1]
    msg = (
        "Not enough samples to train HK surrogate at fidelities: "
        f"{fidelities_with_insufficient_samples}. "
        "At least 2 samples are required per fidelity."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        surr.train()


@pytest.mark.unit
def test_hk_predict() -> None:
    """Test predicting with HK surrogate."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])
    y_pred_hf = surr.predict_values(x_test)
    y_pred_lf = surr.predict_values(x_test, fidelity=1)

    assert y_pred_hf.shape == (2, 1)
    assert y_pred_lf.shape == (2, 1)


@pytest.mark.unit
def test_hk_predict_fid() -> None:
    """Test predicting with HK surrogate at specific fidelity."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])
    y_pred = surr.predict_values(x_test, fidelity=1)

    assert y_pred.shape == (2, 1)


@pytest.mark.unit
def test_hk_predict_no_train() -> None:
    """Test predicting with HK surrogate without training."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())

    def mock_train() -> None:
        """Mock train method that raises an error."""
        raise (RuntimeError("Train called"))

    surr.train = mock_train

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])

    with pytest.raises(RuntimeError, match="Train called"):
        surr.predict_values(x_test)


@pytest.mark.unit
def test_hk_predict_invalid_fidelity() -> None:
    """Test predicting with HK surrogate using invalid fidelity."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])

    with pytest.raises(
        ValueError, match="Fidelity 2 is out of bounds for the HK model."
    ):
        _ = surr.predict_values(x_test, fidelity=2)


@pytest.mark.unit
def test_hk_predict_var() -> None:
    """Test predicting variance with HK surrogate."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])
    y_pred_hf = surr.predict_variance(x_test)
    y_pred_lf = surr.predict_variance(x_test, fidelity=1)

    assert y_pred_hf.shape == (2, 1)
    assert y_pred_lf.shape == (2, 1)


@pytest.mark.unit
def test_hk_predict_var_fid() -> None:
    """Test predicting variance with HK surrogate at specific fidelity."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])
    y_pred = surr.predict_variance(x_test, fidelity=1)

    assert y_pred.shape == (2, 1)


@pytest.mark.unit
def test_hk_predict_var_invalid_fidelity() -> None:
    """Test predicting variance with HK surrogate using invalid fidelity."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())
    surr.train()

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])

    with pytest.raises(
        ValueError, match="Fidelity 2 is out of bounds for the HK model."
    ):
        _ = surr.predict_variance(x_test, fidelity=2)


@pytest.mark.unit
def test_hk_predict_var_no_train() -> None:
    """Test predicting variance without training the HK surrogate model."""
    surr = HK(config=base_config())
    surr.set_training_data(base_samples())

    def mock_train() -> None:
        """Mock train method that raises an error."""
        raise (RuntimeError("Train called"))

    surr.train = mock_train

    x_test = np.array([[0.25, 0.25], [0.75, 0.25]])

    with pytest.raises(RuntimeError, match="Train called"):
        surr.predict_variance(x_test)


@pytest.mark.unit
def test_hk_log_likelihood_2() -> None:
    """Test the log likelihood calculation of the HK surrogate model.

    Test the log likelihood calculation of the HK surrogate model against a manually
    computed expected value.
    The value is calculated using theta = 0.4, n = 2, fid = 0.
    With F_{HF}(x)=2x + 5 and F_{LF}(x)=0.5 * F_{HF}(x) + 2 * x - 10.
    With samples at x = [0.5] and x = [2.5].
    This results in a log likelihood of -7.29255

    :return : None
    """
    expected_log_likelihood = -7.29255
    theta = np.array([0.4])

    training_data = [
        {"coord": [0.5], "value": 6, "fid": 0},
        {"coord": [2.5], "value": 10, "fid": 0},
        {"coord": [0.5], "value": -6.75, "fid": 1},
        {"coord": [2.5], "value": -1.25, "fid": 1},
    ]
    config = Config(
        used_algorithms={"surrogate": "HK"},
        lb=[0.0],
        ub=[3.0],
        functions=[lambda x: 2 * x + 5, lambda x: 0.5 * (2 * x + 5) + 2 * x - 10],
        settings={"surrogate": {"corr": "cubic_spline", "nugget": 0}},
    )
    hk_instance = HK(config=config)
    hk_instance.set_training_data(training_data)
    hk_instance.train()

    y_samples = np.array([6, 10]).T  # HF values at x = [0.5], [2.5]
    f = np.array([-6.75, -1.25]).T  # LF values at x = [0.5], [2.5]
    hk_instance._model.setdefault(0, {})
    hk_instance._model[0]["y_samples"] = y_samples
    hk_instance._model[0]["f"] = f
    hk_instance._model[0]["x"] = np.array([[0.5], [2.5]])

    calculated_log_likelihood = hk_instance._log_likelihood_hk(theta, n=2, fidelity=0)

    assert np.isclose(
        calculated_log_likelihood, expected_log_likelihood, rtol=1e-4, atol=1e-3
    ), (
        f"Calculated log likelihood {calculated_log_likelihood} does not match expected"
        f" {expected_log_likelihood}"
    )
