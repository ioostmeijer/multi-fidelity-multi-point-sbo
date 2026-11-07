import numpy as np
import pytest
from smt.surrogate_models import KRG as SMT_KRG

from sbo import Config
from sbo.surrogate import KRG


def mock_config() -> Config:
    """Create a mock configuration for testing.

    Returns
    -------
    Mock configuration for testing.
    """
    config = Config(
        used_algorithms={"surrogate": "KRG"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
    )
    return config


@pytest.mark.unit
def test_init() -> None:
    """Test the initialisation of the KRG surrogate model."""
    config = mock_config()
    krg_model = KRG(config=config)
    assert krg_model.config == config
    assert not krg_model._trained
    assert isinstance(krg_model._model, SMT_KRG)


@pytest.mark.unit
def test_predict() -> None:
    """Test the predict method of the KRG surrogate model."""
    config = mock_config()
    krg_model = KRG(config=config)

    x_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([0.0, 0.25, 1.0])

    krg_model.set_training_data(
        [
            {"coord": x_train[i], "value": y_train[i], "fid": 0}
            for i in range(len(x_train))
        ]
    )
    krg_model.train()

    # generate smt model
    smt_model_settings = config.settings["surrogate"]
    smt_model_settings.pop("data_dir", None)
    smt_model_settings["seed"] = config.get("random_seed", None)
    smt_model = SMT_KRG(**smt_model_settings)
    smt_model.set_training_values(x_train, y_train)
    smt_model.train()

    x_test = np.linspace(0.0, 1.0, 100)

    # test predict_values
    y_pred_krg = krg_model.predict_values(x_test)
    y_pred_smt = smt_model.predict_values(x_test)
    np.testing.assert_allclose(y_pred_krg, y_pred_smt, rtol=1e-5)

    # test predict_variances
    y_var_krg = krg_model.predict_variance(x_test)
    y_var_smt = smt_model.predict_variances(x_test)
    np.testing.assert_allclose(y_var_krg, y_var_smt, rtol=1e-5)

    # Test predict
    y_pred_krg_full, y_var_krg_full = krg_model.predict(x_test)
    np.testing.assert_allclose(y_pred_krg_full, y_pred_smt, rtol=1e-5)
    np.testing.assert_allclose(y_var_krg_full, y_var_smt, rtol=1e-5)


@pytest.mark.unit
def test_train_without_hf_samples_warns() -> None:
    """Test that training without HF samples logs a warning instead of crashing."""
    from unittest.mock import patch

    config = mock_config()
    krg = KRG(config=config)
    krg.samples = {0: []}

    with patch.object(config.logger, "warning") as mock_warning:
        krg.train()

    assert mock_warning.called
    assert not krg._trained
