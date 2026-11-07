import numpy as np
import pytest
from smt.surrogate_models import RBF as SMT_RBF

from sbo import Config
from sbo.surrogate import RBF


def mock_config() -> Config:
    """Create a mock configuration for testing.

    Returns
    -------
    Mock configuration for testing.
    """
    config = Config(
        used_algorithms={"surrogate": "RBF"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
    )
    return config


@pytest.mark.unit
def test_init() -> None:
    """Test the initialisation of the RBF surrogate model."""
    config = mock_config()
    rbf_model = RBF(config=config)
    assert rbf_model.config == config
    assert not rbf_model._trained
    assert isinstance(rbf_model._model, SMT_RBF)


@pytest.mark.unit
def test_predict_values() -> None:
    """Test the predict_values method of the RBF surrogate model."""
    config = mock_config()
    rbf_model = RBF(config=config)

    x_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([0.0, 0.25, 1.0])

    rbf_model.set_training_data(
        [
            {"coord": x_train[i], "value": y_train[i], "fid": 0}
            for i in range(len(x_train))
        ]
    )
    rbf_model.train()

    # generate smt model
    smt_model_settings = config.settings["surrogate"]
    smt_model_settings.pop("data_dir", None)
    smt_model = SMT_RBF(**smt_model_settings)
    smt_model.set_training_values(x_train, y_train)
    smt_model.train()

    x_test = np.linspace(0.0, 1.0, 100)
    y_pred_rbf = rbf_model.predict_values(x_test)
    y_pred_smt = smt_model.predict_values(x_test)
    np.testing.assert_allclose(y_pred_rbf, y_pred_smt, rtol=1e-5)
