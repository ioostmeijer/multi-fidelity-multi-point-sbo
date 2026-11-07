import numpy as np
import pytest
from smt.applications import MFK as SMT_MFK
from smt.applications import NestedLHS

from sbo import Config
from sbo.surrogate import MFK


def mock_config() -> Config:
    """Create a mock configuration for testing.

    Returns
    -------
    Mock configuration for testing.
    """
    config = Config(
        used_algorithms={"surrogate": "MFK"},
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 1],
    )
    return config


@pytest.mark.unit
def test_init() -> None:
    """Test the initialisation of the MFK surrogate model."""
    config = mock_config()
    mfk_model = MFK(config=config)
    assert mfk_model.config == config
    assert not mfk_model._trained
    assert isinstance(mfk_model._model, SMT_MFK)


@pytest.mark.unit
def test_predict() -> None:
    """Test the predict method of the MFK surrogate model."""
    config = mock_config()
    mfk_model = MFK(config=config)

    lhs = NestedLHS(nlevel=2, xlimits=config.bounds)
    x_train_low, x_train_high = lhs(7)
    y_train_low = np.array([config.functions[0](x) for x in x_train_low])
    y_train_high = np.array([config.functions[1](x) for x in x_train_high])

    mfk_model.set_training_data(
        [
            {"coord": x_train_high[i], "value": y_train_high[i], "fid": 0}
            for i in range(len(x_train_high))
        ]
        + [
            {"coord": x_train_low[i], "value": y_train_low[i], "fid": 1}
            for i in range(len(x_train_low))
        ]
    )
    mfk_model.train()

    # generate smt model
    smt_model_settings = config.settings["surrogate"]
    smt_model_settings.pop("data_dir", None)
    smt_model_settings["seed"] = config.get("random_seed", None)
    smt_model = SMT_MFK(**smt_model_settings)
    smt_model.set_training_values(x_train_low, y_train_low, name=0)
    smt_model.set_training_values(x_train_high, y_train_high)
    smt_model.train()

    x_test = np.linspace(0.0, 1.0, 100)

    # test predict_values
    y_pred_mfk = mfk_model.predict_values(x_test)
    y_pred_smt = smt_model.predict_values(x_test)
    np.testing.assert_allclose(y_pred_mfk, y_pred_smt, rtol=1e-5)

    # test predict_variance
    y_var_mfk = mfk_model.predict_variance(x_test)
    y_var_smt = smt_model.predict_variances(x_test)
    np.testing.assert_allclose(y_var_mfk, y_var_smt, rtol=1e-5)

    # test predict
    y_pred_mfk_full, y_var_mfk_full = mfk_model.predict(x_test)
    np.testing.assert_allclose(y_pred_mfk_full, y_pred_smt, rtol=1e-5)
    np.testing.assert_allclose(y_var_mfk_full, y_var_smt, rtol=1e-5)


def _trained_mfk() -> MFK:
    """Train a 2-fidelity MFK on nested samples."""
    Config.destroy_instance()
    config = mock_config()
    mfk_model = MFK(config=config)
    lhs = NestedLHS(nlevel=2, xlimits=config.bounds, seed=0)
    x_train_low, x_train_high = lhs(7)
    mfk_model.set_training_data(
        [
            {"coord": x, "value": float(config.functions[0](x)[0]), "fid": 0}
            for x in x_train_high
        ]
        + [
            {"coord": x, "value": float(config.functions[1](x)[0]), "fid": 1}
            for x in x_train_low
        ]
    )
    mfk_model.train()
    return mfk_model


@pytest.mark.unit
def test_predict_per_fidelity() -> None:
    """predict_values/predict_variance with fidelity select the SMT level."""
    mfk_model = _trained_mfk()
    x = np.array([[0.1], [0.5], [0.9]])

    assert MFK.supports_fidelity_prediction

    # HF (fidelity 0) is the normal SMT prediction
    np.testing.assert_allclose(
        mfk_model.predict_values(x, fidelity=0), mfk_model._model.predict_values(x)
    )
    np.testing.assert_allclose(
        mfk_model.predict_variance(x, fidelity=0),
        mfk_model._model.predict_variances(x),
    )

    # LF (fidelity 1) is SMT level 1 (the lowest level)
    np.testing.assert_allclose(
        mfk_model.predict_values(x, fidelity=1),
        mfk_model._model._predict_intermediate_values(x, 1),
    )
    np.testing.assert_allclose(
        mfk_model.predict_variance(x, fidelity=1).ravel(),
        mfk_model._model.predict_variances_all_levels(x)[0][:, 0],
    )

    # The LF model is close to the LF function (x**2 + 1), not the HF one
    lf_pred = mfk_model.predict_values(x, fidelity=1).ravel()
    np.testing.assert_allclose(lf_pred, x.ravel() ** 2 + 1, atol=0.1)

    with pytest.raises(ValueError):
        mfk_model.predict_values(x, fidelity=2)
