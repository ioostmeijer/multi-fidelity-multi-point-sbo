import logging
from typing import Generator

import pytest
from helpers.logger_helper import _make_logger

from sbo.DefaultSettings import (
    DEFAULT_SETTINGS,
    EGOSEQUENTIAL_DEFAULT_SETTINGS,
    KRG_DEFAULT_SETTINGS,
)
from sbo.utils import (
    validate_settings,
)
from sbo.utils.validate_settings import check_value


@pytest.mark.unit
def test_validate_settings_default() -> None:
    """Test the validate_settings function with an empty dictionary.

    Should return the default settings as defined in DEFAULT_SETTINGS.
    """
    settings = validate_settings({})
    assert settings == {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }, "Default settings should match the predefined constants."


@pytest.mark.unit
def test_validate_settings_with_none() -> None:
    """Test the validate_settings function with None.

    Should return the default settings as defined in DEFAULT_SETTINGS.
    """
    settings = validate_settings(None)
    assert settings == {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }, "Settings should be None, returning default settings."


@pytest.mark.unit
def test_validate_settings_with_new_value() -> None:
    """Test the validate_settings function with a new value.

    Should return the default settings with the new value for 'number_new_points'.
    """
    settings = validate_settings({"number_new_points": 5})
    assert (
        settings["number_new_points"] == 5
    ), "The number of new points should be updated to 5."
    assert (
        settings["nested"] == DEFAULT_SETTINGS["nested"]["default"]
    ), "Other settings should remain unchanged."


@pytest.mark.unit
def test_validate_settings_with_wrong_key(caplog: Generator) -> None:
    """Test the validate_settings function with a wrong key.

    Should ignore the wrong key and emit a logger warning.
    """
    logger = _make_logger()
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="test_logger")
    settings = validate_settings({"wrong_key": "value"}, logger=logger)
    assert {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    } == settings, "Settings should remain unchanged when an invalid key is provided."
    assert any(
        "wrong_key" in rec.getMessage() for rec in caplog.records
    ), "Expected a warning mentioning 'wrong_key'."


@pytest.mark.unit
def test_validate_settings_with_wrong_value(caplog: Generator) -> None:
    """Test the validate_settings function with a wrong value.

    Should emit a logger warning for the invalid numeric value.
    """
    logger = _make_logger()
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="test_logger")
    settings = validate_settings({"number_new_points": -1}, logger=logger)
    # Ensure a warning mentioning the key was emitted
    assert any(
        "number_new_points" in rec.getMessage() for rec in caplog.records
    ), "Expected a warning mentioning 'number_new_points'."
    assert (
        settings["number_new_points"] == DEFAULT_SETTINGS["number_new_points"]["min"]
    ), "Invalid value should revert to the default number of new points."
    assert (
        settings["nested"] == DEFAULT_SETTINGS["nested"]["default"]
    ), "Other settings should remain unchanged."


@pytest.mark.unit
def test_validate_setting_with_wrong_type(caplog: Generator) -> None:
    """Test the validate_settings function with a wrong type.

    Should emit a logger warning for the type mismatch.
    """
    logger = _make_logger()
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="test_logger")
    settings = validate_settings({"random_seed": "not_an_int"}, logger=logger)
    assert any(
        "random_seed" in rec.getMessage() for rec in caplog.records
    ), "Expected a warning mentioning 'random_seed'."
    assert (
        settings["random_seed"] == DEFAULT_SETTINGS["random_seed"]["default"]
    ), "Invalid type should revert to the default random seed."
    assert (
        settings["nested"] == DEFAULT_SETTINGS["nested"]["default"]
    ), "Other settings should remain unchanged."


@pytest.mark.unit
def test_validate_settings_with_used_algorithms() -> None:
    """Test the validate_settings function with used algorithms.

    Should return the default settings with the used algorithms' settings.
    """
    used_algorithms = {
        "optimisation": "EGOSEQUENTIAL",
    }
    settings = validate_settings(
        {"optimisation": {"convergence_threshold": 2e-6}},
        used_algorithms=used_algorithms,
    )
    resulting_settings = {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }
    resulting_settings["optimisation"] = {
        key: EGOSEQUENTIAL_DEFAULT_SETTINGS[key]["default"]
        for key in EGOSEQUENTIAL_DEFAULT_SETTINGS
    }
    resulting_settings["optimisation"]["convergence_threshold"] = 2e-6
    assert (
        settings == resulting_settings
    ), "Settings should include defaults and the used algorithms' settings."


@pytest.mark.unit
def test_validate_settings_with_used_algorithms_and_empty() -> None:
    """Test the validate_settings function with used algorithms and an empty dictionary.

    Should return the default settings with the used algorithms' settings.
    """
    used_algorithms = {
        "optimisation": "EGOSEQUENTIAL",
    }
    settings = validate_settings({}, used_algorithms=used_algorithms)
    resulting_settings = {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }
    resulting_settings["optimisation"] = {
        key: EGOSEQUENTIAL_DEFAULT_SETTINGS[key]["default"]
        for key in EGOSEQUENTIAL_DEFAULT_SETTINGS
    }
    assert (
        settings == resulting_settings
    ), "Settings should include defaults and the used algorithms' settings."


@pytest.mark.unit
def test_validate_settings_with_used_algorithms_and_invalid_key(
    caplog: Generator,
) -> None:
    """Test the validate_settings function with used algorithms and an invalid key.

    Should ignore the invalid key and emit a logger warning.
    """
    logger = _make_logger()
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="test_logger")
    used_algorithms = {
        "optimisation": "EGOSEQUENTIAL",
    }
    settings = validate_settings(
        {"optimisation": {"invalid_key": "value"}},
        used_algorithms=used_algorithms,
        logger=logger,
    )
    resulting_settings = {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }
    resulting_settings["optimisation"] = {
        key: EGOSEQUENTIAL_DEFAULT_SETTINGS[key]["default"]
        for key in EGOSEQUENTIAL_DEFAULT_SETTINGS
    }
    assert settings == resulting_settings, (
        "Settings should include defaults and the used algorithms' "
        "settings, ignoring invalid keys."
    )
    assert any(
        "invalid_key" in rec.getMessage() for rec in caplog.records
    ), "Expected a warning mentioning 'invalid_key'."


@pytest.mark.unit
def test_validate_settings_with_used_algorithms_and_invalid_value(
    caplog: Generator,
) -> None:
    """Test the validate_settings function with used algorithms and an invalid value.

    Should emit a logger warning for invalid types/values.
    """
    logger = _make_logger()
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="test_logger")
    used_algorithms = {
        "optimisation": "EGOSEQUENTIAL",
    }
    settings = validate_settings(
        {"optimisation": {"convergence_threshold": -1}},
        used_algorithms=used_algorithms,
        logger=logger,
    )
    resulting_settings = {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }
    resulting_settings["optimisation"] = {
        key: EGOSEQUENTIAL_DEFAULT_SETTINGS[key]["default"]
        for key in EGOSEQUENTIAL_DEFAULT_SETTINGS
    }
    # -1 is accepted as a float (-1.0) but is below the minimum, so it is clamped
    resulting_settings["optimisation"]["convergence_threshold"] = 0.0
    assert settings == resulting_settings, (
        "Settings should include defaults and the used algorithms' "
        "settings with the minimum value for a value below the minimum."
    )
    assert any(
        "convergence_threshold" in rec.getMessage() for rec in caplog.records
    ), "Expected a warning mentioning 'convergence_threshold'."


@pytest.mark.unit
def test_validate_settings_with_changed_allowed_value() -> None:
    """Test the validate_settings function with a changed value to a allowed value.

    Should return the default settings with the new value for a setting with a set of
    allowed values.
    """
    used_algorithms = {
        "surrogate": "KRG",
    }
    settings = validate_settings(
        {"surrogate": {"corr": "matern52"}}, used_algorithms=used_algorithms
    )
    resulting_settings = {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }
    resulting_settings["surrogate"] = {
        key: KRG_DEFAULT_SETTINGS[key]["default"] for key in KRG_DEFAULT_SETTINGS
    }
    resulting_settings["surrogate"]["corr"] = "matern52"
    resulting_settings["surrogate"]["seed"] = 42
    assert settings == resulting_settings, (
        "Settings should include defaults and the used algorithms' "
        "settings with the new value for 'corr'."
    )


@pytest.mark.unit
def test_validate_settings_with_changed_not_allowed_value(caplog: Generator) -> None:
    """Test the validate_settings function with a changed not allowed value.

    Should emit a logger warning and return the default value for the option.
    """
    logger = _make_logger()
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="test_logger")
    used_algorithms = {
        "surrogate": "KRG",
    }
    settings = validate_settings(
        {"surrogate": {"corr": "not_allowed"}},
        used_algorithms=used_algorithms,
        logger=logger,
    )
    resulting_settings = {
        key: DEFAULT_SETTINGS[key]["default"] for key in DEFAULT_SETTINGS
    }
    resulting_settings["surrogate"] = {
        key: KRG_DEFAULT_SETTINGS[key]["default"] for key in KRG_DEFAULT_SETTINGS
    }
    resulting_settings["surrogate"]["seed"] = 42
    assert settings == resulting_settings, (
        "Settings should include defaults and the used algorithms' "
        "settings with the default value for 'corr' since 'not_allowed' is not allowed."
    )
    assert any(
        rec.levelno == logging.WARNING for rec in caplog.records
    ), "Expected at least one warning to be logged."


@pytest.mark.unit
def test_validate_settings_w_nested_options() -> None:
    """Test the correct functioning with nested settings."""
    used_algorithms = {"infill": "TwoStep"}
    settings = validate_settings(
        settings={"infill": {"single_fid_infill": "GEI"}},
        used_algorithms=used_algorithms,
    )

    assert (
        settings["infill"]["single_fid_infill"] == "GEI"
    ), "The 'single_fid_infill' should be set to 'GEI' in the nested options."


@pytest.mark.unit
@pytest.mark.parametrize(
    "value_info, value, expected",
    [
        ({"type": float, "default": 1e-10, "min": 0.0}, 0, 0.0),  # int -> float
        ({"type": float | type(None), "default": 1e-6}, 2, 2.0),  # union with float
        ({"type": int, "default": 5}, 2.5, 5),  # float rejected for int
        ({"type": int, "default": 5}, True, 5),  # bool rejected for int
        ({"type": float, "default": 0.5}, False, 0.5),  # bool rejected for float
        ({"type": bool, "default": False}, True, True),
        ({"type": float | tuple, "default": (0.5, 1.0)}, [0.4, 0.9], (0.4, 0.9)),
        ({"type": int | float, "default": 1}, 3, 3),  # int kept when int allowed
    ],
)
def test_check_value_numeric_types(
    value_info: dict, value: object, expected: object
) -> None:
    """Test the int/float/bool/list handling of check_value."""
    result = check_value(value, value_info, "key", logger=_make_logger())

    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.unit
def test_strict_validation_raises() -> None:
    """With strict_validation, an invalid value or unknown key raises."""
    with pytest.raises(ValueError):
        validate_settings(
            {"strict_validation": True, "number_initial_samples": 1},
            logger=_make_logger(),
        )
    with pytest.raises(ValueError):
        validate_settings(
            {"strict_validation": True, "unknown_key": 1}, logger=_make_logger()
        )
    with pytest.raises(ValueError):
        validate_settings(
            {"strict_validation": True, "optimisation": {"max_iterations": 0}},
            used_algorithms={"optimisation": "EGOSEQUENTIAL"},
            logger=_make_logger(),
        )


@pytest.mark.unit
def test_strict_validation_default_false() -> None:
    """By default an invalid value only warns and is replaced."""
    settings = validate_settings({"number_initial_samples": 1}, logger=_make_logger())

    assert settings["strict_validation"] is False
    assert settings["number_initial_samples"] == 2


@pytest.mark.unit
def test_non_str_single_fid_infill_does_not_crash() -> None:
    """A non-str single_fid_infill is replaced by the default instead of crashing."""
    settings = validate_settings(
        {"infill": {"single_fid_infill": 3}},
        used_algorithms={"infill": "TwoStep"},
        logger=_make_logger(),
    )

    assert settings["infill"]["single_fid_infill"] == "EI"


@pytest.mark.unit
def test_missing_type_does_not_change_defaults() -> None:
    """A default setting without a type is not changed by validation."""
    defaults = {"a": {"default": 1}}

    validate_settings({"a": 2}, default_settings=defaults, logger=_make_logger())

    assert "type" not in defaults["a"]
