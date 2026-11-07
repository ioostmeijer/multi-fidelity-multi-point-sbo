import copy
import logging
import types

from .. import DefaultSettings
from .Logger import get_logger


def _handle_invalid(message: str, logger: logging.Logger, strict: bool) -> None:
    """Raise (strict mode) or log a warning for an invalid setting.

    Parameters
    ----------
    message
        The message describing the invalid setting.
    logger
        Logger instance to use for logging warnings.
    strict
        If True, raise a ValueError instead of logging a warning.

    Raises
    ------
    ValueError
        If strict is True.
    """
    if strict:
        logger.error(message)
        raise ValueError(message)
    logger.warning(message)


def validate_settings(
    settings: dict | None = None,
    used_algorithms: dict | None = None,
    default_settings: dict | None = None,
    logger: logging.Logger | None = None,
    strict: bool | None = None,
) -> dict:
    """Validate the settings dictionary to ensure it has valid keys and values.

    Parameters
    ----------
    settings
        Dictionary containing settings to validate.
    used_algorithms
        List of algorithms that are currently used, to add the validated settings.
        Keys used:
        - 'optimisation': str
            Name of the optimisation algorithm used
        - 'infill': str
            Name of the infill strategy used
        - 'surrogate': str
            Name of the surrogate used
    default_settings
        Dictionary containing the default settings with their types and default values.
        If None, the DEFAULT_SETTINGS from DEFAULT_SETTINGS  will be used.
    logger
        Logger instance to use for logging warnings.
    strict
        If True, an unknown key or an invalid value raises a ValueError. If False,
        it only logs a warning and the key is ignored or the value is replaced (by
        the default, minimum or maximum). If None, the 'strict_validation' setting
        in settings is used (default: False).

    Returns
    -------
    dict
        The validated settings: every default setting, with the valid values from
        settings, plus the nested settings of the used algorithms.

    Raises
    ------
        ValueError: In strict mode, if a key is unknown or a value is invalid.
    """
    if logger is None:
        logger = get_logger()

    # Check if default_settings is None, if so, use the DEFAULT_SETTINGS
    if default_settings is None:
        default_settings = getattr(DefaultSettings, "DEFAULT_SETTINGS", {})

    # Determine the strict mode from the settings if not given explicitly
    if strict is None:
        strict_setting = settings.get("strict_validation") if settings else None
        strict = strict_setting if isinstance(strict_setting, bool) else False

    # Setup the validated settings with (copies of) the default values
    validated_settings = {
        key: copy.deepcopy(default_settings[key]["default"]) for key in default_settings
    }

    # Check all the values in the input settings
    if settings is not None:
        for key in settings:
            # Check if the key is a valid settings entry
            if key not in default_settings:
                _handle_invalid(
                    f"{key} is not a valid setting entry, ignoring.", logger, strict
                )
                continue

            value_info = default_settings[key]
            if value_info.get("type", None) is None:
                logger.warning(
                    f"Type for setting '{key}' is not defined in default settings."
                )
                # Use a copy, so the shared default settings are not changed
                value_info = {**value_info, "type": type(value_info["default"])}

            # Add the validated setting value to the output settings
            validated_settings[key] = check_value(
                settings[key],
                value_info,
                key,
                logger=logger,
                strict=strict,
            )

    # Add the validated settings of the used algorithms to the default settings
    if used_algorithms:
        if settings is None:
            settings = {}
        for key in used_algorithms:
            algorithm = used_algorithms[key]
            if not isinstance(algorithm, str):
                # Invalid algorithm names are reported when the component is created
                continue
            default_settings_name = algorithm.upper() + "_DEFAULT_SETTINGS"
            default_settings_to_use = getattr(
                DefaultSettings, default_settings_name, {}
            )

            # If twostep infill is used, add the single fidelity infill default
            # settings as well
            if default_settings_name == "TWOSTEP_DEFAULT_SETTINGS":
                single_fid_infill = settings.get("infill", {}).get(
                    "single_fid_infill", None
                )
                if not isinstance(single_fid_infill, str):
                    # None or an invalid value: the value itself is validated (and
                    # replaced by the default) below, so use the default here
                    single_fid_infill = default_settings_to_use.get(
                        "single_fid_infill"
                    ).get("default")
                single_fid_infill_def_sett_name = (
                    single_fid_infill.upper() + "_DEFAULT_SETTINGS"
                )
                default_settings_to_use = {
                    **getattr(DefaultSettings, single_fid_infill_def_sett_name, {}),
                    **default_settings_to_use,
                }

            validated_settings[key] = validate_settings(
                settings.get(key, None),
                None,
                default_settings_to_use,
                logger=logger,
                strict=strict,
            )

    # Check if seed for KRG is set equal to the overall random seed.
    if "seed" in validated_settings.get("surrogate", {}):
        if validated_settings["surrogate"]["seed"] != validated_settings["random_seed"]:
            logger.warning(
                f"Surrogate seed ({validated_settings['surrogate']['seed']}) is not "
                f"equal to the overall random seed "
                f"({validated_settings['random_seed']})"
                f", setting surrogate seed to overall random seed for reproducibility."
            )
            validated_settings["surrogate"]["seed"] = validated_settings["random_seed"]

    return validated_settings


def check_value(
    value: float | int | str | bool | list | dict,
    value_info: dict,
    key: str,
    logger: logging.Logger | None = None,
    strict: bool = False,
) -> float | int | str | bool | list | dict:
    """Check if the settings value is valid.

    Parameters
    ----------
    value
        Value to check.
    value_info
        Dictionary containing the type, default value, and optional constraints.
        Possible optional constraints are:
        - min: float | int, minimum value
        - max: float | int, maximum value
        - allowed: list, list of accepted values
    key
        Key of the setting being checked, used for warning messages.
    logger
        Logger instance to use for logging warnings.
    strict
        If True, an invalid value raises a ValueError instead of being replaced.

    Returns
    -------
    float | int | str | bool | list | dict
        Validated value, or the default value if the input value is invalid.

    Raises
    ------
    ValueError
        In strict mode, if the value is invalid.
    """
    if logger is None:
        logger = get_logger()

    # Check expected type
    if "type" in value_info:
        expected_type = value_info["type"]
        allowed_types = (
            expected_type.__args__
            if isinstance(expected_type, types.UnionType)
            else (expected_type,)
        )

        if isinstance(value, bool):
            # bool is a subclass of int, only accept it for bool settings
            wrong_type = bool not in allowed_types
        elif isinstance(value, int) and float in allowed_types:
            # Accept an int for a float setting, converted to float
            if int not in allowed_types:
                value = float(value)
            wrong_type = False
        elif isinstance(value, list) and tuple in allowed_types:
            # Convert list to tuple if the expected type is tuple (e.g. from JSON)
            value = tuple(value) if list not in allowed_types else value
            wrong_type = False
        elif isinstance(value, tuple) and list in allowed_types:
            # Convert tuple to list if the expected type is list
            value = list(value) if tuple not in allowed_types else value
            wrong_type = False
        else:
            wrong_type = not isinstance(value, expected_type)

        if wrong_type:
            _handle_invalid(
                f"Value for setting '{key}' is of type {type(value)}, "
                f"expected {value_info['type']}, using default value: "
                f"{value_info['default']}.",
                logger,
                strict,
            )
            return copy.deepcopy(value_info["default"])

    # Check min value
    if "min" in value_info and value is not None:
        if value < value_info["min"]:
            _handle_invalid(
                f"Value for setting '{key}' ({value}) is below the minimum of "
                f"{value_info['min']}, using minimum value: {value_info['min']}.",
                logger,
                strict,
            )
            return copy.deepcopy(value_info["min"])

    # Check max value
    if "max" in value_info and value is not None:
        if value > value_info["max"]:
            _handle_invalid(
                f"Value for setting '{key}' ({value}) is above the maximum of "
                f"{value_info['max']}, using max value: {value_info['max']}.",
                logger,
                strict,
            )
            return copy.deepcopy(value_info["max"])

    # Check accepted values
    if "allowed" in value_info and value is not None:
        if value not in value_info["allowed"]:
            _handle_invalid(
                f"Value for setting '{key}' is not in the list of accepted "
                f"values {value_info['allowed']}, using default value: "
                f"{value_info['default']}.",
                logger,
                strict,
            )
            return copy.deepcopy(value_info["default"])

    return value
