import functools
import importlib
import inspect
import json
import logging
import warnings
from typing import Any, Callable, ClassVar, Optional

import numpy as np
from scipy.optimize import NonlinearConstraint

from ..benchmarks import get_benchmark
from ..constants import EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL
from .Logger import delete_logger, get_logger
from .validate_settings import validate_settings


class Config:
    """Singleton class to manage the settings used.

    This class is used to define the problem to be solved, including:
    - The lower and upper bounds of the problem
    - The functions to be used in the optimisation
    - The constraints to be used in the optimisation
    - The settings for the optimisation process

    It exposes properties for:
    - lb: The lower bound of the problem
    - ub: The upper bound of the problem
    - bounds: The combination of the bounds
    - functions: The list of functions to be used in the optimisation
    - constraints: A dict containing a list of equality and a list of inequality
        constrains
    - inequality constraints: A list of the inequality constraints.
    - equality constraints: A list of the equality constraints.
    - valid_keys: The list of valid setting keys

    It also exposes methods for:
    - get_setting: Get a setting value
    - get: Get a setting value using dot notation
    - set_setting: Set a setting value using dot notation
    - get_function: Get a specific function
    - update_settings: Update the settings
    - update_used_algorithms: Update the used algorithms and re-validate settings
    - validate_bounds: Static method to validate the bounds

    Retrieving and updating settings can be done in several ways:
    - Using get_setting and set_setting methods
        x = config.get_setting('max_iter', algorithm='optimisation')
        config.update_settings('max_iter', 200, algorithm='optimisation')
    - Using get and set methods with dot notation
        x = config.get('optimisation.max_iter')
        config.set('optimisation.max_iter', 200)
    - Using indexing operator overloads __getitem__ and __setitem__
        x = config['optimisation', 'max_iter']
        config['optimisation', 'max_iter'] = 200


    Attributes
    ----------
    name
        The name of the Config instance.
    """

    _instance: ClassVar[Optional["Config"]] = None
    _initialised: bool

    # Core problem definition
    _lb: np.ndarray
    _ub: np.ndarray
    _functions: list[Callable]
    name: str

    # Algorithm configuration
    _used_algorithms: dict[str, str]
    _settings: dict[str, Any]
    _dot_settings: dict[str, Any]

    # Constraints — inexpensive (passed directly to the optimiser)
    _constraints: dict[str, dict[str, list[Callable]]]
    _scipy_constraints: list[NonlinearConstraint]

    # Constraints — expensive (backed by surrogates)
    _expensive_constraint_surrogates: dict[str, list[Callable]]
    _expensive_constraint_surrogates_scipy: dict[str, NonlinearConstraint]

    # Logger
    _logger: logging.Logger

    def __new__(cls, *args, **kwargs) -> "Config":
        """Create or return the singleton class instance.

        Returns
        -------
        Config
            The singleton class instance.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        used_algorithms: dict,
        lb: np.ndarray | list[float],
        ub: np.ndarray | list[float],
        functions: list[Callable],
        inexpensive_constraints: dict | None = None,
        expensive_constraints: dict | None = None,
        settings: dict | None = None,
        logger: logging.Logger | None = None,
        name: str = __name__,
    ) -> None:
        """Initialise the Settings singleton class.

        Parameters
        ----------
        used_algorithms
            A dictionary containing the algorithms being used.
            Allowed keys are:
            - 'optimisation': str
            - 'infill': str
            - 'surrogate': str
        lb
            The lower bound of the problem ([x1, x2, ..., xn])
        ub
            The upper bound of the problem ([x1, x2 ..., xn])
        functions
            The list of functions to be used in the optimisation, ordered from HF to LF
        inexpensive_constraints
            A dict containing the constraints. The dict has two possible keys:
            equality and inequality. Each containing a list of constraint functions.
            If None no constraints will be used.
            These constraints will be used as is and will be directly used.
        expensive_constraints
            A dict containing the constraints. The dict has two possible keys:
            equality and inequality. Each containing a list of constraints. Each
            constraint is a list of Callable constraints, ordered from HF to LF.
            If None no expensive constraints will be used.
            From these constraints a surrogate will be created, which will be used in
            the optimisation. This surrogate will be updated throughout the optimisation
            process.
        settings
            A dictionary containing the settings to initialise with.
            If None, default settings will be used.
        logger
            The logger instance to use. Defaults to the module-level logger.
        name
            The name of the Config instance.

        Raises
        ------
        ValueError
            If used_algorithms has keys other than 'optimisation', 'infill' and
            'surrogate', no functions are given, the bounds are invalid, the constraints
            are malformed, or expensive constraints are given with an infill strategy
            that does not support them.
        """
        if logger is None:
            logger = get_logger()

        # Config is a singleton. A new Config(...) call defines a new problem, so
        # re-initialise the existing instance instead of silently keeping the old
        # problem definition. Clear all instance state to avoid stale attributes.
        if getattr(self, "_initialised", False):
            self._logger.warning(
                "Config was already initialised; re-initialising it with the new "
                "arguments. Objects that hold this Config now see the new problem. "
                "Call Config.destroy_instance() first to make this explicit."
            )
            self.__dict__.clear()

        # Initialise logger
        self._logger = logger

        if any(
            key not in ["optimisation", "infill", "surrogate"]
            for key in used_algorithms
        ):
            raise ValueError(
                "used_algorithms can only contain the keys: "
                "'optimisation', 'infill', 'surrogate'."
            )
        self._used_algorithms = used_algorithms
        self._settings = {}

        # Check functions
        if not functions:
            raise ValueError("At least one function should be provided.")

        # Wrap the functions, so every function returns a 1D float array (a function
        # may also return a plain float) and calls are logged in debug mode.
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug("Debug mode enabled, logging every function call.")
        self._functions = [
            self._wrap_objective(func, idx) for idx, func in enumerate(functions)
        ]

        self._constraints = {
            "inexpensive": {"equality": [], "inequality": []},
            "expensive": {"equality": [], "inequality": []},
        }
        for constraint_type, value in [
            ("inexpensive", inexpensive_constraints),
            ("expensive", expensive_constraints),
        ]:
            if value is not None:
                if not isinstance(value, dict):
                    msg = (
                        f"{constraint_type}_constraints should be a dict with keys "
                        f"'inequality' and 'equality'."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                for key in value.keys():
                    if key not in ["inequality", "equality"]:
                        msg = (
                            f"Invalid key ({key}) in {constraint_type}_constraints "
                            "dict, only 'inequality' and 'equality' are allowed."
                            "Ignoring this key."
                        )
                        self.logger.warning(msg)
                self._constraints[constraint_type] = {
                    "inequality": value.get("inequality", []),
                    "equality": value.get("equality", []),
                }

        # Check that the infill strategy supports expensive constraints. Empty
        # constraint dicts or lists count as no constraints.
        if any(self._constraints["expensive"].values()) and (
            used_algorithms.get("infill", None)
            not in EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL
        ):
            msg = (
                "Expensive constraints are provided but the selected infill "
                f"algorithm ({used_algorithms.get('infill', None)}) is "
                f"not compatible with expensive constraints. Please select an infill"
                f" algorithm from the following list: "
                f"{EXPENSIVE_CONSTRAINT_COMPATIBLE_INFILL}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self._scipy_constraints = []
        for key, value in self._constraints["inexpensive"].items():
            n_constr = len(value)
            if key == "equality" and n_constr > 0:
                self._scipy_constraints.append(
                    NonlinearConstraint(
                        fun=self._vectorise(value),
                        lb=np.zeros(n_constr) - 1e-8,
                        ub=np.zeros(n_constr) + 1e-8,
                    )
                )
            elif key == "inequality" and n_constr > 0:
                self._scipy_constraints.append(
                    NonlinearConstraint(
                        fun=self._vectorise(value),
                        lb=-np.inf * np.ones(n_constr),
                        ub=np.zeros(n_constr),
                    )
                )

        self._expensive_constraint_surrogates: dict[str, list[Callable]] = {
            "inequality": [],
            "equality": [],
        }
        self._expensive_constraint_surrogates_scipy: dict[str, NonlinearConstraint] = {}

        # Check bounds
        if isinstance(lb, list):
            lb = np.array(lb)
        if isinstance(ub, list):
            ub = np.array(ub)
        self.validate_bounds(lb, ub)
        self._lb = lb
        self._ub = ub

        if settings:
            self.update_settings(settings)
        else:
            # update settings with default values
            self._settings = validate_settings(
                settings=self._settings,
                used_algorithms=self._used_algorithms,
                logger=self.logger,
            )

        self._dot_settings = self._flatten_settings(self._settings)

        # Seeded random generator shared by the components (instead of seeding the
        # global NumPy random state)
        self._rng = np.random.default_rng(self._settings["random_seed"])

        # Set the name
        self.name = name

        # Set the initialised flag
        self._initialised = True

    def _flatten_settings(self, settings: dict, prefix: str = "") -> dict[str, Any]:
        """Flatten nested settings dictionary into dot notation.

        Parameters
        ----------
        settings
            The nested settings dictionary to flatten.
        prefix
            The key prefix of this (nested) level; empty for the top level.

        Returns
        -------
        dict[str, Any]
            The settings with nested keys joined by dots, e.g.
            {'optimisation.max_iterations': 300}.
        """
        result = {}
        for key, value in settings.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_settings(value, full_key))
            else:
                result[full_key] = value
        return result

    def keys(self) -> list[str]:
        """Get the list of valid setting keys.

        Returns
        -------
        list[str]
            list of all valid setting keys
        """
        return list(self._dot_settings.keys())

    @property
    def logger(self) -> logging.Logger:
        """Get the logger.

        Returns
        -------
        logging.Logger
            The logger instance.
        """
        return self._logger

    @property
    def valid_keys(self) -> list[str]:
        """Get the list of valid setting keys.

        Returns
        -------
        list[str]
            List of all valid setting keys
        """
        return self.keys()

    @property
    def lb(self) -> np.ndarray:
        """Get the lb.

        Returns
        -------
        np.ndarray
            The lower bound of every design variable, with shape (n_dimensions,).
        """
        return self._lb

    @lb.setter
    def lb(self, new_lb: np.ndarray | list) -> None:
        """Update the lb.

        Parameters
        ----------
        new_lb
            The new lb ([x1, ..., xn])
        """
        if isinstance(new_lb, list):
            new_lb = np.array(new_lb)
        self.validate_bounds(new_lb, self._ub)
        self._lb = new_lb

    @property
    def ub(self) -> np.ndarray:
        """Get the ub.

        Returns
        -------
        np.ndarray
            The upper bound of every design variable, with shape (n_dimensions,).
        """
        return self._ub

    @ub.setter
    def ub(self, new_ub: np.ndarray | list) -> None:
        """Update the ub.

        Parameters
        ----------
        new_ub
            The new ub ([x1, ..., xn])
        """
        if isinstance(new_ub, list):
            new_ub = np.array(new_ub)
        self.validate_bounds(self._lb, new_ub)
        self._ub = new_ub

    def set_bounds(self, lb: np.ndarray | list, ub: np.ndarray | list) -> None:
        """Update both bounds in one call.

        The lb and ub setters each validate the new bound against the current other
        bound, so moving the whole domain (e.g. from [0, 1] to [2, 3]) is not possible
        one bound at a time. This method validates the new bounds together.

        Parameters
        ----------
        lb
            The new lower bound ([x1, ..., xn])
        ub
            The new upper bound ([x1, ..., xn])
        """
        if isinstance(lb, list):
            lb = np.array(lb)
        if isinstance(ub, list):
            ub = np.array(ub)
        self.validate_bounds(lb, ub)
        self._lb = lb
        self._ub = ub

    @property
    def bounds(self) -> np.ndarray:
        """Get the combination of the bounds.

        Returns
        -------
        np.ndarray
            The combined bounds, [[lb1, ub1], ..., [lbn, ubn]]
        """
        return np.column_stack((self._lb, self._ub))

    def _wrap_objective(self, func: Callable, index: int) -> Callable:
        """Wrap an objective function so it always returns a 1D float array.

        The framework reads the objective value as the first element of the returned
        array. The wrapper converts a returned float, list or array of any shape to a
        1D float array, so a function may return a plain float. In debug mode every
        call and its result are logged.

        Parameters
        ----------
        func
            The objective function, taking the coordinates as a 1D np.ndarray.
        index
            The fidelity level of the function (0 = HF), used in the log messages.

        Returns
        -------
        Callable
            The wrapped function, with the name and docstring of func.
        """

        @functools.wraps(func)
        def wrapped_function(*args, **kwargs) -> np.ndarray:
            """Call the objective function and convert its result to a 1D array.

            Returns
            -------
            np.ndarray
                The result of the objective function as a 1D float array.
            """
            debug = self.logger.isEnabledFor(logging.DEBUG)
            if debug:
                self.logger.debug(
                    f"Calling function index: {index}, name "
                    f"{getattr(func, '__name__', func)}, with args: {args}, "
                    f"kwargs: {kwargs}"
                )
            result = np.atleast_1d(np.asarray(func(*args, **kwargs), dtype=float))
            result = result.ravel()
            if debug:
                self.logger.debug(
                    f"Function index: {index}, name "
                    f"{getattr(func, '__name__', func)} returned: {result}"
                )
            return result

        return wrapped_function

    @property
    def functions(self) -> list[Callable]:
        """Get the list of functions.

        Returns
        -------
        list[Callable]
            The list of functions, from HF to LF. Each returns a 1D float array (see
            _wrap_objective), also when the original function returns a float.
        """
        return self._functions

    def constraints(self, return_surrogate: bool = False) -> dict[str, dict]:
        """Get all constraints.

        Parameters
        ----------
        return_surrogate
            Whether to return the surrogates of the expensive constraints. If False, the
            real functions are returned

        Returns
        -------
        dict[str, dict]
            The constraints, as a dict with the keys ``'expensive'`` and
            ``'inexpensive'``. Each holds
            ``{'inequality': [...], 'equality': [...]}``.
        """
        if return_surrogate:
            return {
                **self._constraints,
                "expensive": self._expensive_constraint_surrogates,
            }
        return self._constraints

    def expensive_constraints(
        self, return_surrogate: bool = False
    ) -> dict[str, list[Callable]]:
        """Get the expensive constraints.

        Parameters
        ----------
        return_surrogate
            Whether to return the surrogates of the expensive constraints. If False, the
            real functions are returned

        Returns
        -------
        dict[str, list[Callable]]
            The expensive constraints:
            {'inequality': [...], 'equality': [...]}
        """
        if not return_surrogate:
            return self._expensive_constraint_surrogates
        return self._constraints.get("expensive", {"inequality": [], "equality": []})

    @property
    def inexpensive_constraints(self) -> dict[str, list[Callable]]:
        """Get the inexpensive constraints.

        Returns
        -------
        dict[str, list[Callable]]
            The constraints:
            {'inequality': [...], 'equality': [...]}
        """
        return self._constraints.get("inexpensive", {"inequality": [], "equality": []})

    def equality_constraints(self, return_surrogate: bool = False) -> dict[str, list]:
        """Get all equality constraints.

        Parameters
        ----------
        return_surrogate
            Whether to return the surrogates of the expensive constraints. If False, the
            real functions are returned

        Returns
        -------
        dict[str, list]
            The equality constraints:
            {'expensive': [...], 'inexpensive': [...]}
        """
        if return_surrogate:
            return {
                "expensive": self._expensive_constraint_surrogates.get("equality", []),
                "inexpensive": self._constraints.get("inexpensive", {}).get(
                    "equality", []
                ),
            }
        return {
            "expensive": self._constraints.get("expensive", {}).get("equality", []),
            "inexpensive": self._constraints.get("inexpensive", {}).get("equality", []),
        }

    def expensive_equality_constraints(
        self, return_surrogate: bool = False
    ) -> list[list[Callable]] | list[Callable]:
        """Get the expensive equality constraints.

        Parameters
        ----------
        return_surrogate
            Whether to return the surrogates of the expensive constraints. If False, the
            real functions are returned

        Returns
        -------
        list[list[Callable]] | list[Callable]
            The constraints:
            A list of the expensive equality constraints, each being a list of Callables
            order from HF to LF if not return_surrogate.
            A list of Callables of the surrogates of the expensive equality constraints.
        """
        if return_surrogate:
            return self._expensive_constraint_surrogates.get("equality", [])
        return self._constraints.get("expensive", {}).get("equality", [])

    @property
    def inexpensive_equality_constraints(self) -> list[Callable]:
        """Get the inexpensive equality constraints.

        Returns
        -------
        list[Callable]
            A list of the inexpensive equality constraints.
        """
        return self._constraints.get("inexpensive", {}).get("equality", [])

    def inequality_constraints(self, return_surrogate: bool = False) -> dict[str, list]:
        """Get all inequality constraints.

        Parameters
        ----------
        return_surrogate
            Whether to return the surrogates of the expensive constraints. If False, the
            real functions are returned

        Returns
        -------
        dict[str, list]
            The constraints:
            {'expensive': [...], 'inexpensive': [...]}
        """
        if return_surrogate:
            return {
                "expensive": self._expensive_constraint_surrogates.get(
                    "inequality", []
                ),
                "inexpensive": self._constraints.get("inexpensive", {}).get(
                    "inequality", []
                ),
            }
        return {
            "expensive": self._constraints.get("expensive", {}).get("inequality", []),
            "inexpensive": self._constraints.get("inexpensive", {}).get(
                "inequality", []
            ),
        }

    def expensive_inequality_constraints(
        self, return_surrogate: bool = False
    ) -> list[list[Callable]] | list[Callable]:
        """Get the expensive inequality constraints.

        Parameters
        ----------
        return_surrogate
            Whether to return the surrogates of the expensive constraints. If False, the
            real functions are returned

        Returns
        -------
        list[list[Callable]] | list[Callable]
            The constraints:
            A list of the expensive inequality constraints, each being a list of
            Callables order from HF to LF if not return_surrogate.
            Else a list of Callables of the surrogates of the inequality constraints.
        """
        if return_surrogate:
            return self._expensive_constraint_surrogates.get("inequality", [])
        return self._constraints.get("expensive", {}).get("inequality", [])

    @property
    def inexpensive_inequality_constraints(self) -> list[Callable]:
        """Get the inexpensive inequality constraints.

        Returns
        -------
        list[Callable]
            The inexpensive inequality constraints g(x), satisfied when g(x) <= 0. Empty
            if none are given.
        """
        return self._constraints.get("inexpensive", {}).get("inequality", [])

    @property
    def surrogate_constraints(self) -> dict[str, list[Callable]]:
        """Get the surrogate of the expensive constraints.

        Returns
        -------
        dict[str, list[Callable]]
            The constraints:
            A dict containing the surrogate of the expensive constraints. The dict has
            two keys: 'inequality' and 'equality'. With each entry a list of the
            surrogate instances.
        """
        return self._expensive_constraint_surrogates

    @surrogate_constraints.setter
    def surrogate_constraints(
        self, surrogate_constraints: dict[str, list[Callable]]
    ) -> None:
        """Set the surrogate of the expensive constraints.

        Parameters
        ----------
        surrogate_constraints
            A dict containing the surrogate of the expensive constraints. The dict has
            two keys: 'inequality' and 'equality'. With each entry a list of the
            surrogate instances.
        """
        self.surrogate_inequality_constraints = surrogate_constraints.get(
            "inequality", []
        )
        self.surrogate_equality_constraints = surrogate_constraints.get("equality", [])

    @property
    def surrogate_inequality_constraints(self) -> list[Callable]:
        """Get the surrogate of the expensive inequality constraints.

        Returns
        -------
        list[Callable]
            A list of the surrogate instances for the expensive inequality constraints.
        """
        return self._expensive_constraint_surrogates.get("inequality", [])

    @surrogate_inequality_constraints.setter
    def surrogate_inequality_constraints(
        self, surrogate_inequality_constraints: list[Callable]
    ) -> None:
        """Set the surrogate of the expensive inequality constraints.

        Parameters
        ----------
        surrogate_inequality_constraints
            A list of the surrogate instances for the expensive inequality constraints.
        """
        self._expensive_constraint_surrogates["inequality"] = (
            surrogate_inequality_constraints
        )

        # Update the scipy constraints
        n_constr = len(surrogate_inequality_constraints)
        if n_constr > 0:
            self._expensive_constraint_surrogates_scipy["inequality"] = (
                NonlinearConstraint(
                    fun=self._vectorise(surrogate_inequality_constraints),
                    lb=-np.inf * np.ones(n_constr),
                    ub=np.zeros(n_constr),
                )
            )

    @property
    def surrogate_equality_constraints(self) -> list[Callable]:
        """Get the surrogate of the expensive equality constraints.

        Returns
        -------
        list[Callable]
            A list of the surrogate instances for the expensive equality constraints.
        """
        return self._expensive_constraint_surrogates["equality"]

    @surrogate_equality_constraints.setter
    def surrogate_equality_constraints(
        self, surrogate_equality_constraints: list[Callable]
    ) -> None:
        """Set the surrogate of the expensive equality constraints.

        Parameters
        ----------
        surrogate_equality_constraints
            A list of the surrogate instances for the expensive equality constraints.
        """
        self._expensive_constraint_surrogates["equality"] = (
            surrogate_equality_constraints
        )

        # Update the scipy constraints
        n_constr = len(surrogate_equality_constraints)
        if n_constr > 0:
            self._expensive_constraint_surrogates_scipy["equality"] = (
                NonlinearConstraint(
                    fun=self._vectorise(surrogate_equality_constraints),
                    lb=np.zeros(n_constr) - 1e-8,
                    ub=np.zeros(n_constr) + 1e-8,
                )
            )

    def scipy_constraints(
        self, return_inexpensive: bool = True, return_surrogate: bool = False
    ) -> list[NonlinearConstraint]:
        """Get the inexpensive constraints in a format for scipy optimisers.

        Parameters
        ----------
        return_inexpensive
            Whether to return the inexpensive constraints.
        return_surrogate
            Whether to return the surrogate constraints.

        Returns
        -------
        list[NonlinearConstraint]
            A list of the constraints formatted for scipy optimisers.
        """
        if return_surrogate and return_inexpensive:
            return self._scipy_constraints + list(
                self._expensive_constraint_surrogates_scipy.values()
            )
        if return_surrogate and not return_inexpensive:
            return list(self._expensive_constraint_surrogates_scipy.values())
        if not return_surrogate and return_inexpensive:
            return self._scipy_constraints
        return []

    @property
    def num_fidelities(self) -> int:
        """Get the number of fidelities.

        Returns
        -------
        int
            The number of fidelity levels (the number of functions).
        """
        return len(self._functions)

    @property
    def dimension(self) -> int:
        """Get the problem dimension.

        Returns
        -------
        int
            The number of design variables.
        """
        return len(self._lb)

    @property
    def n_dimensions(self) -> int:
        """Get the problem dimension (alias of dimension).

        Returns
        -------
        int
            The number of design variables.
        """
        return self.dimension

    @property
    def rng(self) -> np.random.Generator:
        """Get the random generator, seeded with the 'random_seed' setting.

        Returns
        -------
        np.random.Generator
            The generator shared by the components of this optimisation.
        """
        return self._rng

    @property
    def settings(self) -> dict[str, Any]:
        """Get the settings dictionary.

        Returns
        -------
        dict[str, Any]
            The validated (nested) settings.
        """
        return self._settings

    @property
    def used_algorithms(self) -> dict:
        """Get the used algorithms.

        Returns
        -------
        dict
            The used algorithms ({'surrogate': str, 'infill': str,
            'optimisation': str}).
        """
        return self._used_algorithms

    def get_setting(
        self, key: str, algorithm: str | None = None
    ) -> Any:  # noqa: ANN401
        """Get a setting value.

        Parameters
        ----------
        key
            The key of the setting to get.
        algorithm
            The algorithm section to get the setting from.
            If specified, returns ``settings[algorithm][key]``.
            If None, returns ``settings[key]``.

        Returns
        -------
        Any
            The value of the setting.
        """
        return (
            self._settings.get(algorithm, {})[key] if algorithm else self._settings[key]
        )

    def get(
        self,
        key: str,
        default: Any = None,  # noqa: ANN401
        *,
        strict: bool = True,
    ) -> Any:  # noqa: ANN401
        """Get a setting value using dot notation.

        Parameters
        ----------
        key
            The key of the setting to get, in dot notation.
        default
            The default value to return if the key is not found.
        strict
            If True, raises a KeyError if the key is not found.
            If False, returns the default value if the key is not found.

        Returns
        -------
        Any
            The value of the setting.

        Raises
        ------
        KeyError
            If strict is True and the key is not in the settings.
        """
        if key not in self._dot_settings and strict:
            self.logger.error(f"Key '{key}' not found in settings.")
            raise KeyError(f"Key '{key}' not found in settings.")
        return self._dot_settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set a setting value using dot notation.

        Parameters
        ----------
        key
            The key of the setting to set, in dot notation.
        value
            The value to set the setting to.
        """
        algorithm, key = key.split(".", 1) if "." in key else (None, key)
        self.update_settings(key=key, value=value, algorithm=algorithm)

    def get_function(self, fid_level: int) -> Callable:
        """Get a specific function.

        Parameters
        ----------
        fid_level
            The index of the function to retrieve (order HF to LF)

        Returns
        -------
        Callable
            The selected function.

        Raises
        ------
        IndexError
            If the fidelity level is out of range.
        """
        if 0 <= fid_level < len(self._functions):
            return self._functions[fid_level]
        raise IndexError("The selected fid level is out of range.")

    def update_settings(self, *args, **kwargs) -> None:
        """Handle the update of settings.

        Possible inputs:
        - dict: a dictionary with key-value pairs
            (possibly nested for optimisation, infill, surrogate)
        - key: string, value: any: a single key-value pair to update in the top level
        - key: string, value: any, algorithm: string: a single key-value pair to update
            in a specific algorithm section (eg: 'optimisation', 'infill', 'surrogate')

        Parameters
        ----------
        *args
            See notes above
        **kwargs
            See notes above
        """
        if args and len(args) == 1 and isinstance(args[0], dict):
            settings_dict = args[0]
            for key, value in settings_dict.items():
                if isinstance(value, dict):
                    if key not in self._settings:
                        self._settings[key] = {}
                    for sub_key, sub_value in value.items():
                        self._settings[key][sub_key] = sub_value
                else:
                    self._settings[key] = value

        elif len(args) == 2:
            key = args[0]
            value = args[1]
            if "algorithm" in kwargs:
                algorithm = kwargs.pop("algorithm")
                self._settings[algorithm][key] = value
            else:
                self._settings[key] = value

        elif "key" in kwargs and "value" in kwargs:
            key = kwargs.pop("key")
            value = kwargs.pop("value")
            algorithm = kwargs.pop("algorithm", None)

            if algorithm is None:
                self._settings[key] = value
            else:
                self._settings[algorithm][key] = value

        # Validate the settings after update
        self._settings = validate_settings(
            settings=self._settings,
            used_algorithms=self._used_algorithms,
            logger=self.logger,
        )

        # Update the dot notation settings
        self._dot_settings = self._flatten_settings(self._settings)

    def update_used_algorithms(self, used_algorithms: dict) -> None:
        """Update the used algorithms and re-validate settings.

        Parameters
        ----------
        used_algorithms
            A dictionary containing the new used algorithms.
        """
        self._used_algorithms = used_algorithms

        # Re-validate settings with the new used_algorithms
        self._settings = validate_settings(
            settings=self._settings,
            used_algorithms=self._used_algorithms,
            logger=self.logger,
        )

        # Update the dot notation settings
        self._dot_settings = self._flatten_settings(self._settings)

    def __getitem__(self, key: str | tuple[str, str]) -> Any:  # noqa: ANN401
        """Overload the indexing operator to get settings values.

        Parameters
        ----------
        key
            The key of the setting to get.
            If a tuple of (algorithm, real_key) is provided, gets
                settings[algorithm][real_key].
            If a string is provided, gets settings[key].

        Returns
        -------
        Any
            The value of the setting.

        Raises
        ------
        KeyError
            If the key is not a str or a tuple of two str.
        """
        if isinstance(key, tuple) and len(key) == 2:
            algorithm, real_key = key
            return self.get_setting(real_key, algorithm)

        if isinstance(key, str) and key in self._dot_settings:
            return self.get(key)

        if isinstance(key, str):
            return self.get_setting(key)

        self._logger.error(f"Invalid key type for Config.__getitem__: {type(key)}")
        raise KeyError("Invalid key type for Config.__getitem__")

    def __setitem__(
        self, key: str | tuple[str, str], value: Any  # noqa: ANN401
    ) -> None:
        """Overload the indexing operator to set settings values.

        Parameters
        ----------
        key
            The key of the setting to set.
            If a tuple of (algorithm, real_key) is provided, sets
                settings[algorithm][real_key] = value.
            If a string is provided, sets settings[key] = value.
        value
            The value to set the setting to.

        Raises
        ------
        KeyError
            If the key is not a str or a tuple of two str.
        """
        if isinstance(key, tuple) and len(key) == 2:
            algorithm, real_key = key
            self.update_settings(key=real_key, value=value, algorithm=algorithm)
            return

        if isinstance(key, str) and key in self._dot_settings:
            self.set_setting(key=key, value=value)
            return

        if isinstance(key, str):
            self.update_settings(key=key, value=value)
            return

        self._logger.error(f"Invalid key type for Config.__setitem__: {type(key)}")
        raise KeyError("Invalid key type for Config.__setitem__")

    @staticmethod
    def validate_bounds(lb: np.ndarray, ub: np.ndarray) -> None:
        """Staticmethod which raises a ValueError if the bounds are not correct.

        Parameters
        ----------
        lb
            the lower bound ([x1, ..., xn])
        ub
            The upper bound ([x1, ..., xn])

        Raises
        ------
        ValueError
            If the bounds have different lengths, or ub <= lb in any dimension.
        """
        lb = np.atleast_1d(np.asarray(lb, dtype=float))
        ub = np.atleast_1d(np.asarray(ub, dtype=float))
        if lb.shape != ub.shape:
            raise ValueError("The lower and upper bound should have equal length.")
        if np.any(lb >= ub):
            raise ValueError("Not all ub > lb.")

    @classmethod
    def from_json(cls: "Config", input_settings: str | dict) -> "Config":
        """Create the singleton instance from a JSON file.

        The functions are set by the "functions" entry, with a "module" and an
        "attribute":

        - "module": "sbo.benchmarks": the attribute is the name of a benchmark problem
            (case and spaces don't matter, e.g. "forrester" or "FORRESTER"), see
            `sbo.benchmarks.get_benchmark`;
        - "module": "mf2": deprecated, gives a DeprecationWarning and uses the
            benchmark problem of the same name from `sbo.benchmarks` (mf2 itself is
            not needed);
        - any other module: the attribute is a class, which is instantiated without
            arguments, or an object such as a `MultiFidelityFunction` instance. Either
            must have `functions`, `l_bound` and `u_bound` (see
            `sbo.benchmarks.FunctionSet`).

        Parameters
        ----------
        input_settings
            The path to the JSON file or the loaded json as a dict.

        Returns
        -------
        Config
            The singleton instance.

        Raises
        ------
        ValueError
            If n_fidelities is 0 or negative, or the benchmark problem does not exist.
        """
        if isinstance(input_settings, dict):
            input_dict = input_settings
        else:
            input_dict = None
            with open(input_settings, "r") as f:
                input_dict = json.load(f)
        resulting_config_dict = input_dict.copy()

        # Get the functions
        function_set = cls._load_function_set(
            input_dict["functions"]["module"], input_dict["functions"]["attribute"]
        )

        n_functions = input_dict["functions"].get("n_fidelities", None)

        clamp_msg = None
        if n_functions is not None and n_functions > len(function_set.functions):
            clamp_msg = (
                f"n_fidelities={n_functions} is larger than the "
                f"{len(function_set.functions)} available fidelities; "
                "using all of them."
            )
        if n_functions is None or n_functions >= len(function_set.functions):
            resulting_config_dict["functions"] = function_set.functions
        elif n_functions <= 0:
            raise ValueError(
                "Currently 0 or a negative number of functions are to be used, "
                "this is an invalid setting."
            )
        else:
            resulting_config_dict["functions"] = function_set.functions[:n_functions]
        resulting_config_dict["lb"] = np.array(function_set.l_bound)
        resulting_config_dict["ub"] = np.array(function_set.u_bound)

        # Import the constraints if given
        if "constraints" in input_dict and input_dict["constraints"]:
            constraint_module = importlib.import_module(
                input_dict["constraints"].pop("module")
            )
            for key in ["inequality", "equality"]:
                if key in input_dict["constraints"]:
                    constraint_funcs = []
                    for func_name in input_dict["constraints"][key]:
                        constraint_func = getattr(constraint_module, func_name)
                        constraint_funcs.append(constraint_func)
                    resulting_config_dict["constraints"][key] = constraint_funcs

        # Set up the logger
        if "logger_instance" in input_dict:
            logger_instance = input_dict["logger_instance"]
            resulting_config_dict["logger"] = logger_instance
            resulting_config_dict.pop("logger_name", None)
            resulting_config_dict.pop("logger_instance", None)

        elif "logger_name" in input_dict:
            logger_instance = get_logger(
                name=input_dict.get("logger_name", __name__),
                **input_dict.get("logger", {}),
            )
            resulting_config_dict.pop("logger_name", None)
            resulting_config_dict["logger"] = logger_instance

        elif isinstance(input_dict.get("logger"), dict):
            # Logger settings without a name: use them for the default logger
            resulting_config_dict["logger"] = get_logger(**input_dict["logger"])

        if clamp_msg is not None:
            (resulting_config_dict.get("logger") or get_logger()).warning(clamp_msg)

        return cls(**resulting_config_dict)

    @staticmethod
    def _import_function_module(module_name: str) -> object:
        """Import the module that holds the functions.

        Parameters
        ----------
        module_name
            The name of the module.

        Returns
        -------
        object
            The imported module.

        Raises
        ------
        ModuleNotFoundError
            If the module can not be imported.
        """
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            msg = f"Could not import the function module '{module_name}': {e}."
            get_logger().error(msg)
            raise ModuleNotFoundError(msg) from e

    @classmethod
    def _load_function_set(
        cls: "type[Config]", module_name: str, attribute: str
    ) -> Any:  # noqa: ANN401
        """Get the set of functions named in the "functions" entry of from_json.

        Parameters
        ----------
        module_name
            The "module" entry: "sbo.benchmarks", "mf2" (deprecated) or another module.
        attribute
            The "attribute" entry: the name of a benchmark problem, or of a class or
            object in the module.

        Returns
        -------
        Any
            An object with `functions`, `l_bound` and `u_bound` (see
            `sbo.benchmarks.FunctionSet`).
        """
        if module_name == "sbo.benchmarks":
            return get_benchmark(attribute)

        if module_name == "mf2":
            # stacklevel=3: points at the code that called from_json
            warnings.warn(
                '"module": "mf2" is deprecated and will be removed in version 2.0.0; '
                'use "module": "sbo.benchmarks" instead.',
                DeprecationWarning,
                stacklevel=3,
            )
            return get_benchmark(attribute)

        function_module = cls._import_function_module(module_name)
        function_set = getattr(function_module, attribute)
        # A class is instantiated; an object (e.g. a MultiFidelityFunction instance)
        # is used as it is
        return function_set() if inspect.isclass(function_set) else function_set

    # Convert inexpensive constraints to scipy format
    @staticmethod
    def _vectorise(funcs: list[Callable]) -> Callable:
        """Convert a list of constraint functions to a single vectorised function.

        Parameters
        ----------
        funcs
            A list of constraint functions, each taking x as input and returning a
            scalar.

        Returns
        -------
        Callable
            A single function that takes x as input and returns a vector of the
            outputs of the individual constraint functions.
        """

        def vectorised_func(x: np.ndarray) -> np.ndarray:
            """The resulting vectorised function.

            Parameters
            ----------
            x
                The input to the constraint functions.

            Returns
            -------
            np.ndarray
                A vector containing the outputs of the individual constraint
                functions.
            """
            return np.array([func(x) for func in funcs])

        return vectorised_func

    @classmethod
    def get_instance(cls: type["Config"]) -> "Config":
        """Get the singleton instance.

        Returns
        -------
        Config
            The singleton instance.

        Raises
        ------
        ValueError
            If no Config instance has been created yet.
        """
        if not cls._instance:
            raise ValueError("Config instance has not been initialised yet.")
        return cls._instance

    @classmethod
    def destroy_instance(cls: type["Config"], destroy_logger: bool = False) -> None:
        """Destroy the singleton instance.

        Parameters
        ----------
        destroy_logger
            Whether to also delete the logger associated with the instance.
        """
        if destroy_logger and cls._instance:
            logger = cls._instance.logger
            delete_logger(logger.name)

        if cls._instance:
            cls._instance = None
            cls._initialised = False
            cls._settings = {}
            cls._dot_settings = {}
