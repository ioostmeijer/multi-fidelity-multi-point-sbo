import importlib
import inspect
import json
import sys
import warnings
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest
from mf2 import forrester

from sbo.benchmarks import FORRESTER
from sbo.DefaultSettings import DEFAULT_SETTINGS
from sbo.utils import Config


def funcs_behave_equal(
    f: Callable,
    g: Callable,
    samples: None | list[float] = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> bool:
    """Check if two functions behave the same on a set of sample inputs.

    This function first attempts to compare the source code of the functions.
    If that fails, it compares the bytecode and constants.
    Finally, it runs both functions on a set of sample inputs and compares the outputs.

    Parameters
    ----------
    f
        First function to compare.
    g
        Second function to compare.
    samples
        Sample inputs to test the functions on. If None, a default set is used.
    atol
        Absolute tolerance for numerical comparisons.
    rtol
        Relative tolerance for numerical comparisons.

    Returns
    -------
    True if the functions behave the same, False otherwise.
    """
    # try to compare source
    try:
        if inspect.getsource(f).strip() == inspect.getsource(g).strip():
            return True
    except (OSError, TypeError):
        pass

    # fallback: compare bytecode & constants
    try:
        if (
            f.__code__.co_code == g.__code__.co_code
            and f.__code__.co_consts == g.__code__.co_consts
        ):
            return True
    except AttributeError:
        pass

    # Compare results on a set of sample inputs
    if samples is None:
        samples = [0, 1, -1, 0.5, 2]
    for s in samples:
        out_f = f(s)
        out_g = g(s)
        if isinstance(out_f, np.ndarray) or isinstance(out_g, np.ndarray):
            if not np.allclose(out_f, out_g, atol=atol, rtol=rtol):
                return False
        else:
            if out_f != out_g:
                return False
    return True


@pytest.fixture
def mock_validate_settings() -> Mock:
    """Fixture to mock the validate_settings function in the Config module."""
    mod = importlib.import_module(Config.__module__)
    with patch.object(
        mod,
        "validate_settings",
        return_value={
            "optimisation": {"max_iter": 100},
            "additional": "value",
            "random_seed": 42,
        },
    ) as mock:
        yield mock


@pytest.fixture
def basic_config(mock_validate_settings: Mock) -> Config:
    """Fixture to provide a basic Config instance for testing."""
    return Config(
        used_algorithms={
            "infill": "Algorithm 1",
            "optimisation": "Algorithm Opt",
            "surrogate": "Algorithm Sur",
        },
        lb=[0.0],
        ub=[5.0],
        functions=[lambda x: x**2],
        settings=None,
    )


@pytest.mark.unit
def test_config_init(basic_config: Config) -> None:
    """Test the initialisation of the Config class."""
    config = basic_config
    assert config._used_algorithms == {
        "infill": "Algorithm 1",
        "optimisation": "Algorithm Opt",
        "surrogate": "Algorithm Sur",
    }
    assert config.lb == [0.0]
    assert config.ub == [5.0]
    assert len(config.functions) == 1
    assert config._settings["optimisation"]["max_iter"] == 100


@pytest.mark.unit
def test_singleton_behaviour(basic_config: Config) -> None:
    """Test that a second Config call re-initialises the singleton instance."""
    config1 = basic_config
    with patch.object(config1.logger, "warning") as mock_warning:
        config2 = Config(
            used_algorithms={"infill": "Algorithm 2"},
            lb=[5.0],
            ub=[10.0],
            functions=[lambda x: x**2 + 1],
            settings=None,
        )
    assert config1 is config2
    assert mock_warning.called
    # The new problem definition is used, not the old one
    assert config2.used_algorithms == {"infill": "Algorithm 2"}
    assert config2.lb == [5.0]
    assert config2.ub == [10.0]


@pytest.mark.unit
def test_destroy_instance(basic_config: Config) -> None:
    """Test that destroying the Config instance allows for a new instance."""
    config1 = basic_config
    Config.destroy_instance()
    config2 = Config(
        used_algorithms={"infill": "Algorithm 2"},
        lb=[5.0],
        ub=[10.0],
        functions=[lambda x: x**2 + 1],
        settings=None,
    )
    assert config1 is not config2


@pytest.mark.unit
def test_invalid_functions(mock_validate_settings: Mock) -> None:
    """Test that providing an empty list of functions raises a ValueError."""
    with pytest.raises(ValueError, match="At least one function should be provided."):
        Config(
            used_algorithms={"infill": "Algorithm 1"},
            lb=[0.0],
            ub=[5.0],
            functions=[],
            settings=None,
        )


@pytest.mark.unit
def test_mismatched_bounds(mock_validate_settings: Mock) -> None:
    """Test that providing mismatched lengths for lb and ub raises a ValueError."""
    with pytest.raises(
        ValueError, match="he lower and upper bound should have equal length."
    ):
        Config(
            used_algorithms={"infill": "Algorithm 1"},
            lb=[0.0, 1.0],
            ub=[5.0],
            functions=[lambda x: x**2],
            settings=None,
        )


@pytest.mark.unit
def test_invalid_bounds(mock_validate_settings: Mock) -> None:
    """Test that providing invalid bounds raises a ValueError."""
    with pytest.raises(ValueError, match="Not all ub > lb."):
        Config(
            used_algorithms={"infill": "Algorithm 1"},
            lb=[0.0, 6.0],
            ub=[5.0, 5.0],
            functions=[lambda x: x**2],
            settings=None,
        )


@pytest.mark.unit
def test_properties(basic_config: Config) -> None:
    """Test the properties of the Config class."""
    config = basic_config
    assert config.valid_keys == ["optimisation.max_iter", "additional", "random_seed"]
    assert config.lb == [0.0]
    assert config.ub == [5.0]
    assert np.array_equal(config.bounds, np.array([[0.0, 5.0]]))
    assert funcs_behave_equal(config.functions[0], lambda x: x**2)


@pytest.mark.unit
def test_flatten_settings(basic_config: Config) -> None:
    """Test the flattening of settings in the Config class."""
    config = basic_config
    assert config._dot_settings == {
        "additional": "value",
        "optimisation.max_iter": 100,
        "random_seed": 42,
    }


@pytest.mark.unit
def test_valid_keys(basic_config: Config) -> None:
    """Test the retrieval of valid keys in the Config class."""
    config = basic_config
    assert config.valid_keys == ["optimisation.max_iter", "additional", "random_seed"]


@pytest.mark.unit
def test_validate_settings_called(mock_validate_settings: Mock) -> None:
    """Test that the validate_settings function is called during initialisation."""
    Config(
        used_algorithms={"infill": "Algorithm 1"},
        lb=[0.0],
        ub=[5.0],
        functions=[lambda x: x**2],
        settings=None,
    )
    mock_validate_settings.assert_called_once()


@pytest.mark.unit
def test_access_settings(mock_validate_settings: Mock) -> None:
    """Test accessing settings in the Config class."""
    config = Config(
        used_algorithms={"infill": "Algorithm 1"},
        lb=[0.0],
        ub=[5.0],
        functions=[lambda x: x**2],
        settings={"optimisation": {"max_iter": 200}, "additional": "value"},
    )
    assert config._settings["optimisation"]["max_iter"] == 100
    assert config.get("additional") == "value"
    assert config.get("optimisation.max_iter") == 100
    assert config["additional"] == "value"
    assert config[("optimisation", "max_iter")] == 100
    assert config["optimisation.max_iter"] == 100
    with pytest.raises(KeyError, match="Key 'nonexistent' not found in settings."):
        assert config.get("nonexistent")
    with pytest.raises(KeyError, match="Key 'nonexistent.key' not found in settings."):
        assert config.get("nonexistent.key")


@pytest.mark.unit
def test_update_used_algorithms(basic_config: Config) -> None:
    """Test updating the used algorithms in the Config class."""
    config = basic_config
    config.update_used_algorithms({"infill": "Algorithm 2"})
    assert config._used_algorithms == {"infill": "Algorithm 2"}


@pytest.mark.unit
def test_update_settings(basic_config: Config) -> None:
    """Test updating settings in the Config class."""
    config = basic_config
    config.update_settings({"optimisation": {"max_iter": 150}})
    assert config._settings["optimisation"]["max_iter"] == 150
    config.update_settings("max_iter", 200, algorithm="optimisation")
    assert config._settings["optimisation"]["max_iter"] == 200
    config["additional"] = "new_value"
    assert config._settings["additional"] == "new_value"


@pytest.mark.unit
def test_update_settings_set(basic_config: Config) -> None:
    """Test setting settings in the Config class."""
    config = basic_config
    config["optimisation.max_iter"] = 175
    assert config._settings["optimisation"]["max_iter"] == 175
    config[("optimisation", "max_iter")] = 200
    assert config._settings["optimisation"]["max_iter"] == 200


@pytest.mark.unit
def test_wrong_used_algorithms(mock_validate_settings: Mock) -> None:
    """Test that providing wrong used_algorithms raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="used_algorithms can only contain the keys: "
        "'optimisation', 'infill', 'surrogate'.",
    ):
        Config(
            used_algorithms={"wrong_key": "Some Algorithm"},
            lb=[0.0],
            ub=[5.0],
            functions=[lambda x: x**2],
            settings=None,
        )


@pytest.mark.unit
def test_w_validate_settings() -> None:
    """Test the creation of Config with validate_settings."""
    config = Config(
        used_algorithms={},
        lb=[0.0],
        ub=[5.0],
        functions=[lambda x: x**2],
    )
    resulting_settings = {
        key: value["default"] for key, value in DEFAULT_SETTINGS.items()
    }
    assert config._settings == resulting_settings


@pytest.mark.unit
def test_config_from_json(tmp_path: Path) -> None:
    """Test the from_json method of the Config class."""
    config_data = {
        "used_algorithms": {
            "infill": "Random",
            "optimisation": "EgoSequential",
            "surrogate": "KRG",
        },
        "functions": {"module": "mf2", "attribute": "forrester"},
        "settings": {
            "optimisation": {"max_iterations": 150},
            "additional": "value",
        },
    }
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    config = Config.from_json(config_file)

    assert config._used_algorithms == {
        "infill": "Random",
        "optimisation": "EgoSequential",
        "surrogate": "KRG",
    }
    assert config.lb == forrester.l_bound
    assert config.ub == forrester.u_bound
    assert len(config.functions) == len(forrester.functions)
    for idx, func in enumerate(forrester.functions):
        assert funcs_behave_equal(config.functions[idx], func)
    assert config._settings["optimisation"]["max_iterations"] == 150
    assert "additional" not in config._settings


@pytest.mark.unit
def test_get_instance(basic_config: Config) -> None:
    """Test the get_instance method of the Config class."""
    config1 = basic_config
    config2 = Config.get_instance()
    assert config1 is config2


@pytest.mark.unit
def test_get_instance_no_instance() -> None:
    """Test that get_instance raises an error if no instance exists."""
    Config.destroy_instance()
    with pytest.raises(
        ValueError, match="Config instance has not been initialised yet."
    ):
        Config.get_instance()


@pytest.mark.unit
def test_config_from_json_file(tmp_path: Path) -> None:
    """Test the from_json method of the Config class with settings passed as file."""
    config_data = {
        "used_algorithms": {
            "infill": "Random",
            "optimisation": "EgoSequential",
            "surrogate": "KRG",
        },
        "functions": {
            "module": "mf2",
            "attribute": "forrester",
        },
        "settings": {
            "optimisation": {"max_iterations": 150},
            "additional": "value",
        },
    }
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    config = Config.from_json(config_file)

    assert config._used_algorithms == {
        "infill": "Random",
        "optimisation": "EgoSequential",
        "surrogate": "KRG",
    }
    assert config.lb == forrester.l_bound
    assert config.ub == forrester.u_bound
    assert len(config.functions) == len(forrester.functions)
    for idx, func in enumerate(forrester.functions):
        assert funcs_behave_equal(config.functions[idx], func)
    assert config._settings["optimisation"]["max_iterations"] == 150
    assert "additional" not in config._settings


@pytest.mark.unit
def test_config_from_json_dict(tmp_path: Path) -> None:
    """Test the from_json method of the Config class with settings passed as dict."""
    config_data = {
        "used_algorithms": {
            "infill": "Random",
            "optimisation": "EgoSequential",
            "surrogate": "KRG",
        },
        "functions": {
            "module": "mf2",
            "attribute": "forrester",
        },
        "settings": {
            "optimisation": {"max_iterations": 150},
            "additional": "value",
        },
    }

    config = Config.from_json(config_data)

    assert config._used_algorithms == {
        "infill": "Random",
        "optimisation": "EgoSequential",
        "surrogate": "KRG",
    }
    assert config.lb == forrester.l_bound
    assert config.ub == forrester.u_bound
    assert len(config.functions) == len(forrester.functions)
    for idx, func in enumerate(forrester.functions):
        assert funcs_behave_equal(config.functions[idx], func)
    assert config._settings["optimisation"]["max_iterations"] == 150
    assert "additional" not in config._settings


def _constraint_config(**kwargs) -> Config:
    """Create a single-fidelity config with the given constraint arguments."""
    return Config(
        used_algorithms={
            "infill": "EI",
            "optimisation": "EgoSequential",
            "surrogate": "KRG",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        **kwargs,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "constraints",
    [None, {}, {"inequality": []}, {"inequality": [], "equality": []}],
)
def test_empty_constraints_are_accepted(constraints: dict | None) -> None:
    """Test that missing or empty constraints count as no constraints."""
    config = _constraint_config(
        inexpensive_constraints=constraints, expensive_constraints=constraints
    )

    assert config.scipy_constraints() == []
    assert not any(config.expensive_constraints().values())
    assert not any(config.inexpensive_constraints.values())


@pytest.mark.unit
def test_inexpensive_constraints_are_accepted() -> None:
    """Test that inexpensive constraints do not raise the expensive-constraint error."""

    def g(x: np.ndarray) -> float:
        return float(x[0]) - 0.5

    config = _constraint_config(inexpensive_constraints={"inequality": [g]})

    assert config.inexpensive_inequality_constraints == [g]
    assert len(config.scipy_constraints()) == 1


@pytest.mark.unit
def test_expensive_constraints_incompatible_infill_raises() -> None:
    """Test that expensive constraints with an incompatible infill raise."""

    def g(x: np.ndarray) -> float:
        return float(x[0]) - 0.5

    with pytest.raises(ValueError, match="Expensive constraints are provided"):
        _constraint_config(expensive_constraints={"inequality": [[g]]})


@pytest.mark.unit
@pytest.mark.parametrize(
    "lb, ub",
    [([0.0, 0.0], [1.0, 1.0]), ([-5], [5]), (0.0, 1.0), (np.array([0, 1]), (2, 3))],
)
def test_validate_bounds_accepts_valid_bounds(lb: object, ub: object) -> None:
    """Valid bounds (lists, tuples, arrays or scalars) do not raise."""
    Config.validate_bounds(lb, ub)


@pytest.mark.unit
@pytest.mark.parametrize(
    "lb, ub",
    [([0.0, 2.0], [1.0, 1.0]), ([1.0], [1.0]), ([0.0], [1.0, 2.0])],
)
def test_validate_bounds_rejects_invalid_bounds(lb: list, ub: list) -> None:
    """Test that lb >= ub in any dimension, or unequal lengths, raise a ValueError."""
    with pytest.raises(ValueError):
        Config.validate_bounds(lb, ub)


@pytest.mark.unit
def test_rng_is_seeded_from_random_seed() -> None:
    """Two configs with the same random_seed give the same random numbers."""
    draws = []
    for _ in range(2):
        Config.destroy_instance()
        config = _constraint_config(settings={"random_seed": 7})
        draws.append(config.rng.random(3))
    np.testing.assert_array_equal(draws[0], draws[1])


@pytest.mark.unit
def test_set_bounds_moves_whole_domain() -> None:
    """set_bounds updates both bounds at once; separate setters cannot."""
    config = _constraint_config()  # lb [0], ub [1]

    with pytest.raises(ValueError):
        config.lb = [2.0]  # 2 >= current ub 1

    config.set_bounds([2.0], [3.0])
    assert list(config.lb) == [2.0]
    assert list(config.ub) == [3.0]

    with pytest.raises(ValueError):
        config.set_bounds([3.0], [2.0])


@pytest.mark.unit
def test_from_json_mf2_uses_sbo_benchmarks_with_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that "module": "mf2" warns and uses sbo.benchmarks, also without mf2."""
    real_import = importlib.import_module

    def fake_import_module(name: str, *args, **kwargs) -> object:
        if name == "mf2":
            raise ModuleNotFoundError("No module named 'mf2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.warns(DeprecationWarning, match='"module": "sbo.benchmarks"'):
        config = Config.from_json(
            {
                "used_algorithms": {"surrogate": "KRG", "infill": "EI"},
                "functions": {"module": "mf2", "attribute": "Forrester"},
            }
        )

    assert len(config.functions) == len(FORRESTER.functions)
    for idx, func in enumerate(FORRESTER.functions):
        assert funcs_behave_equal(config.functions[idx], func)
    assert list(config.lb) == list(FORRESTER.l_bound)
    assert list(config.ub) == list(FORRESTER.u_bound)


@pytest.mark.unit
@pytest.mark.parametrize("attribute", ["FORRESTER", "forrester", "Forrester"])
def test_from_json_sbo_benchmarks(attribute: str) -> None:
    """Test that "module": "sbo.benchmarks" finds a problem by name, without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = Config.from_json(
            {
                "used_algorithms": {"surrogate": "KRG", "infill": "EI"},
                "functions": {"module": "sbo.benchmarks", "attribute": attribute},
            }
        )

    assert len(config.functions) == len(FORRESTER.functions)
    for idx, func in enumerate(FORRESTER.functions):
        assert funcs_behave_equal(config.functions[idx], func)
    assert list(config.lb) == list(FORRESTER.l_bound)
    assert list(config.ub) == list(FORRESTER.u_bound)


@pytest.mark.unit
def test_from_json_sbo_benchmarks_unknown_name() -> None:
    """Test that an unknown benchmark name gives a ValueError."""
    with pytest.raises(ValueError, match="Unknown benchmark"):
        Config.from_json(
            {
                "used_algorithms": {"surrogate": "KRG", "infill": "EI"},
                "functions": {"module": "sbo.benchmarks", "attribute": "unknown"},
            }
        )


@pytest.mark.unit
def test_from_json_other_module_with_instance() -> None:
    """Test that an object (not a class) in another module is used as it is."""
    config = Config.from_json(
        {
            "used_algorithms": {"surrogate": "KRG", "infill": "EI"},
            "functions": {
                "module": "sbo.benchmarks.forrester",
                "attribute": "FORRESTER",
                "n_fidelities": 1,
            },
        }
    )

    assert len(config.functions) == 1
    assert funcs_behave_equal(config.functions[0], FORRESTER.functions[0])


@pytest.mark.unit
def test_from_json_other_module_with_class(tmp_path: Path) -> None:
    """Test that a class in another module is instantiated."""
    module_file = tmp_path / "my_functions_for_from_json.py"
    module_file.write_text(
        "import numpy as np\n"
        "\n"
        "\n"
        "class MyFunctions:\n"
        "    def __init__(self):\n"
        "        self.functions = [lambda x: np.sum(x**2)]\n"
        "        self.l_bound = np.array([0.0])\n"
        "        self.u_bound = np.array([2.0])\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        config = Config.from_json(
            {
                "used_algorithms": {"surrogate": "KRG", "infill": "EI"},
                "functions": {
                    "module": "my_functions_for_from_json",
                    "attribute": "MyFunctions",
                },
            }
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("my_functions_for_from_json", None)

    assert len(config.functions) == 1
    assert list(config.ub) == [2.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "func",
    [
        lambda x: float(np.sum(x**2)),  # plain float
        lambda x: int(round(float(np.sum(x**2)))),  # plain int
        lambda x: [float(np.sum(x**2))],  # list
        lambda x: np.sum(x**2),  # numpy scalar
        lambda x: np.atleast_1d(np.sum(x**2)),  # 1D array
        lambda x: np.array([[np.sum(x**2)]]),  # 2D array
    ],
)
def test_objective_function_output_is_1d_float_array(func: Callable) -> None:
    """Test that every objective function returns a 1D float array via Config."""
    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "EI", "optimisation": "EgoSequential"},
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        functions=[func],
    )

    result = config.functions[0](np.array([0.5, 1.0]))

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.dtype == float
    assert result[0] == pytest.approx(1.25, abs=1.0)  # int rounding allowed


@pytest.mark.unit
def test_objective_function_keeps_name() -> None:
    """Test that the wrapped objective function keeps the original name."""

    def my_objective(x: np.ndarray) -> float:
        return float(x[0])

    Config.destroy_instance()
    config = Config(
        used_algorithms={"infill": "EI"},
        lb=[0.0],
        ub=[1.0],
        functions=[my_objective],
    )

    assert config.functions[0].__name__ == "my_objective"
