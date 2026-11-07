import importlib
import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from logger_helper import _make_logger

from sbo import SBO, Config
from sbo.infill import GEI
from sbo.optimisation import EgoSequential


def _get_algorithms(
    src_path: str | Path, alg_type: str, files_not: list[str]
) -> list[str]:
    """Load all the possible algorithms available.

    Parameters
    ----------
    src_path
        Path to the source folder.
    alg_type
        The algorithm type to load (infill, optimisation or surrogate).
    files_not
        List of files not to include in the options


    Returns
    -------
    A list with all the algorithms available.
    """
    directory_to_search = src_path.joinpath(alg_type).resolve()
    return [
        strategy_name[:-3]
        for strategy_name in os.listdir(directory_to_search)
        if strategy_name.endswith(".py") and strategy_name not in files_not
    ]


def _dynamic_class_import(alg_type: str, specific_class: str) -> Any:  # noqa ANN401
    """Dynamically import the class.

    Parameters
    ----------
    alg_type
        The algorithm type (infill, surrogate or optimisation)
    specific_class
        The specific class to import.

    Returns
    -------
    The imported class.
    """
    module = importlib.import_module(f"sbo.{alg_type}")
    return getattr(module, specific_class)


try:
    from mockmpi import mock_mpiexec
    from mpi4py import MPI

    def _test_init(comm: MPI.Comm, used_algorithms: dict) -> None:
        """The test function used in test_sbo_initialisation if MPI is available.

        This test differs from MPI not available, as it does not allow the optimisation
        class to be also be EgoSequential.

        Parameters
        ----------
        comm
            The mock comm module
        used_algorithms
            The dictionary containing the used algorithms.
        """
        if comm.rank == 0:
            config = Config(
                used_algorithms=used_algorithms,
                lb=[0.0],
                ub=[1.0],
                functions=[lambda x: x**2, lambda x: x**2 + 0.1],
                logger=_make_logger(),
            )
            sbo = SBO(config, mpi_comm=comm)
            infill = used_algorithms["infill"]
            surrogate = used_algorithms["surrogate"]
            optimisation = used_algorithms["optimisation"]

            assert isinstance(
                sbo._infill, _dynamic_class_import("infill", infill) | GEI
            ), (
                f"The initialised infill algorithms is not of the correct type for "
                f"{infill}."
            )
            assert isinstance(
                sbo._surrogate, _dynamic_class_import("surrogate", surrogate)
            ), (
                f"The initialised surrogate algorithms is not of the correct type for "
                f"{surrogate}."
            )
            assert isinstance(
                sbo._optimisation, _dynamic_class_import("optimisation", optimisation)
            ), (
                f"The initialised optimisation algorithms is not of the correct type "
                f"for {optimisation}."
            )

            Config.destroy_instance()

except (ImportError, RuntimeError):
    mock_mpiexec = None

    def _test_init(used_algorithms: dict) -> None:
        """The test function used in test_sbo_initialisation if MPI is not available.

        This test differs from MPI available, as it does allow the optimisation
        class to be also be EgoSequential. Which is the algorithm instantiated from
        EgoAsync and EgoMPI if MPI is not available.

        Parameters
        ----------
        used_algorithms
            The dictionary containing the used algorithms.
        """
        config = Config(
            used_algorithms=used_algorithms,
            lb=[0.0],
            ub=[1.0],
            functions=[lambda x: x**2, lambda x: x**2 + 0.1],
            logger=_make_logger(),
        )
        sbo = SBO(config)
        infill = used_algorithms["infill"]
        surrogate = used_algorithms["surrogate"]
        optimisation = used_algorithms["optimisation"]
        assert isinstance(sbo._infill, _dynamic_class_import("infill", infill) | GEI), (
            f"The initialised infill algorithms is not of the correct type for "
            f"{infill}."
        )
        assert isinstance(
            sbo._surrogate, _dynamic_class_import("surrogate", surrogate)
        ), (
            f"The initialised surrogate algorithms is not of the correct type for "
            f"{surrogate}."
        )
        assert isinstance(
            sbo._optimisation,
            _dynamic_class_import("optimisation", optimisation) | EgoSequential,
        ), (
            f"The initialised optimisation algorithms is not of the correct type for "
            f"{optimisation}."
        )

        Config.destroy_instance()


def _nested_samples(samples: dict[int, list[dict]]) -> bool:
    """Helper function to check if samples are nested.

    Parameters
    ----------
    samples
        The samples dictionary from the surrogate model.

    Returns
    -------
    True if samples are nested, False otherwise.
    """
    hf_coords = {tuple(sample["coord"]) for sample in samples.get(0, [])}
    lf_coords = {tuple(sample["coord"]) for sample in samples.get(1, [])}
    return hf_coords.issubset(lf_coords)


@pytest.mark.unit
def test_sbo_initialisation() -> None:
    """Test that sbo can be initialised with every available algorithm."""
    src_path = Path(__file__).parent.joinpath("..", "..", "src/sbo").resolve()

    # CEI and CoKRG are not implemented yet, and FilteredGEI needs CoKRG; they raise
    # NotImplementedError on init
    infill_alg = _get_algorithms(
        src_path, "infill", ["__init__.py", "Base.py", "CEI.py", "FilteredGEI.py"]
    )
    surrogate_alg = _get_algorithms(
        src_path, "surrogate", ["__init__.py", "Base.py", "CoKRG.py"]
    )
    optim_alg = _get_algorithms(src_path, "optimisation", ["__init__.py", "Base.py"])

    for comb in product(infill_alg, surrogate_alg, optim_alg):
        used_algorithms = {
            "infill": comb[0],
            "surrogate": comb[1],
            "optimisation": comb[2],
        }

        if mock_mpiexec is not None:
            mock_mpiexec(4, _test_init, used_algorithms=used_algorithms)

        else:
            _test_init(used_algorithms=used_algorithms)


@pytest.mark.unit
def test_sbo_initialisation_invalid_algorithm() -> None:
    """Test that sbo raises an error when an invalid algorithm is provided."""

    def _test_invalid_alg(used_alg: dict) -> None:
        """Helper function to test if an error is raised."""
        config = Config(
            used_algorithms=used_alg,
            lb=[0.0],
            ub=[1.0],
            functions=[lambda x: x**2],
            logger=_make_logger(),
        )
        with pytest.raises(ImportError):
            sbo = SBO(config)  # noqa F841
        Config.destroy_instance()

    for comb in product(
        ["EgoSequential", "some non-existent method"],
        ["KRG", "some non-existent method"],
        ["EI", "some non-existent method"],
    ):
        if comb != ("EgoSequential", "KRG", "EI"):
            used_algorithms = {
                "optimisation": comb[0],
                "surrogate": comb[1],
                "infill": comb[2],
            }
            _test_invalid_alg(used_algorithms)


@pytest.mark.unit
def test_sbo_initialisation_w_initial_samples() -> None:
    """Test that sbo can be initialised with initial samples."""
    initial_samples = [
        {"coord": [0.1], "value": 0.01, "fid": 0},
        {"coord": [0.5], "value": 0.25, "fid": 0},
    ]
    config = Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        logger=_make_logger(),
    )
    sbo = SBO(config, initial_samples=initial_samples)

    assert sbo._surrogate.samples == {
        0: initial_samples
    }, "Initial samples were not set correctly in SBO."


@pytest.mark.unit
def test_sbo_initialisation_wo_initial_samples_single_fid() -> None:
    """Test that sbo can be initialised without initial samples for single-fidelity."""
    config_sf = Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        logger=_make_logger(),
    )
    sbo_sf = SBO(config_sf)
    # Check if n samples are generated
    assert len(sbo_sf._surrogate.samples[0]) == config_sf.get(
        "number_initial_samples"
    ), (
        "Not the correct number of initial samples were generated for "
        "single-fidelity."
    )

    # Check if all samples are HF (fid == 0)
    assert list(sbo_sf._surrogate.samples.keys()) == [
        0
    ], "Initial samples do not have the correct fidelity for single-fidelity."


@pytest.mark.unit
def test_sbo_initialisation_wo_initial_samples_mult_fid_nested() -> None:
    """Test that sbo can be initialised without initial samples for nested multi-fid."""
    config_mf_nested = Config(
        used_algorithms={
            "infill": "GEI",
            "surrogate": "HK",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 0.1],
        settings={"nested": True},
    )
    sbo_mf_nested = SBO(config_mf_nested)
    # Check if all HF samples are included in the LF samples
    assert _nested_samples(sbo_mf_nested._surrogate.samples), (
        "Initial samples are not nested between fidelities for multi-fidelity "
        "nested."
    )

    # Check if the correct number of samples are generated for each fidelity
    num_initial_samples = config_mf_nested.get("number_initial_samples")
    assert len(sbo_mf_nested._surrogate.samples[0]) == num_initial_samples, (
        "Not the correct number of initial samples were generated for HF in "
        "multi-fidelity nested."
    )
    assert len(sbo_mf_nested._surrogate.samples[1]) == 2 * num_initial_samples, (
        "Not the correct number of initial samples were generated for LF in "
        "multi-fidelity nested."
    )

    # Check if only fidelity 0 and 1 are present
    assert set(sbo_mf_nested._surrogate.samples.keys()) == {0, 1}, (
        "Initial samples do not have the correct fidelities for multi-fidelity "
        "nested."
    )


@pytest.mark.unit
def test_sbo_initialisation_wo_initial_samples_mult_fid_non_nested() -> None:
    """Test that sbo can be init without initial samples for non-nested multi-fid."""
    config_mf_non_nested = Config(
        used_algorithms={
            "infill": "GEI",
            "surrogate": "HK",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2, lambda x: x**2 + 0.1],
        settings={"nested": False},
    )
    sbo_mf_non_nested = SBO(config_mf_non_nested)
    # Check if the total number of samples generated matches the expected count
    num_samples = len(sbo_mf_non_nested._surrogate.samples[0]) + len(
        sbo_mf_non_nested._surrogate.samples[1]
    )
    assert num_samples >= config_mf_non_nested.get("number_initial_samples"), (
        "Not enough correct total number of initial samples were generated for "
        "multi-fidelity non-nested."
    )
    assert (
        num_samples <= config_mf_non_nested.get("number_initial_samples") + 2
    ), "Too many initial samples were generated for multi-fidelity non-nested."

    # Check if the samples are not nested between fidelities
    assert not _nested_samples(sbo_mf_non_nested._surrogate.samples), (
        "Initial samples are nested between fidelities for multi-fidelity "
        "non-nested."
    )

    # Check if only fidelity 0 and 1 are present
    assert set(sbo_mf_non_nested._surrogate.samples.keys()) == {0, 1}, (
        "Initial samples do not have the correct fidelities for multi-fidelity "
        "nested."
    )


@pytest.mark.unit
def test_change_used_alg() -> None:
    """Test that changing the used algorithm initialises the new used algorithm.

    Check both change made to config and change made to the sbo instance itself. Both
    changes should be reflected in both places.
    """
    from sbo.infill import EI, Random
    from sbo.optimisation import EgoSequential
    from sbo.surrogate import KRG, RBF

    # Setup initial config
    original_used_alg = {
        "infill": "Random",
        "surrogate": "KRG",
        "optimisation": "EgoSequential",
    }
    config = Config(
        used_algorithms=original_used_alg,
        lb=[0.0],
        ub=[1.0],
        logger=_make_logger(),
        functions=[lambda x: x**2],
    )
    sbo = SBO(config)

    # Test the instances are correct initially
    assert isinstance(sbo._infill, Random)
    assert isinstance(sbo._surrogate, KRG)
    assert isinstance(sbo._optimisation, EgoSequential)

    # Update the instances from the sbo instance
    sbo.update_used_algorithms({"infill": "EI"})
    assert config.used_algorithms == {
        "infill": "EI",
        "surrogate": "KRG",
        "optimisation": "EgoSequential",
    }
    assert isinstance(sbo._infill, EI)

    # Update the instances from the config
    config.update_used_algorithms({"surrogate": "RBF"})

    # Overwrite the optimisation solve method, since sbo has to be called to update
    # change in algorithm
    def mock_solve(*args, **kwargs) -> str:
        """Mock solve method used for testing."""
        return "Called"

    sbo._optimisation.solve = mock_solve

    # Call sbo, to update the instances
    sbo()

    assert config.used_algorithms == {
        "infill": "EI",
        "surrogate": "RBF",
        "optimisation": "EgoSequential",
    }
    assert isinstance(sbo._surrogate, RBF)

    # Last test settings multiple items:
    sbo.update_used_algorithms(original_used_alg)
    assert config.used_algorithms == original_used_alg
    assert isinstance(sbo._optimisation, EgoSequential)
    assert isinstance(sbo._infill, Random)
    assert isinstance(sbo._surrogate, KRG)


@pytest.mark.unit
def test_sbo_run() -> None:
    """Test the run method calls the optimisation run."""
    config = Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
    )
    sbo = SBO(config)

    # override the optimisation run method for testing.
    def mock_solve(*args, **kwargs) -> str:
        return "Called"

    sbo._optimisation.solve = mock_solve

    assert sbo() == "Called"


@pytest.mark.unit
def test_sbo_optimum() -> None:
    """Test the optimum property calls the optimisation run."""
    config = Config(
        used_algorithms={
            "infill": "Random",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        logger=_make_logger(),
    )
    sbo = SBO(config)

    # Set optimum manually for testing
    sbo._optimum = {"coord": [0.0], "value": [0.0], "fid": 0}

    def mock_run(*args, **kwargs) -> dict:
        """Mock run method to return a new optimum."""
        new_point = {"coord": [0.5], "value": [0.25], "fid": 0}
        sbo._optimum = new_point
        return new_point

    sbo.run = mock_run

    # test the set optimum is returned
    assert sbo.optimum == {
        "coord": [0.0],
        "value": [0.0],
        "fid": 0,
    }, "Optimum property did not return the expected value."
    assert sbo() == {
        "coord": [0.0],
        "value": [0.0],
        "fid": 0,
    }, "SBO call did not return the expected optimum from run method."

    # Set the optimum to None to test if the attribute is updated after run
    sbo._optimum = None
    assert sbo.optimum == {
        "coord": [0.5],
        "value": [0.25],
        "fid": 0,
    }, "Optimum property did not update after run method was called."
    assert sbo() == {
        "coord": [0.5],
        "value": [0.25],
        "fid": 0,
    }, "SBO call did not return the expected optimum from run method."


@pytest.mark.unit
def test_hk_single_fidelity_keeps_initial_samples() -> None:
    """Test that HK -> KRG on a single fidelity keeps the user's initial samples."""
    from sbo.surrogate import KRG

    initial_samples = [
        {"coord": [0.1], "value": 0.01, "fid": 0},
        {"coord": [0.5], "value": 0.25, "fid": 0},
        {"coord": [0.9], "value": 0.81, "fid": 0},
    ]
    config = Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": "HK",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        logger=_make_logger(),
    )
    sbo = SBO(config, initial_samples=initial_samples)
    assert isinstance(sbo._surrogate, KRG)

    def mock_solve(*args, **kwargs) -> str:
        return "Called"

    sbo._optimisation.solve = mock_solve
    sbo.run()

    # run() must not rebuild the surrogate and drop the initial samples
    assert isinstance(sbo._surrogate, KRG)
    assert [p["coord"] for p in sbo._surrogate.samples[0]] == [[0.1], [0.5], [0.9]]


@pytest.mark.unit
def test_update_used_algorithms_invalid_key_and_no_aliasing() -> None:
    """Test that an invalid key is ignored and the config does not share the backup."""
    from sbo.infill import EI, Random

    config = Config(
        used_algorithms={
            "infill": "Random",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[0.0],
        ub=[1.0],
        functions=[lambda x: x**2],
        logger=_make_logger(),
    )
    sbo = SBO(config)

    # Invalid key is ignored without raising
    sbo.update_used_algorithms({"invalid": "x", "infill": "EI"})
    assert isinstance(sbo._infill, EI)
    assert config.used_algorithms is not sbo._backup_used_alg

    # A later change to the config is detected at run()
    config.used_algorithms["infill"] = "Random"

    def mock_solve(*args, **kwargs) -> str:
        return "Called"

    sbo._optimisation.solve = mock_solve
    sbo.run()
    assert isinstance(sbo._infill, Random)


def _run_small_sbo(n_dim: int = 1) -> SBO:
    """Run a small single-fidelity optimisation for the plotting tests."""
    config = Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[0.0] * n_dim,
        ub=[1.0] * n_dim,
        functions=[lambda x: np.atleast_1d(np.sum(np.asarray(x) ** 2))],
        settings={"random_seed": 0, "optimisation": {"max_iterations": 1}},
        logger=_make_logger(),
    )
    sbo = SBO(config)
    sbo.run()
    return sbo


@pytest.mark.unit
def test_plot_surrogate_no_iterations_warns(tmp_path: Path) -> None:
    """Test that plotting with no matching iterations warns instead of crashing."""
    from unittest.mock import patch

    sbo = _run_small_sbo()
    with patch.object(sbo.config.logger, "warning") as mock_warning:
        sbo.plot_surrogate("t", filename=str(tmp_path / "p.png"), iterations=[999])

    assert mock_warning.called


@pytest.mark.unit
@pytest.mark.parametrize("n_dim", [1, 2])
def test_plot_surrogate_up_to_str_and_str_iterations(
    tmp_path: Path, n_dim: int
) -> None:
    """Test that up_to as str and iterations as str or int both work."""
    sbo = _run_small_sbo(n_dim)
    samples_before = {
        fid: list(points) for fid, points in sbo._surrogate.samples.items()
    }

    sbo.plot_surrogate("t", filename=str(tmp_path / "a.png"), up_to="2")
    sbo.plot_surrogate("t", filename=str(tmp_path / "b.png"), iterations=["0", 1])

    assert (tmp_path / "a.png").exists()
    assert (tmp_path / "b.png").exists()
    # The surrogate gets its own training data back after plotting
    assert {fid: len(points) for fid, points in sbo._surrogate.samples.items()} == {
        fid: len(points) for fid, points in samples_before.items()
    }


def _quadratic_config(settings: dict) -> Config:
    """Single-fidelity EI/KRG config on [-5, 5] for the stopping tests."""
    return Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": "KRG",
            "optimisation": "EgoSequential",
        },
        lb=[-5.0],
        ub=[5.0],
        functions=[lambda x: np.atleast_1d(np.sum((np.asarray(x) - 1.0) ** 2))],
        settings=settings,
        logger=_make_logger(),
    )


_NO_CONVERGENCE = {
    "convergence_threshold": None,
    "relative_convergence_threshold": None,
    "min_infill_value": None,
}


@pytest.mark.unit
def test_max_iterations_is_exact() -> None:
    """max_iterations=3 runs exactly 3 iterations."""
    config = _quadratic_config(
        {"optimisation": {"max_iterations": 3, **_NO_CONVERGENCE}}
    )
    sbo = SBO(config)
    sbo.run()

    iterations = [k for k in sbo.optimisation_data if isinstance(k, int)]
    assert len(iterations) == 3


@pytest.mark.unit
def test_max_function_calls_hf_includes_final_evaluation() -> None:
    """max_function_calls_HF=4 gives 4 HF calls, including the final evaluation."""
    config = _quadratic_config(
        {
            "optimisation": {
                "max_function_calls_HF": 4,
                "max_iterations": None,
                **_NO_CONVERGENCE,
            }
        }
    )
    sbo = SBO(config)
    sbo.run()

    assert sbo.optimisation_data["total"]["function_calls"][0] == 4
    assert sbo.optimisation_data["total"]["function_calls_HF"] == 4


@pytest.mark.unit
def test_ensure_enough_initial_samples_does_not_change_input() -> None:
    """The initial samples list passed by the user is not extended in place."""
    config = _quadratic_config({})
    initial_samples = [{"coord": np.array([1.0]), "value": 0.0, "fid": 0}]

    sbo = SBO(config, initial_samples=initial_samples)

    assert len(initial_samples) == 1
    assert len(sbo._surrogate.samples[0]) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "surrogate, functions",
    [
        ("KRG", [lambda x: float(np.sum((np.asarray(x) - 1.0) ** 2))]),
        (
            "HK",
            [
                lambda x: float(np.sum((np.asarray(x) - 1.0) ** 2)),
                lambda x: float(np.sum((np.asarray(x) - 1.2) ** 2) + 0.1),
            ],
        ),
    ],
)
def test_run_with_float_returning_functions(surrogate: str, functions: list) -> None:
    """Test a full run with objective functions that return a plain float."""
    config = Config(
        used_algorithms={
            "infill": "EI",
            "surrogate": surrogate,
            "optimisation": "EgoSequential",
        },
        lb=[-5.0],
        ub=[5.0],
        functions=functions,
        settings={"optimisation": {"max_iterations": 2}},
        logger=_make_logger(),
    )
    sbo = SBO(config)
    optimum = sbo.run()

    assert isinstance(optimum["value"], float)
    assert np.asarray(optimum["coord"]).shape == (1,)
    for points in sbo._surrogate.samples.values():
        for p in points:
            assert isinstance(p["value"], float)
