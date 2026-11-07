import pytest
from template_e2e import template_for_e2e_test

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


@pytest.mark.e2e
def test_e2e_single_function_non_mpi() -> None:
    """Test end-to-end optimisation on a single fidelity function, not using MPI."""
    for infill_strategy in ["EI", "GEI"]:
        used_algorithms = {
            "optimisation": "EgoSequential",
            "surrogate": "KRG",
            "infill": infill_strategy,
        }
        settings = {"number_new_points": 1}
        template_for_e2e_test(used_algorithms, settings, single_fid=True)


@pytest.mark.e2e
@pytest.mark.mpi(min_size=2)
def test_e2e_single_function_mpi() -> None:
    """Test end-to-end optimisation on a single fidelity function, using MPI."""
    if MPI is None:
        pytest.skip("mpi4py is not available")

    for optimisation_algorithm in ["EgoAsync", "EgoMPI"]:
        for infill_strategy in ["EI", "GEI"]:
            used_algorithms = {
                "optimisation": optimisation_algorithm,
                "surrogate": "KRG",
                "infill": infill_strategy,
            }
            template_for_e2e_test(
                used_algorithms, {}, single_fid=True, mpi_comm=MPI.COMM_WORLD
            )
