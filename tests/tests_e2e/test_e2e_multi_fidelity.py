import pytest
from template_e2e import template_for_e2e_test


@pytest.mark.e2e
def test_e2e_multi_fidelity_non_mpi() -> None:
    """Test end-to-end optimisation on a multi-fidelity function, not using MPI."""
    for infill_strategy in ["TwoStep"]:
        used_algorithms = {
            "optimisation": "EgoSequential",
            "surrogate": "HK",
            "infill": infill_strategy,
        }
        settings = {"number_new_points": 1}
        template_for_e2e_test(used_algorithms, settings, single_fid=False)


@pytest.mark.e2e
@pytest.mark.mpi
def test_e2e_multi_fidelity_mpi() -> None:
    """Test end-to-end optimisation on a single fidelity function, using MPI."""
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError):
        pytest.skip("mpi4py is not available")
        MPI = None

    if MPI is None or MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip("This test requires at least 2 MPI processes.")

    for optimisation_algorithm in ["EgoAsync", "EgoMPI"]:
        for infill_strategy in ["TwoStep"]:
            used_algorithms = {
                "optimisation": optimisation_algorithm,
                "surrogate": "HK",
                "infill": infill_strategy,
            }
            template_for_e2e_test(
                used_algorithms, {}, single_fid=False, mpi_comm=MPI.COMM_WORLD
            )
