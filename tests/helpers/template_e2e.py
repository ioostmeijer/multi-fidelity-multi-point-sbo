import pytest
from logger_helper import _make_logger
from mf2 import forrester

from sbo import SBO, Config

# Absolute tolerance on the final optimum found by the end-to-end tests
RESULT_TOLERANCE = 1e-2


@pytest.mark.skip("Template function, not a test case.")
def template_for_e2e_test(
    used_algorithms: dict,
    settings: dict,
    single_fid: bool = False,
    mpi_comm: "MPI.Comm | None" = None,  # noqa: F821
) -> None:
    """Template used in end-to-end optimisation tests.

    The end-to-end tests check that a full optimisation runs and returns a solution, not
    the accuracy of the algorithms (that is covered by the verification tests). They
    therefore use the cheap, one-dimensional Forrester function, whose single global
    optimum is easy to reach.

    used_algorithms: dict
        Dictionary specifying the algorithms to be used in the optimisation.
    settings: dict
        Dictionary specifying the settings for the optimisation.
    single_fid: bool
        If True, tests single-fidelity optimisation; otherwise, multi-fidelity.
    :return: None
    """
    if single_fid:
        functions = [forrester.functions[0]]
    else:
        functions = forrester.functions

    config = Config(
        used_algorithms=used_algorithms,
        lb=forrester.l_bound,
        ub=forrester.u_bound,
        functions=functions,
        settings=settings,
        logger=_make_logger(),
    )
    sbo = SBO(config)
    found_optimum = sbo()

    # Check if the found optimum is of the correct format
    assert isinstance(found_optimum, dict), (
        f"Optimum should be a dict, " f"got {type(found_optimum)}."
    )
    assert all(key in found_optimum for key in ["coord", "value"]), (
        f"Optimum should have 'coord' and 'value' keys, "
        f"while it has: {list(found_optimum)}."
    )

    # Known optimum of the high-fidelity Forrester function (x ~ 0.757, f ~ -6.021)
    coord_expected = forrester.x_opt
    func_expected = float(forrester.functions[0](coord_expected.reshape(1, -1))[0])

    if mpi_comm is not None:
        mpi_comm.Barrier()

    # Absolute tolerance for the final optimum (value and coordinates)
    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        assert abs(found_optimum["value"] - func_expected) < RESULT_TOLERANCE, (
            f"Found optimum value {found_optimum['value']} differs from "
            f"expected {func_expected} by more than {RESULT_TOLERANCE}."
        )
        assert all(
            abs(a - b) < RESULT_TOLERANCE
            for a, b in zip(found_optimum["coord"], coord_expected)
        ), (
            f"Found optimum coord {found_optimum['coord']} differs from "
            f"expected coord {coord_expected} by more than {RESULT_TOLERANCE}."
        )

    # Delete config
    Config.destroy_instance()

    if mpi_comm is not None:
        mpi_comm.Barrier()
