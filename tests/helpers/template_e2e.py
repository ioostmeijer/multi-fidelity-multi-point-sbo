import pytest
from logger_helper import _make_logger

from sbo import SBO, Config
from sbo.benchmarks import FORRESTER

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
        functions = [FORRESTER.functions[0]]
    else:
        functions = FORRESTER.functions

    config = Config(
        used_algorithms=used_algorithms,
        lb=FORRESTER.l_bound,
        ub=FORRESTER.u_bound,
        functions=functions,
        settings=settings,
        logger=_make_logger(),
    )
    sbo = SBO(config)

    try:
        found_optimum = sbo()

        if mpi_comm is not None:
            mpi_comm.Barrier()

        # Only the coordinator (rank 0) returns the optimum; the workers return an
        # empty dict.
        if mpi_comm is None or mpi_comm.Get_rank() == 0:
            _check_optimum(found_optimum)
        else:
            assert found_optimum == {}, (
                f"A worker rank should return an empty dict, got {found_optimum}."
            )
    finally:
        # Always clean up and synchronise, also when an assertion fails: otherwise the
        # other ranks wait at the next collective call forever (deadlock).
        Config.destroy_instance()

        if mpi_comm is not None:
            mpi_comm.Barrier()


def _check_optimum(found_optimum: dict) -> None:
    """Check the format of the optimum and that it is the Forrester optimum."""
    assert isinstance(found_optimum, dict), (
        f"Optimum should be a dict, " f"got {type(found_optimum)}."
    )
    assert all(key in found_optimum for key in ["coord", "value"]), (
        f"Optimum should have 'coord' and 'value' keys, "
        f"while it has: {list(found_optimum)}."
    )

    # Known optimum of the high-fidelity Forrester function (x ~ 0.757, f ~ -6.021)
    coord_expected = FORRESTER.x_opt
    func_expected = float(FORRESTER.functions[0](coord_expected.reshape(1, -1))[0])

    # Absolute tolerance for the final optimum (value and coordinates)
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
