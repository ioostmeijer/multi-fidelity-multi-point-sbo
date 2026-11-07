import pytest
from logger_helper import _make_logger
from mf2 import himmelblau

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

    used_algorithms: dict
        Dictionary specifying the algorithms to be used in the optimisation.
    settings: dict
        Dictionary specifying the settings for the optimisation.
    single_fid: bool
        If True, tests single-fidelity optimisation; otherwise, multi-fidelity.
    :return: None
    """
    if single_fid:
        functions = [himmelblau.functions[0]]
    else:
        functions = himmelblau.functions

    config = Config(
        used_algorithms=used_algorithms,
        lb=himmelblau.l_bound,
        ub=himmelblau.u_bound,
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

    # Check if the found optimum is close to the known optimum
    func_expected = 0.0
    coord_expected = [
        [3, 2],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ]

    if mpi_comm is not None:
        mpi_comm.Barrier()

    # Absolute tolerance for the final optimum (value and coordinates)
    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        assert abs(found_optimum["value"] - func_expected) < RESULT_TOLERANCE, (
            f"Found optimum value {found_optimum['value']} differs from "
            f"expected {func_expected} by more than {RESULT_TOLERANCE}."
        )
        assert any(
            all(
                abs(a - b) < RESULT_TOLERANCE
                for a, b in zip(found_optimum["coord"], ce)
            )
            for ce in coord_expected
        ), (
            f"Found optimum coord {found_optimum['coord']} differs from "
            f"expected coords {coord_expected} by more than {RESULT_TOLERANCE}."
        )

    # Delete config
    Config.destroy_instance()

    if mpi_comm is not None:
        mpi_comm.Barrier()
