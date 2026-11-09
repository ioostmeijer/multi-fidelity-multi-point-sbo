import numpy as np
import pytest
from logger_helper import _make_logger
from smt.sampling_methods import LHS

from sbo import Config
from sbo.benchmarks import ALL_BENCHMARKS
from sbo.infill import EI
from sbo.optimisation import EgoMPI
from sbo.surrogate import KRG


@pytest.mark.verification
@pytest.mark.mpi
def test_verification_ego_mpi() -> None:
    """Verify the implementation of EGO mpi strategy.

    The idea is to test all benchmark problems in sbo.benchmarks with a KRG surrogate
    and EI infill criterion. It is supposed that if the optimisation algorithm reaches
    a similar optimum it is correctly implemented. The reference optima are calculated
    using SMT's implementation of EGO with KRG and EI and a separate solution from
    `scipy.optimize`. This also is based on the assumption that KRG and EI are
    correctly implemented. This is tested in other unit and verification tests.
    """
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError):
        pytest.skip("MPI not available", allow_module_level=True)
        MPI = None

    report = ""
    failures = []
    for selected_functions in ALL_BENCHMARKS:
        if selected_functions.name == "currin":
            function = (
                lambda x, selected_functions=selected_functions: -1
                * selected_functions.functions[0](x)
            )  # Maximisation problem noqa: E731
        else:
            function = selected_functions.functions[0]

        # Get the known optimum
        known_opt_x = selected_functions.x_opt
        known_opt_y = function(known_opt_x)

        optimum_solution = {"value": float("inf")}
        # Run for several random seeds to overcome possible bad initial state
        for random_seed in range(10):
            used_algorithms = {
                "optimisation": "EgoSequential",
                "surrogate": "KRG",
                "infill": "EI",
            }
            config = Config(
                used_algorithms=used_algorithms,
                functions=[function],
                lb=selected_functions.l_bound,
                ub=selected_functions.u_bound,
                logger=_make_logger(),
                settings={"number_new_points": 1, "random_seed": random_seed},
            )

            # Instantiate the EgoAsync, KRG and EI classes
            surrogate = KRG(config)
            infill = EI(config)
            ego_mpi = EgoMPI(config)

            # Get initial samples using LHS
            lhs = LHS(xlimits=config.bounds, seed=random_seed)
            x_initial = lhs(5 * config.n_dimensions)
            y_initial = function(x_initial)

            # Set training data for surrogate and train
            training_data = [
                {"coord": x_initial[i], "value": y_initial[i], "fid": 0}
                for i in range(len(x_initial))
            ]
            surrogate.set_training_data(training_data)
            surrogate.train()

            # Perform optimisation
            solution_ego_mpi = ego_mpi(surrogate, infill)

            # Update the best solution
            if (
                solution_ego_mpi is not None
                and solution_ego_mpi["value"] < optimum_solution["value"]
            ):
                optimum_solution = solution_ego_mpi.copy()

            if MPI is not None:
                MPI.COMM_WORLD.Barrier()

            # Delete the config on every rank (it is a per-process singleton) to set
            # up for the next seed
            Config.destroy_instance()

            # Stop once a seed reaches the known optimum: later seeds cannot improve
            # the best solution. Rank 0 decides and broadcasts the decision, so all
            # ranks leave the loop together.
            if MPI is not None:
                converged = None
                if MPI.COMM_WORLD.Get_rank() == 0:
                    converged = bool(
                        np.isclose(
                            optimum_solution["value"], known_opt_y, atol=1e-3, rtol=0
                        )
                    )
                if MPI.COMM_WORLD.bcast(converged, root=0):
                    break

        # Assert the found optimum and solution are close.
        if MPI is not None and MPI.COMM_WORLD.Get_rank() == 0:
            try:
                assert np.isclose(
                    optimum_solution["value"], known_opt_y, atol=1e-3, rtol=0
                ), (
                    "The found solution and the known optimum are not equal for "
                    f"function: {selected_functions.name}.\n"
                    f"known optimum: {known_opt_y}\n"
                    f"Found optimum: {optimum_solution['value']}"
                    f"\nThe difference: {known_opt_y - optimum_solution['value']}\n"
                    f"Relative difference: "
                    f"{(known_opt_y - optimum_solution['value']) / known_opt_y}"
                )

                # Assert that the found optimum is never better than the actual optimum
                assert optimum_solution["value"] >= known_opt_y, (
                    "The optimum is better than the known optimum for function: "
                    f"{selected_functions.name}. \n"
                    f"Known optimum: {optimum_solution['value']}\nFound optimum: "
                    f"{known_opt_y}"
                )

                # Print the absolute and relative difference
                diff = known_opt_y - optimum_solution["value"]
                rel_diff = diff / optimum_solution["value"]
                report += (
                    "-" * 50 + f"\nFunction: {selected_functions.name}\n"
                    f"Absolute difference: {diff}\n"
                    f"Relative difference: {rel_diff}\n"
                    f"Found optimum: {optimum_solution['value']}\n"
                    f"Known optimum: {known_opt_y}\n"
                )
            except AssertionError as error:
                failures.append(str(error))

        if MPI is not None:
            MPI.COMM_WORLD.Barrier()

    print(report)

    assert not failures, "\n\n".join(failures)
