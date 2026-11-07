import numpy as np
import pytest
from logger_helper import _make_logger
from mf2 import bi_fidelity_functions
from smt.sampling_methods import LHS

from sbo import Config
from sbo.infill import EI
from sbo.optimisation import EgoAsync
from sbo.surrogate import KRG


@pytest.mark.verification
@pytest.mark.mpi
def test_verification_ego_async() -> None:
    """Verify the implementation of EGO async strategy.

    The idea is to test all functions given in mf2 with a KRG surrogate and EI infill
    criterion. It is supposed that if the optimisation algorithm reaches a similar
    optimum it is correctly implemented. The reference optima are calculated using SMT's
    implementation of EGO with KRG and EI and a separate solution from `scipy.optimize`.
    This also is based on the assumption that KRG and EI are correctly implemented. This
    is tested in other unit and verification tests.
    """
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError):
        pytest.skip("MPI not available", allow_module_level=True)
        MPI = None

    report = ""

    # Select all functions except Hartmann6 and Borehole. These functions are too
    # expensive to continuously be testing. The EGO async has been shown to be able
    # to solve these problems.
    for selected_functions in bi_fidelity_functions[:-2]:
        if selected_functions.name == "Currin":
            function = (
                lambda x, selected_functions=selected_functions: -1
                * selected_functions[0](x)
            )  # Maximisation problem noqa: E731
        else:
            function = selected_functions.functions[0]

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
                settings={
                    "number_new_points": 1,
                    "random_seed": random_seed,
                    "function_delay": None,
                },
            )

            # Instantiate the EgoAsync, KRG and EI classes
            surrogate = KRG(config)
            infill = EI(config)
            ego_async = EgoAsync(config)

            # Get initial samples using LHS
            lhs = LHS(xlimits=config.bounds)
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
            solution_ego_async = ego_async(surrogate, infill)

            # Update the best solution
            if (
                solution_ego_async is not None
                and solution_ego_async["value"] < optimum_solution["value"]
            ):
                optimum_solution = solution_ego_async.copy()

            if MPI is not None:
                MPI.COMM_WORLD.Barrier()

            # Delete the config to set up for the next test
            if MPI is not None:
                Config.destroy_instance()

        # Get the known optimum
        known_opt_x = selected_functions.x_opt
        known_opt_y = function(known_opt_x)

        # Assert the found optimum and solution are close.
        if MPI is not None and MPI.COMM_WORLD.Get_rank() == 0:
            assert np.isclose(
                optimum_solution["value"], known_opt_y, atol=1e-3, rtol=0
            ), (
                "The found solution and the known optimum are not equal for function: "
                f"{selected_functions.name}.\nknown optimum: {known_opt_y}\n"
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

        if MPI is not None:
            MPI.COMM_WORLD.Barrier()

    print(report)
