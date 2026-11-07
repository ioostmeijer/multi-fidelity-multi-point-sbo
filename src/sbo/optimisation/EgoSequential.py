import copy
import time

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None

from ..infill import InfillBase
from ..surrogate import SurrogateBase
from .Base import OptimisationBase


class EgoSequential(OptimisationBase):
    """Class to perform Efficient Global Optimisation (EGO) in a sequential manner.

    This class implements the EGO algorithm, which uses a surrogate model and an infill
    strategy to iteratively find the optimal parameters that minimise a high-fidelity
    objective function. The optimisation process continues until a stopping criterion is
    met. The new samples are evaluated one by one, and the surrogate model is updated
    with the new samples after each iteration.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the EgoSequential class."""
        super().__init__(*args, **kwargs)

    def solve(
        self, surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs
    ) -> dict:
        """Run the EGO Sequential Optimisation algorithm.

        This method is based on the method implemented in SMT. It uses a surrogate model
        and a specified infill strategy to find the optimal parameters that minimise
        the (HF) objective function.
        As stopping criteria, it uses:
        - The max number of iterations specified in the settings.
        - The difference between the best value found in the last iteration
          and the best value found in the previous iteration, which must be less than
          a specified tolerance.

        Parameters
        ----------
        surrogate
            The surrogate model used to approximate the objective function.
        infill
            The infill strategy used to select new points to evaluate.

        Returns
        -------
        dict
            The best point found, as a dictionary with keys "value", "coord", and "fid".
        """
        if MPI is not None and MPI.COMM_WORLD.Get_rank() != 0:
            self.config.logger.info(
                "EGO Sequential Optimisation is only run on the root process."
            )
            return None

        # Save initial state to optimisation data
        self._optimisation_data["init"] = {
            "best_coord": None,
            "best_value": None,
            "infill_value": None,
            "function_calls": None,
            "function_calls_HF": None,
            "sample_points": copy.deepcopy(surrogate.samples),
            "time_elapsed": None,
            "surrogate_optimum": None,
        }

        # Setup counters and timers
        iteration = 0
        best_point = {"value": float("inf"), "coord": None, "fid": None}
        start_time = time.time()
        function_calls = dict.fromkeys(range(self.config.num_fidelities), 0)
        prev_surr_opt = {"value": float("inf"), "coord": None, "fid": None}

        # Start optimisation loop
        self.config.logger.info("Starting EGO Sequential Optimisation")
        while True:
            self.config.logger.info(f"Iteration {iteration} started")

            # Setup timer for the iteration and save previous best value
            iter_start_time = time.time()
            prev_best = best_point

            # get new points to evaluate
            new_points, infill_value = infill(surrogate)

            # Check if new points not already sampled, if so remove from tasks:
            new_points = self._remove_duplicates(new_points, surrogate)

            # Evaluate the new points sequentially
            for task in new_points:
                for point in task:
                    self.evaluate_point(point, inplace=True)
                    function_calls[point["fid"]] += 1

            # Flatten points list and add to the surrogate model
            evaluated_points = [point for task in new_points for point in task]
            surrogate.add_training_data(evaluated_points)
            surrogate.train()

            # Get the current best value
            best_point = min(surrogate.samples[0], key=lambda p: p["value"])
            curr_surr_opt = {**surrogate.get_optimum(fidelity=0), "fid": 0}

            # Save the data and calculate stopping criteria
            stop = self._run_end_iteration(
                iteration=iteration,
                function_calls=function_calls,
                best_val=best_point["value"],
                prev_best_val=prev_best["value"],
                infill_value=infill_value,
                best_coord=best_point["coord"],
                surrogate=surrogate,
                start_time_iteration=iter_start_time,
                current_surrogate_optimum=curr_surr_opt,
                prev_surrogate_optimum=prev_surr_opt,
            )

            # if stopping criteria met, break the loop
            if stop:
                break

            # Update iteration counter
            self.config.logger.info(f"Iteration {iteration} finished")
            iteration += 1

            # Update the prev best points
            prev_surr_opt = curr_surr_opt

        # Calculate the best point, add to dict
        best_point = min(surrogate.samples[0], key=lambda p: p["value"])
        surr_opt = {**surrogate.get_optimum(), "fid": 0}
        surr_opt = self.evaluate_point(surr_opt, inplace=False, bypass_delay=True)
        self._count_final_evaluation(function_calls)
        if surr_opt["value"] < best_point["value"]:
            best_point = surr_opt
        self._optimisation_data.setdefault("solution", None)
        self._optimisation_data["solution"] = best_point
        self._optimisation_data["total"]["best_coord"] = best_point["coord"]
        self._optimisation_data["total"]["best_value"] = best_point["value"]
        self._optimisation_data[iteration]["best_coord"] = best_point["coord"]
        self._optimisation_data[iteration]["best_value"] = best_point["value"]

        # compute total time taken
        total_time = time.time() - start_time
        self._optimisation_data["total"]["time_elapsed"] = total_time
        self.config.logger.info(
            "EGO Sequential Optimisation completed in " f"{total_time:.2f} seconds."
        )

        # return the best point
        return best_point
