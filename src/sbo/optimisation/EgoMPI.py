import copy
from time import time

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None

from ..infill import InfillBase
from ..surrogate import SurrogateBase
from ..utils import Config
from .Base import OptimisationBase
from .EgoSequential import EgoSequential


class EgoMPI(OptimisationBase):
    """Class implementing EGO algorithm using MPI.

    This class implements the EGO algorithm using MPI. The sample locations per
    iteration are divided among all the workers, which will each calculate the points
    allocated. The results are gathered on the coordinator rank and added to the
    surrogate. Note that this process is sequential, thus all results for one iteration
    have to be calculated before the process is advanced.

    Attributes
    ----------
    mpi_comm : MPI.Comm
        The MPI communicator used for communication between the coordinator and worker
        processes.
    mpi_status : MPI.Status
        The MPI status object used for receiving messages.
    mpi_rank : int
        The rank of the current MPI process.
    mpi_size : int
        The total number of MPI processes.
    All OptimisationBase attributes.
    """

    def __new__(
        cls, *args, mpi_comm: "MPI.Comm | None" = None, **kwargs
    ) -> "EgoMPI | EgoSequential":
        """Create an instance of the EgoMPI or EgoSequential class.

        This method returns an instance of the EgoMPI or EgoSequential class. If MPI and
        more than 1 worker is available it returns an instance of EgoMPI. Else the
        algorithm used is equal to the one used in EgoSequential and it will return an
        instance of the EgoSequential class.

        Parameters
        ----------
        mpi_comm
            The MPI communicator to use. If None, MPI.COMM_WORLD is used.

        Returns
        -------
        EgoMPI | EgoSequential
            An instance of the EgoMPI or EgoSequential class.

        Raises
        ------
        ValueError
            If no config is passed.
        """
        # Get config from kwargs or args
        config = kwargs.get("config", None)
        if config is None:
            for a in args:
                if isinstance(a, Config):
                    config = a
                    break
        if config is None:
            raise ValueError("Config object must be provided to EgoAsync.")
        try:
            mpi_comm = mpi_comm if mpi_comm else MPI.COMM_WORLD

            if mpi_comm.Get_size() < 2:
                config.logger.info(
                    "EgoMPI, does not change compared to EgoSequential for < 2 mpi "
                    "processes. Continuing with EgoSequential."
                )
                return EgoSequential(*args, **kwargs)

            return super(EgoMPI, cls).__new__(cls)

        except (ImportError, AttributeError):
            config.logger.info(
                "EgoMPI, requires mpi4py and MPI, continuing with EgoSequential."
            )
            return EgoSequential(*args, **kwargs)

    def __init__(self, *args, mpi_comm: "MPI.Comm | None" = None, **kwargs) -> None:
        """Initialise the EgoMPI class.

        Parameters
        ----------
        mpi_comm
            The MPI communicator to use. If None, MPI.COMM_WORLD is used.
        """
        super().__init__(*args, **kwargs)

        self.mpi_comm = mpi_comm if mpi_comm else MPI.COMM_WORLD
        self.mpi_status = MPI.Status()
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

    def solve(
        self, surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs
    ) -> dict | None:
        """Run the MPI EGO Optimisation algorithm.

        Parameters
        ----------
        surrogate
            Surrogate model to use for optimisation.
        infill
            Infill strategy to use for selecting new points.

        Returns
        -------
        dict | None
            The best point found on the coordinator process, None on worker processes.

        Raises
        ------
        ValueError
            If fewer than 2 MPI processes are available (EgoMPI needs a coordinator and
            at least one worker).
        """
        # Test if enough MPI processes are available
        if self.mpi_size < 2:
            msg = "EgoMPI requires at least 2 MPI processes."
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Solve as coordinator or worker based on rank
        if self.mpi_rank == 0:
            return self._solve_coordinator(surrogate, infill)
        return self._solve_worker()

    def _solve_coordinator(self, surrogate: SurrogateBase, infill: InfillBase) -> dict:
        """Solve the coordinator process.

        The coordinator process first selects the new sample locations based on the
        passed infill algorithm and surrogate. These tasks are sent to all workers. The
        workers and the coordinator process calculate the results for every point, after
        which the coordinator receives all results and updates the surrogate. This is
        repeated until one of the stopping criteria is met, when the optimisation will
        stop and the optimum is returned. Note that this process is sequential, thus all
        results for one iteration have to be calculated before the process is advanced.

        Parameters
        ----------
        surrogate
            Surrogate model to use for optimisation.
        infill
            Infill strategy to use for selecting new points.

        Returns
        -------
        dict
            The best point found: the best HF sample, or the evaluated surrogate optimum
            if that is better ({'coord': np.ndarray, 'value': float, 'fid': 0, ...}).
        """
        # Check if the number of new points exceeds the rank size
        if self.config["number_new_points"] > self.mpi_size:
            self.config.logger.warning(
                f"Number of new points exceeds the possible number of workers. Setting"
                f" to number of workers (was {self.config['number_new_points']}, "
                f"set to {self.mpi_size})."
            )
            self.update_number_of_points(self.mpi_size)

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

        # setup counters
        iteration = 0
        best_point = {"coord": None, "value": float("inf")}
        curr_surr_opt = {"coord": None, "value": float("inf")}
        start_time = time()
        function_calls = dict.fromkeys(range(self.config.num_fidelities), 0)

        # Start optimisation
        self.config.logger.info("Starting EGO MPI optimisation loop.")

        while True:
            iteration_time = time()
            prev_best = best_point
            prev_surr_opt = curr_surr_opt
            self.config.logger.info(f"Starting iteration {iteration}")

            # Select new points
            tasks, infill_value = infill(surrogate)

            # Add new points up to the maximum number of new points per iteration or the
            # mpi size
            multi_point_alg = self.config.get("optimisation.multipoint_algorithm", None)
            while len(tasks) < min(self.mpi_size, self.config["number_new_points"]):
                pending_points = [point for task in tasks for point in task]
                if multi_point_alg == "CL":
                    new_tasks, _ = self._constant_liar(
                        surrogate, infill, pending_points
                    )
                else:
                    if multi_point_alg != "KB":
                        msg = (
                            f"Unknown multi-point algorithm: {multi_point_alg}, "
                            f"using KB."
                        )
                        self.config.logger.warning(msg)

                    new_tasks, _ = self._kriging_believer(
                        surrogate, infill, pending_points
                    )
                tasks += new_tasks

            # If more tasks than number of points or ranks, cut off extra tasks
            if len(tasks) > min(self.config["number_new_points"], self.mpi_size):
                tasks = tasks[: self.config["number_new_points"]]

            # Check if the points are not already sampled
            tasks = self._remove_duplicates(tasks, surrogate)

            # Get tasks in correct format for MPI.scatter (ensure each worker has list)
            if len(tasks) < self.mpi_size:
                tasks += [[] for _ in range(self.mpi_size - len(tasks))]

            # Divide tasks between workers
            self.config.logger.debug("Distributing tasks.")
            task = self.mpi_comm.scatter(tasks, root=0)

            # Calculate own task
            result = [self.evaluate_point(point, inplace=False) for point in task]

            # Gather results
            results = self.mpi_comm.gather(result, root=0)
            results = [point for task in results for point in task]
            self.config.logger.debug("Received results from worker processes.")

            # Add results to the surrogate
            surrogate.add_training_data(results)
            surrogate.train()

            # Get best point from new surrogate
            best_point = min(surrogate.samples[0], key=lambda p: p["value"])
            curr_surr_opt = {**surrogate.get_optimum(fidelity=0), "fid": 0}

            # Update function calls
            for point in results:
                fid_point = int(point["fid"])
                function_calls[fid_point] = function_calls.get(fid_point, 0) + 1

            # Finish iteration and calculate stopping criteria
            self.config.logger.info(f"Finished iteration {iteration}")
            stop = self._run_end_iteration(
                iteration=iteration,
                function_calls=function_calls,
                best_val=best_point["value"],
                prev_best_val=prev_best["value"],
                infill_value=infill_value,
                best_coord=best_point["coord"],
                surrogate=surrogate,
                start_time_iteration=iteration_time,
                current_surrogate_optimum=curr_surr_opt,
                prev_surrogate_optimum=prev_surr_opt,
            )

            # Stop optimisation if finished
            if stop:
                self.mpi_comm.scatter(["Stop" for _ in range(self.mpi_size)], root=0)
                break

            # Increment iteration number
            iteration += 1

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

        # Compute total time taken and update info
        total_time = time() - start_time
        self._optimisation_data["total"]["time_elapsed"] = total_time
        self.config.logger.info(f"Finished optimisation in {total_time:.2f} seconds.")

        # return the optimum
        return best_point

    def _solve_worker(self) -> None:
        """Solve the MPI EGO as worker.

        The worker receives tasks from the coordinator rank, calculates their values and
        returns the results.
        """
        while True:
            task = None

            # receive new points
            task = self.mpi_comm.scatter(task, root=0)

            # Check if stop task and stop
            if task == "Stop":
                self.config.logger.debug(f"Worker {self.mpi_rank} has exited.")
                break

            # Calculate results
            self.config.logger.debug(
                f"Worker {self.mpi_rank} starting calculations for "
                f"{len(task)} points."
            )
            result = [self.evaluate_point(point, inplace=False) for point in task]
            self.config.logger.debug(f"Worker {self.mpi_rank} finished calculations.")

            # Send the results
            self.mpi_comm.gather(result, root=0)

        return
