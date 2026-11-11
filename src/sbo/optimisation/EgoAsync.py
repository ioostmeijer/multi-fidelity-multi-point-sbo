import copy
import time

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None

from ..infill import InfillBase
from ..surrogate import SurrogateBase
from ..utils import Config
from .Base import OptimisationBase
from .EgoSequential import EgoSequential


class EgoAsync(OptimisationBase):
    """Class implementing Asynchronous Efficient Global Optimisation (EGO).

    This class implements the Asynchronous EGO algorithm, which samples new points based
    on a specified infill strategy. These points are selected on the coordinator process
    and the generated task is sent to an available worker process for evaluation. Once
    the worker process has completed the evaluation, the results are sent back to the
    coordinator process, which updates the surrogate model with the new data. The
    optimisation continues until a stopping criterion is met.
    If the surrogate is not updated a new point can not be selected since it will select
    a pending point. To resolve this issue, either the kriging Believer (KB) or Constant
    Liar (CL) multi-point algorithm is used to generate new points, using the existing
    surrogate and pending points.

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
    """

    def __new__(
        cls, *args, mpi_comm: "MPI.Comm | None" = None, **kwargs
    ) -> "EgoAsync | EgoSequential":
        """Create a new instance of EgoAsync or fallback to EgoSequential.

        Create a new instance of EgoAsync if there are at least 2 MPI processes
        available. If not, log an info message and return an instance of EgoSequential.

        Parameters
        ----------
        mpi_comm
            The MPI communicator to use. If None, MPI.COMM_WORLD is used.

        Returns
        -------
        EgoAsync | EgoSequential
            An instance of EgoAsync or EgoSequential

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
                    "EgoAsync requires at least 2 MPI processes. Continuing with "
                    "EgoSequential."
                )
                return EgoSequential(*args, **kwargs)

            return super(EgoAsync, cls).__new__(cls)

        except (ImportError, AttributeError):
            config.logger.info(
                "EgoAsync requires mpi4py and MPI, continuing with EgoSequential."
            )
            return EgoSequential(*args, **kwargs)

    def __init__(self, *args, mpi_comm: "MPI.Comm | None" = None, **kwargs) -> None:
        """Initialise EgoAsync optimiser.

        All arguments required by OptimisationBase should be provided.

        Parameters
        ----------
        mpi_comm
            MPI communicator to use. If None, MPI.COMM_WORLD is used.
        """
        super().__init__(*args, **kwargs)

        self.mpi_comm = mpi_comm if mpi_comm is not None else MPI.COMM_WORLD
        self.mpi_status = MPI.Status()
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        self.config.logger.debug(
            f"EgoAsync initialised on MPI rank {self.mpi_rank} "
            f"of {self.mpi_size} processes."
        )

    def solve(
        self, surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs
    ) -> dict | None:
        """Run the Asynchronous EGO Optimisation algorithm.

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
            If fewer than 2 MPI processes are available (EgoAsync needs a coordinator
            and at least one worker).
        """
        # Test if enough MPI processes are available
        if self.mpi_size < 2:
            msg = "EgoAsync requires at least 2 MPI processes."
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Solve as coordinator or worker based on MPI rank
        if self.mpi_rank == 0:
            optimum = self._solve_coordinator(surrogate, infill)
        else:
            optimum = self._solve_worker()
        return optimum

    def _solve_coordinator(self, surrogate: SurrogateBase, infill: InfillBase) -> dict:
        """Run the coordinator process of the Asynchronous EGO Optimisation algorithm.

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

        # Setup catalogue of pending points and free workers
        pending_points = []
        # 0 is the coordinator and does not evaluate points, due to a limitation of the
        # Python GIL
        free_workers = list(range(1, self.mpi_size))

        # Setup solution data
        best_point = {"coord": None, "value": float("inf")}
        curr_surr_opt = {"coord": None, "value": float("inf")}
        prev_surr_opt = {"coord": None, "value": float("inf")}
        infill_value = float("inf")
        function_calls = dict.fromkeys(range(self.config.num_fidelities), 0)

        # Flag to keep track of updated surrogate
        surrogate_updated = True

        # Get starting time
        start_time = time.time()

        # Counter for the received results, used as the (str) iteration key.
        # It is zero-padded so that sorting the keys keeps the arrival order.
        result_counter = 0

        # Generate a set of all stopped workers
        stopped_workers = set()

        # Start the optimisation
        self.config.logger.info("Starting asynchronous EGO optimisation.")
        while True:
            # Check if there are workers free
            if free_workers:
                # Create and send task(s) to free worker(s)
                infill_value = self._add_tasks_to_workers(
                    free_workers,
                    pending_points,
                    surrogate_updated,
                    surrogate,
                    infill,
                )

                # Update the new surrogate flag
                surrogate_updated = False

            # Check for incoming results from workers
            if self.mpi_comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=22, status=self.mpi_status
            ):
                # Add worker to free workers
                worker_rank = self.mpi_status.Get_source()
                free_workers.append(worker_rank)

                # Get results from worker
                results = self.mpi_comm.recv(
                    source=worker_rank, tag=22, status=self.mpi_status
                )
                self.config.logger.debug(
                    f"Coordinator received results {results} from worker {worker_rank}"
                )

                if results == "STOPPED":
                    stopped_workers.add(worker_rank)
                    self.config.logger.debug(
                        f"Worker {worker_rank} has stopped: "
                        f"all stopped workers: {stopped_workers}."
                    )
                    continue

                # Add the points to the surrogate model
                surrogate.add_training_data(results)

                # Remove points from pending points
                pending_points = [
                    point
                    for point in pending_points
                    if not any(
                        (point["coord"] == sample["coord"]).all()
                        and point["fid"] == sample["fid"]
                        for sample in surrogate.samples.get(point["fid"], [])
                    )
                ]

                # Update the new surrogate flag
                surrogate_updated = True

                # Update the best point
                previous_best = best_point
                best_point = min(surrogate.samples[0], key=lambda p: p["value"])
                prev_surr_opt = curr_surr_opt
                curr_surr_opt = {**surrogate.get_optimum(), "fid": 0}

                # Update function calls
                for res in results:
                    fid_point = res["fid"]
                    function_calls[fid_point] = function_calls.get(fid_point, 0) + 1

                # Use the result counter as the iteration key. A str key marks the
                # asynchronous case (no max_iterations criterion).
                iteration_key = f"{result_counter:06d}"
                result_counter += 1

                # Save the data and calculate stopping criteria
                stop = self._run_end_iteration(
                    iteration=iteration_key,
                    function_calls=function_calls,
                    best_val=best_point["value"],
                    prev_best_val=previous_best["value"],
                    infill_value=infill_value,
                    best_coord=best_point["coord"],
                    surrogate=surrogate,
                    start_time_iteration=results[0]["start_time"],
                    current_surrogate_optimum=curr_surr_opt,
                    prev_surrogate_optimum=prev_surr_opt,
                )
                self._optimisation_data[iteration_key]["time_since_start"] = (
                    time.time() - start_time
                )

                # if stopping criteria met, break the loop
                if stop:
                    break

            # Wait for a short time to avoid busy waiting
            time.sleep(0.1)

        # Track stop signals
        sent_stop_signals = set()

        # Wait for all workers to finish and stop them
        while len(stopped_workers) < self.mpi_size - 1:
            # Send stop signals to free workers if not send yet
            for worker_rank in range(1, self.mpi_size):
                if worker_rank in free_workers and worker_rank not in sent_stop_signals:
                    self.mpi_comm.send("STOP", dest=worker_rank, tag=11)
                    sent_stop_signals.add(worker_rank)

            # 2. Check for any incoming messages (Results or Stopped signals)
            if self.mpi_comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=22, status=self.mpi_status
            ):
                source_rank = self.mpi_status.Get_source()
                results = self.mpi_comm.recv(source=source_rank, tag=22)

                # if result is stopped, add to stopped workers
                if results == "STOPPED":
                    stopped_workers.add(source_rank)
                    self.config.logger.debug(
                        f"Worker {source_rank} confirmed shutdown."
                    )
                # else add points to final data
                else:
                    self.config.logger.debug(
                        f"Coordinator received final result from {source_rank}"
                    )
                    surrogate.add_training_data(results)
                    if source_rank not in free_workers:
                        free_workers.append(source_rank)

                    # Count the evaluations received during shutdown
                    for res in results:
                        fid_point = res["fid"]
                        function_calls[fid_point] = function_calls.get(fid_point, 0) + 1

            time.sleep(0.1)

        # Get the optimum point, including the results received during shutdown
        best_point = min(surrogate.samples[0], key=lambda p: p["value"])
        prev_surr_opt = curr_surr_opt
        curr_surr_opt = {**surrogate.get_optimum(), "fid": 0}
        opt_surr = self.evaluate_point(
            dict(curr_surr_opt), inplace=False, bypass_delay=True
        )
        self._count_final_evaluation(function_calls)
        if opt_surr["value"] < best_point["value"]:
            best_point = opt_surr

        # Save the final data
        _ = self._run_end_iteration(
            iteration="final",
            function_calls=function_calls,
            best_val=best_point["value"],
            prev_best_val=best_point["value"],
            infill_value=infill_value,
            best_coord=best_point["coord"],
            surrogate=surrogate,
            start_time_iteration=start_time,
            current_surrogate_optimum=curr_surr_opt,
            prev_surrogate_optimum=prev_surr_opt,
        )

        # Add the best point to dict
        self._optimisation_data.setdefault("solution", None)
        self._optimisation_data["solution"] = best_point

        # Add the total time taken to the optimisation data
        total_time = time.time() - start_time
        self.config.logger.info(
            f"Async EGO Optimisation completed in {total_time:.2f} seconds."
        )
        self._optimisation_data["total"]["time_elapsed"] = total_time

        # return the optimum
        return best_point

    def _solve_worker(self) -> None:
        """Run the worker process to evaluate points.

        This process receives tasks from the coordinator process, evaluates them and
        sends the results back to the coordinator process.
        """
        while True:
            task = self.mpi_comm.recv(source=0, tag=11, status=self.mpi_status)

            if task == "STOP":
                self.config.logger.debug("Worker {} stopping.".format(self.mpi_rank))
                self.mpi_comm.send("STOPPED", dest=0, tag=22)
                break

            self.config.logger.debug(
                f"Worker {self.mpi_rank} received {len(task)} points ({task})"
            )
            results = [self.evaluate_point(pt) for pt in task]
            self.config.logger.debug(
                f"Worker {self.mpi_rank} sending results ({results})"
            )
            self.mpi_comm.send(results, dest=0, tag=22)
        return None

    def _add_tasks_to_workers(
        self,
        free_workers: list[int],
        pending_points: list[dict],
        surrogate_updated: bool,
        surrogate: SurrogateBase,
        infill: InfillBase,
    ) -> float:
        """Create and send tasks to free workers.

        Parameters
        ----------
        free_workers
            List of free worker ranks.
        pending_points
            List of currently pending points.
        surrogate_updated
            Flag indicating if the surrogate has been updated.
        surrogate
            Surrogate model to use for optimisation.
        infill
            Infill strategy to use for selecting new points.

        Returns
        -------
        float
            The infill value of the last added task.
        """
        infill_value = float("inf")

        # Create tasks for workers until no free workers remain
        while free_workers:
            # If the surrogate is updated, the infill strategy can be used directly
            if surrogate_updated:
                new_tasks, infill_value = infill(surrogate)

            # If the surrogate is not updated, use KB or CL to generate new tasks
            else:
                multi_point_alg = self.config.get(
                    "optimisation.multipoint_algorithm", None
                )

                if multi_point_alg == "CL":
                    new_tasks, infill_value = self._constant_liar(
                        surrogate, infill, pending_points
                    )
                else:
                    if multi_point_alg != "KB":
                        msg = (
                            f"Unknown multi-point algorithm: {multi_point_alg}, "
                            f"using KB."
                        )
                        self.config.logger.warning(msg)
                    new_tasks, infill_value = self._kriging_believer(
                        surrogate, infill, pending_points
                    )

            # remove duplicate points
            new_tasks = self._remove_duplicates(new_tasks, surrogate, pending_points)

            # Send the tasks to the free workers:
            for task in new_tasks:
                # Add starting time to each point
                task = [{**point, "start_time": time.time()} for point in task]

                # Select a free worker
                if not free_workers:
                    break
                worker = free_workers.pop(0)

                # Send to the worker
                self.config.logger.debug(f"Sending task {task} to worker {worker}")
                self.mpi_comm.send(task, dest=worker, tag=11)

                # Add to pending points
                pending_points.extend(task)

                # update the surrogate updated flag
                surrogate_updated = False

        return infill_value
