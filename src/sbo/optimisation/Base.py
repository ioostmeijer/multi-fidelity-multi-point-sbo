import copy
import os
import pickle
import time
from typing import Any, Union

import numpy as np

from ..infill import InfillBase
from ..surrogate import SurrogateBase
from ..utils import Base

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


class OptimisationBase(Base):
    """Base class for optimisation algorithms."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the OptimisationBase class."""
        super().__init__(*args, **kwargs)

        # Set up empty optimisation process data dict
        _empty_per_iteration_dict = {
            "best_coord": None,
            "best_value": None,
            "infill_value": None,
            "function_calls": {},
            "function_calls_HF": 0,
            "sample_points": None,
            "time_elapsed": None,
        }
        self._optimisation_data: dict[Union[str, int], Any] = {
            "total": copy.deepcopy(_empty_per_iteration_dict)
        }

        # Save the config data to optimisation data
        # expensive_constraints = self.config.expensive_constraints()
        # inexpensive_constraints = self.config.inexpensive_constraints
        self._optimisation_data["config"] = {
            "lb": self.config.lb,
            "ub": self.config.ub,
            "functions": [func.__name__ for func in self.config.functions],
            # "equality_constraints": {
            #     "expensive": [
            #         constr.__name__ for constr in expensive_constraints["equality"]
            #     ],
            #     "inexpensive": [
            #         constr.__name__ for constr in inexpensive_constraints["equality"]
            #     ],
            # },
            # "inequality_constraints": {
            #     "expensive": [
            #         constr.__name__ for constr in expensive_constraints["inequality"]
            #     ],
            #     "inexpensive": [
            #        constr.__name__ for constr in inexpensive_constraints["inequality"]
            #     ],
            # },
            "num_fidelities": self.config.num_fidelities,
            "dimensions": self.config.n_dimensions,
            "settings": self.config.settings,
            "used_algorithms": self.config.used_algorithms,
        }

    @property
    def optimisation_data(self) -> dict:
        """Get the optimisation process data.

        Returns
        -------
        dict
            A dictionary containing data related to the optimisation process.
        """
        return self._optimisation_data

    def solve(
        self, surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs
    ) -> dict:
        """Run the optimisation process using the provided surrogate model and infill.

        Parameters
        ----------
        surrogate
            An instance of a surrogate model.
        infill
            An instance of an infill criterion.

        Returns
        -------
        dict
            A dictionary containing the result of the optimisation process.
            ({'coord': np.ndarray, 'value': float})

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        msg = (
            "The 'solve' method must be implemented in the derived class of "
            "OptimisationBase."
        )
        self.config.logger.error(msg)
        raise NotImplementedError(msg)

    def evaluate_point(
        self, point: dict, inplace: bool = False, bypass_delay: bool = False
    ) -> dict:
        """Evaluate a given point using the objective function.

        Parameters
        ----------
        point
            A dictionary containing the point to be evaluated.
            ({'coord': np.ndarray, "fid": int})
        inplace
            If True, the evaluation result is added to the input point dictionary.
            If False, a new dictionary is returned with the evaluation result.
        bypass_delay
            If True, the delay specified will be skipped.

        Returns
        -------
        dict
            A dictionary containing the evaluated point and its objective value.
            ({'coord': np.ndarray, "fid": int, 'value': float})
        """
        # Add starting time to point if not already present
        if inplace:
            if "start_time" not in point:
                point["start_time"] = time.time()
            if MPI is not None:
                point["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
            else:
                point["mpi_rank"] = 0
        elif not inplace:
            evaluated_point = copy.deepcopy(point)
            if "start_time" not in evaluated_point:
                evaluated_point["start_time"] = time.time()
            if MPI is not None:
                evaluated_point["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
            else:
                evaluated_point["mpi_rank"] = 0

        fidelity = point["fid"]
        func = self.config.functions[fidelity]
        value = func(point["coord"])[0]
        if inplace:
            point["value"] = value
        else:
            evaluated_point["value"] = value

        # Pause if a function delay is set
        function_delay = self.config.get("function_delay", None)
        if function_delay is not None and not bypass_delay:
            if len(function_delay) == 0:
                delay = None
            elif fidelity < len(function_delay):
                delay = function_delay[fidelity]
            else:
                # Fewer delays than fidelities: use the delay of the last fidelity
                delay = function_delay[-1]
                self.config.logger.warning(
                    f"No function delay given for fidelity {fidelity} "
                    f"(function_delay has {len(function_delay)} entries); using the "
                    f"last value ({delay} s)."
                )
            if delay is None:
                self.config.logger.warning(
                    f"No function delay found for fidelity {fidelity}. "
                    "No delay will be applied."
                )
            else:
                time.sleep(delay)

        if inplace:
            point["end_time"] = time.time()
            return point
        else:
            evaluated_point["end_time"] = time.time()
            return evaluated_point

    def _count_final_evaluation(self, function_calls: dict) -> None:
        """Count the final HF evaluation of the surrogate optimum.

        The stopping criteria on function calls reserve one HF call for this
        evaluation, so the totals stay within max_function_calls(_HF).

        Parameters
        ----------
        function_calls
            The function calls per fidelity, updated in place.
            ({fidelity_level: num_calls, ...})
        """
        function_calls[0] = function_calls.get(0, 0) + 1
        self._optimisation_data["total"]["function_calls"] = function_calls
        self._optimisation_data["total"]["function_calls_HF"] = function_calls[0]

    def _calc_stopping_criteria(
        self,
        iteration: int | str,
        function_call: dict,
        best_val: float,
        infill_value: float,
        total_data: dict,
        optimisation_data: dict,
    ) -> bool:
        r"""Calculate the stopping criteria for the optimisation process.

        Calculates whether any of the stopping criteria have been met based on the
        current iteration, function calls, best objective value, previous best
        objective value, and infill value. The implemented stopping criteria are:

        - Maximum number of iterations.
        - Convergence threshold, based on the change in the surrogate optimum
            $\hat{y}^*_t$ over the last $w$ iterations (`convergence_window`):
            $\lvert \hat{y}^*_{t-w} - \hat{y}^*_t \rvert < \varepsilon$.
        - Relative convergence threshold:
            $\lvert \hat{y}^*_{t-w} - \hat{y}^*_t \rvert
            / \max\left(\lvert \hat{y}^*_{t-w} \rvert, \epsilon_{\mathrm{mach}}\right)
            < \varepsilon_r$.
        - Maximum number of function calls.
        - Maximum number of high-fidelity function calls.
        - Minimum infill value.

        Parameters
        ----------
        iteration
            The current iteration number.
        function_call
            A dictionary containing the number of function calls made.
            ({fidelity_level: num_calls, ...})
        best_val
            The best objective value found so far.
        infill_value
            The infill value of the current iteration.
        total_data
            A dictionary containing the total data collected during the optimisation
            process.
        optimisation_data
            A dictionary containing the data collected during the optimisation
            process for each iteration.

        Returns
        -------
        bool
            True if any stopping criteria are met, False otherwise.
        """
        # Calculate the convergence criterion values
        prev_best_val = float("inf")
        best_val = optimisation_data[iteration]["surrogate_optimum"]["value"]
        num_iterations_ago = self.config.get("optimisation.convergence_window")
        if len(optimisation_data.keys()) > num_iterations_ago and iteration not in [
            "final",
            "total",
        ]:
            sorted_keys = sorted(
                [
                    k
                    for k in optimisation_data
                    if k not in ["final", "total", "init", "config"]
                ]
            )
            try:
                key_to_consider = sorted_keys[-num_iterations_ago - 1]
                prev_best_val = optimisation_data[key_to_consider]["surrogate_optimum"][
                    "value"
                ]
            except IndexError:
                prev_best_val = float("inf")

        if prev_best_val != float("inf"):
            conv_value = abs(prev_best_val - best_val)
            eps = np.finfo(float).eps
            denom = abs(prev_best_val)
            if denom < eps:
                denom = eps
            rel_conv_value = conv_value / denom
        else:
            conv_value = float("inf")
            rel_conv_value = float("inf")

        stopping_criteria = {
            "iteration": {
                "threshold": self.config.get("optimisation.max_iterations", None),
                "lambda": lambda iteration_num, threshold: iteration_num >= threshold,
                # Number of completed iterations (the counter starts at 0)
                "value": iteration + 1 if isinstance(iteration, int) else iteration,
            },
            "convergence": {
                "threshold": self.config.get(
                    "optimisation.convergence_threshold", None
                ),
                "lambda": lambda diff, threshold: diff < threshold,
                "value": conv_value,
            },
            "relative_convergence": {
                "threshold": self.config.get(
                    "optimisation.relative_convergence_threshold", None
                ),
                "lambda": lambda rel_diff, threshold: rel_diff < threshold,
                "value": rel_conv_value,
            },
            "function_calls": {
                "threshold": self.config.get("optimisation.max_function_calls", None),
                "lambda": lambda calls, threshold: calls >= threshold,
                # + 1 reserves the final HF evaluation of the surrogate optimum
                "value": sum(total_data["function_calls"].values()) + 1,
            },
            "function_calls_HF": {
                "threshold": self.config.get(
                    "optimisation.max_function_calls_HF", None
                ),
                "lambda": lambda calls_hf, threshold: calls_hf >= threshold,
                # + 1 reserves the final HF evaluation of the surrogate optimum
                "value": function_call[0] + 1,
            },
            "infill_value": {
                "threshold": self.config.get("optimisation.min_infill_value", None),
                "lambda": lambda infill_value_test, threshold: (
                    infill_value_test < threshold
                ),
                "value": infill_value,
            },
        }

        criteria_values = {
            k: v_dict["value"] for k, v_dict in stopping_criteria.items()
        }
        self.config.logger.debug(
            f"Stopping criteria values at iteration {iteration}: {criteria_values}"
        )

        # Remove convergence criteria for which the threshold is None (not set)
        for key in list(stopping_criteria.keys()):
            if stopping_criteria[key]["threshold"] is None:
                stopping_criteria.pop(key)

        # Deal with Async case where iteration is a str (time)
        if isinstance(iteration, str):
            stopping_criteria.pop("iteration", None)

        for key, val_dict in stopping_criteria.items():
            threshold = val_dict["threshold"]
            value = val_dict["value"]
            if threshold is not None:
                if val_dict["lambda"](value, threshold):
                    self.config.logger.info(
                        f"Stopping criteria met: {key} with threshold {threshold} and "
                        f"value {value}."
                    )
                    return True
        return False

    def _run_end_iteration(
        self,
        iteration: int | str,
        function_calls: dict,
        best_val: float,
        prev_best_val: float,
        infill_value: float,
        best_coord: np.ndarray,
        surrogate: SurrogateBase,
        start_time_iteration: float,
        current_surrogate_optimum: dict,
        prev_surrogate_optimum: dict,
    ) -> bool:
        """Handle the end of an optimisation iteration, saving data and checking.

        This is to be run after every iteration of the optimisation process. It saves
        relevant data from the iteration and checks if any stopping criteria have been
        met.

        Parameters
        ----------
        iteration
            The current iteration number.
        function_calls
            A dictionary containing the number of function calls made.
            ({fidelity_level: num_calls, ...})
        best_val
            The best objective value found so far.
        prev_best_val
            The best objective value found in the previous iteration.
        infill_value
            The infill value of the current iteration.
        best_coord
            The coordinates of the best point found so far.
        surrogate
            An instance of a surrogate model.
        start_time_iteration
            The start time of the iteration.
        current_surrogate_optimum
            The current optimum of the surrogate model.
        prev_surrogate_optimum
            The previous optimum of the surrogate model.

        Returns
        -------
        bool
            True if any stopping criteria are met, False otherwise.
        """
        time_elapsed_iteration = time.time() - start_time_iteration
        save_data = {
            "best_coord": copy.deepcopy(best_coord),
            "best_value": best_val,
            "infill_value": infill_value,
            "function_calls": copy.deepcopy(function_calls),
            "function_calls_HF": function_calls[0],
            "sample_points": copy.deepcopy(surrogate.samples),
            "time_elapsed": time_elapsed_iteration,
            "surrogate_optimum": copy.deepcopy(current_surrogate_optimum),
        }
        self._optimisation_data[iteration] = save_data

        self._optimisation_data["total"]["best_coord"] = best_coord
        self._optimisation_data["total"]["best_value"] = best_val
        self._optimisation_data["total"]["infill_value"] = infill_value
        self._optimisation_data["total"]["function_calls"] = function_calls
        self._optimisation_data["total"]["function_calls_HF"] = function_calls[0]
        self._optimisation_data["total"]["sample_points"] = surrogate.samples.copy()
        self._optimisation_data["total"]["surrogate_optimum"] = (
            current_surrogate_optimum.copy()
        )

        return self._calc_stopping_criteria(
            iteration=iteration,
            function_call=function_calls,
            best_val=best_val,
            infill_value=infill_value,
            total_data=self._optimisation_data["total"],
            optimisation_data=self._optimisation_data,
        )

    def save_optimisation_data(self, file_name: str) -> None:
        """Save the optimisation data to a pickle file.

        The file is saved in the 'save_folder' setting with the given file_name.

        Parameters
        ----------
        file_name
            The name of the file to save the optimisation data to.
        """
        if not os.path.exists(self.config["save_folder"]):
            os.makedirs(self.config["save_folder"])

        save_path = os.path.join(self.config["save_folder"], f"{file_name}.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(self._optimisation_data, f)

    def _kriging_believer(
        self, surrogate: SurrogateBase, infill: InfillBase, sample_locations: list[dict]
    ) -> tuple[list[list[dict]], float]:
        """Kriging Believer (KB) method.

        This method enables the use of KB in the optimisation process.
        It temporarily add the sample locations to the surrogate training data, with the
        value equal to the surrogate prediction of the sample location at the fidelity
        of that location. Following this, the infill strategy is applied to the
        surrogate model to get the next point. If the surrogate can not predict per
        fidelity (supports_fidelity_prediction is False), the HF prediction is used
        for every location.

        Parameters
        ----------
        surrogate
            An instance of a surrogate model.
        infill
            An instance of an infill criterion.
        sample_locations
            A list of dictionaries containing the sample locations to be added.
            ({'coord': np.ndarray, "fid": int})

        Returns
        -------
        tuple[list[list[dict]], float]
            A tuple containing a list of new sample locations and the infill value.
            (list of list of dict, float)
        """
        # Create a copy of the surrogate data
        original_data = [
            copy.deepcopy(point)
            for fid in surrogate.samples
            for point in surrogate.samples[fid]
        ]
        original_model = copy.deepcopy(surrogate._model)

        # Get the values at the sample locations. The lie is the prediction at the
        # fidelity of the pending point. A surrogate that can not predict per
        # fidelity gets the HF prediction as lie for every point.
        supports_fidelity = getattr(surrogate, "supports_fidelity_prediction", False)
        new_training_data = []
        n_lf_with_hf_lie = 0
        for point in sample_locations:
            correct_coord = np.array(point["coord"]).reshape(1, -1)
            if supports_fidelity:
                predict_kwargs = {"fidelity": point["fid"]}
            else:
                predict_kwargs = {}
                if point["fid"] != 0:
                    n_lf_with_hf_lie += 1
            new_training_data.append(
                {
                    **point,
                    "value": surrogate.predict(
                        correct_coord, return_variance=False, **predict_kwargs
                    )[0][0],
                }
            )
        if n_lf_with_hf_lie:
            self.config.logger.warning(
                f"{type(surrogate).__name__} can not predict per fidelity; the HF "
                f"prediction is used as Kriging Believer lie for {n_lf_with_hf_lie} "
                "lower-fidelity pending point(s)."
            )

        # Set new training data
        surrogate.add_training_data(new_training_data)

        # Train the surrogate with the new data
        surrogate.train()

        # Find the new point(s) to evaluate
        new_point, _ = infill(surrogate)

        # Restore the original surrogate data
        surrogate.set_training_data(original_data)
        surrogate._model = original_model
        surrogate._trained = True

        # return the new points and the infill value as infinity
        # (since the optimisation can not stop on KB)
        return new_point, float("inf")

    def _constant_liar(
        self, surrogate: SurrogateBase, infill: InfillBase, sample_locations: list[dict]
    ) -> tuple[list[list[dict]], float]:
        """Constant Liar (CL) method.

        This method enables the use of CL in the optimisation process. First it adds the
        lie to the surrogate training data at the sample locations. After which the
        infill strategy is applied to this surrogate model to get the next point.
        The options for the lie are:
        - `max': The lie is the maximum value of the surrogate training data.
        - `min': The lie is the minimum value of the surrogate training data.
        - `mean': The lie is the mean value of the surrogate training data.

        Parameters
        ----------
        surrogate
            An instance of a surrogate model.
        infill
            An instance of an infill criterion.
        sample_locations
            A list of dictionaries containing the sample locations to be added.
            ({'coord': np.ndarray, "fid": int})

        Returns
        -------
        tuple[list[list[dict]], float]
            A tuple containing a list of new sample locations and the infill value.
            (list of list of dict, float)

        Raises
        ------
        ValueError
            If the surrogate has no samples to determine the lie value.
        """
        # Create a copy of the surrogate data
        original_data = [
            copy.deepcopy(point)
            for fid in surrogate.samples
            for point in surrogate.samples[fid]
        ]
        original_model = copy.deepcopy(surrogate._model)

        # Get the values for the highest fidelity with samples
        values = []
        for fid in range(self.config.num_fidelities):
            if len(surrogate.samples.get(fid, [])) > 0:
                values = [point["value"] for point in surrogate.samples[fid]]
                break

        if len(values) == 0:
            msg = "No samples available in the surrogate to determine lie value."
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Get value to use as lie
        mode = self.config.get("optimisation.constant_liar_lie", "mean")
        _funcs = {"mean": np.mean, "min": np.min, "max": np.max}
        if mode not in _funcs:
            self.config.logger.warning(
                "Invalid constant_liar_lie configuration. Defaulting to 'mean'."
            )
            mode = "mean"
        func = _funcs.get(mode, np.mean)
        lie_value = func(values)

        # Set new training data with lie values
        surrogate.add_training_data(
            [{**point, "value": lie_value} for point in sample_locations]
        )

        # Train the surrogate with the new data
        surrogate.train()

        # Get the new sample location(s)
        new_points, _ = infill(surrogate)

        # Reset the original training data
        surrogate.set_training_data(original_data)
        surrogate._model = original_model
        surrogate._trained = True

        # return the new points and the infill value as infinity
        # (since the optimisation can not stop on CL)
        return new_points, float("inf")

    def _remove_duplicates(
        self,
        tasks: list[list[dict]],
        surrogate: SurrogateBase,
        pending_points: list[dict] | None = None,
    ) -> list[list[dict]]:
        """Remove duplicate points from the list of tasks.

        This method checks the proposed new points against the existing samples in the
        surrogate model, the pending points (if provided) and the points proposed
        earlier in the same batch, to remove any duplicates. If a duplicate is found,
        it is replaced with a sample at the same location but a higher fidelity, if
        possible. If this is not the case a randomly sampled point is used instead.

        Parameters
        ----------
        tasks
            A list of tasks, where each task is a list of dictionaries containing the
            proposed new points. ({'coord': np.ndarray, "fid": int})
        surrogate
            An instance of a surrogate model.
        pending_points
            A list of dictionaries containing the points that are currently pending
            evaluation. ({'coord': np.ndarray, "fid": int})

        Returns
        -------
        list[list[dict]]
            A list of tasks with duplicate points removed. Each task is a list of
            dictionaries containing the proposed new points.
            ({'coord': np.ndarray, "fid": int})
        """
        if pending_points is None:
            pending_points = []

        # Points accepted so far in this batch. A new point is also a duplicate if it
        # equals a point proposed earlier in the same batch.
        accepted: list[dict] = []
        n_duplicates = 0

        for tsk_index, task in enumerate(tasks):
            new_task = []
            for point in task:
                taken = pending_points + accepted
                if not self._check_point_included(point, surrogate, taken):
                    new_task.append(point)
                    accepted.append(point)
                    continue

                n_duplicates += 1
                replacement = self._replace_duplicate(point, surrogate, taken)
                new_task.append(replacement)
                accepted.append(replacement)
            tasks[tsk_index] = new_task

        if n_duplicates > 0:
            self.config.logger.info(
                f"Found {n_duplicates} duplicate points. Replaced them in the tasks."
            )
        return tasks

    def _replace_duplicate(
        self, duplicate: dict, surrogate: SurrogateBase, taken: list[dict]
    ) -> dict:
        """Get a replacement for a duplicate point.

        The replacement is the same location at a higher fidelity, if that is not a
        duplicate. Otherwise a random point at the lowest fidelity is used.

        Parameters
        ----------
        duplicate
            The duplicate point. ({'coord': np.ndarray, "fid": int})
        surrogate
            An instance of a surrogate model.
        taken
            The pending points and the points already accepted in this batch.

        Returns
        -------
        dict
            The replacement point. ({'coord': np.ndarray, "fid": int})
        """
        # Try to replace with same location but a higher fidelity
        for new_fid in range(duplicate["fid"] - 1, -1, -1):
            replacement_point = {**duplicate, "fid": new_fid}
            if not self._check_point_included(replacement_point, surrogate, taken):
                self.config.logger.info(
                    f"Replacing duplicate point {duplicate} with higher fidelity "
                    f"point at same location with fid {new_fid}."
                )
                return replacement_point

        while True:
            # Replace with a random point
            random_point = {
                "coord": self.config.rng.uniform(self.config.lb, self.config.ub),
                "fid": self.config.num_fidelities - 1,
            }
            if not self._check_point_included(random_point, surrogate, taken):
                self.config.logger.info(
                    f"Replacing duplicate point {duplicate} with random point "
                    f"{random_point}."
                )
                return random_point

    def _check_point_included(
        self, point: dict, surrogate: SurrogateBase, pending_points: list[dict] | None
    ) -> bool:
        """Check if a point is included in the surrogate samples or pending points.

        Returns True if the point is found in either the surrogate samples or the
        pending points, and False otherwise. A point is considered included if it is
        within 1e-4 of the (x_ub - x_lb) for all x in n_dim.

        Parameters
        ----------
        point
            A dictionary containing the point to be checked.
            ({'coord': np.ndarray, "fid": int})
        surrogate
            An instance of a surrogate model.
        pending_points
            A list of dictionaries containing the points that are currently pending
            evaluation. ({'coord': np.ndarray, "fid": int})

        Returns
        -------
        bool
            Whether the point is included in the surrogate samples or pending points.
        """
        fid = point["fid"]

        in_samples = any(
            self._points_equal(sample, point)
            for sample in surrogate.samples.get(fid, [])
        )
        in_pending = (
            False
            if pending_points is None
            else any(
                self._points_equal(pending_point, point)
                for pending_point in pending_points
            )
        )
        return in_samples or in_pending

    def _points_equal(self, a: dict, b: dict) -> bool:
        """Helper that returns True when points are equal, False otherwise.

        Points are equal if all coordinates are within 1e-4 of UB - LB.

        Parameters
        ----------
        a
            The first point.
        b
            The second point.

        Returns
        -------
        bool
            Whether the points are equal or not.
        """
        max_distance = (self.config.ub - self.config.lb) * 1e-4
        return a["fid"] == b["fid"] and np.all(
            np.isclose(a["coord"], b["coord"], atol=max_distance, rtol=0)
        )

    def __call__(
        self, surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs
    ) -> dict:
        """Run the optimisation process using the provided surrogate model and infill.

        Parameters
        ----------
        surrogate
            An instance of a surrogate model.
        infill
            An instance of an infill criterion.

        Returns
        -------
        dict
            A dictionary containing the result of the optimisation process.
            ({'coord': np.ndarray, 'value': float})
        """
        return self.solve(surrogate, infill, *args, **kwargs)
