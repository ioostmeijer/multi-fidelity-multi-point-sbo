import copy
import importlib
import math
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D, lineStyles
from smt.applications import NestedLHS
from smt.sampling_methods import LHS

from .infill import InfillBase
from .optimisation import OptimisationBase
from .surrogate import SurrogateBase
from .utils import Config

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


class SBO:
    """Surrogate-Based Optimisation (SBO) class.

    This class is the entry-point for the end-user to perform surrogate-based
    optimisation, using any combination of the supported surrogate models, infill
    strategies and optimisation algorithms.

    The SBO class initialises the specified components based on the provided
    configuration and orchestrates the optimisation process.

    To use the SBO class, create an instance with a Config object and call the
    `run` method to execute the optimisation process. The optimum solution can be
    accessed via the `optimum` property or by calling the instance directly.

    Attributes
    ----------
        config
            The configuration object containing settings for the SBO process.
        _infill
            The infill strategy component.
        _surrogate
            The surrogate model component.
        _optimisation
            The optimisation algorithm component.
        _optimum
            The optimum solution found by the SBO process.

    Examples
    --------
    >>> config = Config(...)
    >>> sbo = SBO(config)
    >>> optimum = sbo()  # or sbo.run()
    """

    def __init__(
        self, config: Config, *args, initial_samples: list[dict] | None = None, **kwargs
    ) -> None:
        """Initialise the SBO class.

        Parameters
        ----------
        config
            The configuration object containing settings for the SBO process.
        initial_samples
            Optional initial samples to set in the surrogate model.
        """
        self.config: Config = config
        self._optimum: dict | None = None

        self._infill: InfillBase = self._initialise_component(
            *args,
            algorithm=self.config.used_algorithms["infill"],
            algorithm_type="infill",
            config=self.config,
            **kwargs,
        )
        self._surrogate: SurrogateBase = self._initialise_component(
            *args,
            algorithm=self.config.used_algorithms["surrogate"],
            algorithm_type="surrogate",
            config=self.config,
            **kwargs,
        )
        self._optimisation: OptimisationBase = self._initialise_component(
            *args,
            algorithm=self.config.used_algorithms["optimisation"],
            algorithm_type="optimisation",
            config=self.config,
            **kwargs,
        )

        # Take the backup only after the components are created: a component can
        # replace itself (e.g. HK -> KRG for a single fidelity) and update
        # config.used_algorithms. Taking the backup earlier would make run() see a
        # change and rebuild the surrogate without its initial samples.
        self._backup_used_alg: dict = copy.deepcopy(self.config.used_algorithms)

        # Set initial samples if provided else samples using LHS
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            self.config.logger.info(f"Dealing with initial samples with MPI: {MPI}")
            if MPI is not None:
                self.config.logger.info(f"Current rank = {MPI.COMM_WORLD.Get_rank()}")
            if not initial_samples:
                initial_samples = self._get_initial_samples()
            else:
                initial_samples = self._ensure_enough_initial_samples(initial_samples)

            # If MPI is used, send initial samples to other ranks
            if MPI is not None:
                initial_samples = MPI.COMM_WORLD.bcast(initial_samples, root=0)

        # If MPI is used, receive initial samples from root
        elif MPI is not None and MPI.COMM_WORLD.Get_rank() != 0:
            initial_samples = MPI.COMM_WORLD.bcast(initial_samples, root=0)

        if MPI is not None:
            self.config.logger.debug(
                f"Initial samples on rank {MPI.COMM_WORLD.Get_rank()}: "
                f"{initial_samples}"
            )
        self._surrogate.set_training_data(initial_samples)

        # # Set up the surrogates for the constraints.
        # expensive_constraints = self.config.expensive_constraints()
        # self._constraint_surrogates = {k: [] for k in expensive_constraints.keys()}
        # for k, constraints in expensive_constraints.items():
        #     for constr in constraints:
        #         initial_samples_constr = [
        #             constr[s["fid"]](s["coord"]) for s in initial_samples
        #         ]
        #         constr_surr: SurrogateBase = self._initialise_component(
        #             algorithm=self.config.used_algorithms["surrogate"],
        #             algorithm_type="surrogate",
        #             config=self.config,
        #             **kwargs,
        #         )
        #         constr_surr.set_training_data(initial_samples_constr)
        #         self._constraint_surrogates[k].append(constr_surr)
        # self.config.surrogate_constraints = self._constraint_surrogates

    def _get_initial_samples(self) -> list[dict]:
        """Generate the list of initial samples, if required.

        Generate the list of initial samples, based on LHS. This uses the LHS method
        from SMT, if nested samples are required the nestedLHS is used.

        Returns
        -------
        list[dict]
            A list of all the initial points.
        """
        self.config.logger.info("Getting initial samples.")
        if self.config.num_fidelities == 1:
            lhs = LHS(
                xlimits=self.config.bounds,
                seed=self.config.get("random_seed"),
            )
            x_init = lhs(self.config.settings["number_initial_samples"])
            initial_samples = [
                {
                    "coord": x_init[i],
                    "value": self.config.functions[0](x_init[i])[0],
                    "fid": 0,
                }
                for i in range(x_init.shape[0])
            ]
        else:
            if self.config.get("nested"):
                nested_lhs = NestedLHS(
                    xlimits=self.config.bounds,
                    seed=self.config.get("random_seed"),
                    nlevel=self.config.num_fidelities,
                )
                x_init = nested_lhs(self.config.get("number_initial_samples"))
                initial_samples = []
                for fid, x_list in enumerate(x_init):
                    fid = self.config.num_fidelities - fid - 1
                    initial_samples.extend(
                        [
                            {
                                "coord": x_list[i],
                                "value": self.config.functions[fid](x_list[i])[0],
                                "fid": fid,
                            }
                            for i in range(len(x_list))
                        ]
                    )
            else:
                lhs = LHS(
                    xlimits=self.config.bounds,
                    seed=self.config.get("random_seed"),
                )
                x_init = lhs(self.config.get("number_initial_samples"))
                fid = self.config.rng.integers(
                    0, self.config.num_fidelities, size=x_init.shape[0]
                )
                initial_samples = [
                    {
                        "coord": x_init[i],
                        "value": self.config.functions[fid[i]](x_init[i])[0],
                        "fid": int(fid[i]),
                    }
                    for i in range(x_init.shape[0])
                ]
        initial_samples = self._ensure_enough_initial_samples(initial_samples)

        return initial_samples

    def _ensure_enough_initial_samples(self, initial_samples: list[dict]) -> list[dict]:
        """Ensure there are enough initial samples for each fidelity.

        Parameters
        ----------
        initial_samples
            The list of initial samples to check and augment if necessary.

        Returns
        -------
        list[dict]
            The augmented list of initial samples with at least 2 samples per fidelity.
        """
        # Work on a copy, so the list passed by the user is not changed
        initial_samples = list(initial_samples)

        # Ensure at least 2 samples per fidelity
        fid_counts = dict.fromkeys(range(self.config.num_fidelities), 0)
        for sample in initial_samples:
            fid_counts[sample["fid"]] += 1
        for fid, count in fid_counts.items():
            while count < 2:
                # Get new random sample. This is not done by LHS to avoid infinite loop
                # with duplicate samples.
                x_new = self.config.rng.random(self.config.dimension)
                # Scale to bounds
                x_new = x_new * (self.config.ub - self.config.lb) + self.config.lb
                if not any(np.allclose(x_new, p["coord"]) for p in initial_samples):
                    new_sample = {
                        "coord": x_new,
                        "value": self.config.functions[fid](x_new)[0],
                        "fid": fid,
                    }
                    initial_samples.append(new_sample)
                    count += 1

                    if self.config["nested"] and fid != self.config.num_fidelities - 1:
                        # Add lower fidelity samples
                        for lower_fid in range(fid + 1, self.config.num_fidelities):
                            new_lower_sample = {
                                "coord": x_new,
                                "value": self.config.functions[lower_fid](x_new)[0],
                                "fid": lower_fid,
                            }
                            initial_samples.append(new_lower_sample)
                            fid_counts[lower_fid] += 1
        return initial_samples

    def _initialise_component(
        self, algorithm: str, algorithm_type: str, *args, **kwargs
    ) -> InfillBase | SurrogateBase | OptimisationBase:
        """Initialise the component based on the algorithm name.

        Parameters
        ----------
        algorithm
            The name of the algorithm to initialise.
        algorithm_type
            The type of the algorithm (infill, surrogate, optimisation).

        Returns
        -------
        InfillBase | SurrogateBase | OptimisationBase
            An instance of the specified algorithm class.

        Raises
        ------
        ImportError
            If the algorithm can not be found or imported.
        """
        try:
            module = importlib.import_module(
                f".{algorithm_type}.{algorithm}", package=__package__
            )
            cls = getattr(module, algorithm)
            return cls(*args, **kwargs)
        except ImportError as e:
            msg = (
                f"Could not import {algorithm_type} algorithm '{algorithm}', "
                f"with error: '{e}'"
            )
            self.config.logger.error(msg)
            raise ImportError(msg) from e

    @property
    def optimum(self) -> dict:
        """Get the optimum solution found by the SBO process.

        Runs the optimisation first if it has not run yet, or if the used algorithms
        changed.

        Returns
        -------
        dict
            The optimum solution point ({'coord': np.ndarray, 'value': float, ...}).
        """
        if (
            self._optimum is None
            or self.config.used_algorithms != self._backup_used_alg
        ):
            _ = self.run()
        return self._optimum

    @property
    def optimisation_data(self) -> dict:
        """Get the optimisation data from the optimisation process.

        Runs the optimisation first if it has not run yet, or if the used algorithms
        changed.

        Returns
        -------
        dict
            The data per iteration (keyed by iteration) plus the 'init', 'total' and
            'solution' entries.
        """
        if (
            self._optimum is None
            or self.config.used_algorithms != self._backup_used_alg
        ):
            _ = self.run()
        return self._optimisation.optimisation_data

    def run(self) -> dict:
        """Execute the SBO process and return the optimum solution.

        Returns
        -------
        dict
            The optimum solution point ({'coord': np.ndarray, 'value': float}).
        """
        if self.config.used_algorithms != self._backup_used_alg:
            self.update_used_algorithms(self.config.used_algorithms)

        # For not rank==0 optimisation no points need to be processed.
        if MPI is not None and MPI.COMM_WORLD.Get_rank() != 0:
            self._optimisation(surrogate=self._surrogate, infill=self._infill)
            return {}

        if (
            max(
                [
                    len(self._surrogate.samples[fid])
                    for fid in self._surrogate.samples.keys()
                ]
            )
            <= 1
        ):
            self.config.logger.warning(
                "Tried calling SBO run, without any samples in surrogate."
            )
            self._surrogate.set_training_data(self._get_initial_samples())

        self._optimum = self._optimisation(
            surrogate=self._surrogate, infill=self._infill
        )
        return self._optimum

    def save_optimisation_data(self, file: str) -> None:
        """Save the optimisation data to a file.

        Parameters
        ----------
        file
            The file path to save the optimisation data.
        """
        self._optimisation.save_optimisation_data(file)

    def update_used_algorithms(self, new_used_alg: dict) -> None:
        """Update the used algorithms.

        Parameters
        ----------
        new_used_alg
            new used algorithms dictionary
            (required keys: infill, surrogate, optimisation)
        """
        for key, new_alg in new_used_alg.items():
            if key not in self.config.used_algorithms.keys():
                self.config.logger.warning(
                    f"Algorithm '{key}' not a valid key, ignoring."
                )
                continue

            # Test if not the same type, if so continue
            if new_alg == self._backup_used_alg[key]:
                continue

            self._backup_used_alg[key] = new_alg
            self.config.update_used_algorithms(copy.deepcopy(self._backup_used_alg))

            if key == "optimisation":
                self._optimisation = self._initialise_component(
                    algorithm=new_alg,
                    algorithm_type=key,
                    config=self.config,
                )
            elif key == "surrogate":
                # Keep the training data of the old surrogate
                old_samples = copy.deepcopy(self._surrogate.samples)
                self._surrogate = self._initialise_component(
                    algorithm=new_alg,
                    algorithm_type=key,
                    config=self.config,
                )
                if any(old_samples.values()):
                    self._surrogate.set_training_data(old_samples)
            elif key == "infill":
                self._infill = self._initialise_component(
                    algorithm=new_alg,
                    algorithm_type=key,
                    config=self.config,
                )

        # A component can replace itself on creation (e.g. HK -> KRG) and update
        # config.used_algorithms, so sync the backup with the config afterwards.
        self._backup_used_alg = copy.deepcopy(self.config.used_algorithms)

    def __call__(self) -> dict:
        """Allow the SBO instance to be called directly to get the optimum solution.

        Returns
        -------
        dict
            The optimum solution point ({'coord': np.ndarray, 'value': float}).
        """
        return self.optimum

    # Plotting of optimisation data
    def _check_iterations_in_opt_data(self) -> bool:
        """Check if iterations are present in the optimisation data.

        Returns
        -------
        bool
            True if iterations are present in the optimisation data, False otherwise.
        """
        return (
            0 in self._optimisation.optimisation_data.keys()
            or "0" in self._optimisation.optimisation_data.keys()
        )

    def plot_surrogate(
        self,
        title: str,
        filename: str | None = None,
        show: bool = False,
        up_to: int | str | None = None,
        iterations: list[int] | None = None,
        plot_for: str = "iteration",
        true_functions_plot: list[int] | None = None,
        fid_samples_plot: list[int] | None = None,
    ) -> None:
        """Plot the surrogate model for 1D and 2D problems for iteration or timestep.

        Exposed method for plotting the surrogates.

        Parameters
        ----------
        title
            The title for the plot.
        filename
            The filename to save the plot to. If None, the plot will not be saved.
            If a relative path is provided, the relative path from the output directory
            will be used.
        show
            Whether to show the plot after creating it.
        up_to
            The iteration or time step up to which to plot. If None, all iterations or
            time steps will be plotted.
        iterations
            The specific iterations to plot. If provided, only these iterations will be
            plotted. This is only supported for plotting by iteration.
        plot_for
            Whether to plot by iteration or by time step. Currently only "iteration" is
            supported for plotting by iteration.
        true_functions_plot
            The list of fidelities of the true functions to plot. If None, no true
            functions will be plotted.
        fid_samples_plot
            The list of fidelities of the sample points to plot. If None, sample points
            from all fidelities will be plotted. If no sample points are to be plotted,
            set to an empty list.
        """
        # Ensure optimisation data is available
        _ = self.optimum

        if self.config.dimension > 2:
            self.config.logger.warning(
                "Plotting surrogate is only supported for 1D and 2D problems."
            )
            return

        if plot_for not in ["iteration"]:
            self.config.logger.warning(
                f"Invalid plot_for value '{plot_for}', defaulting to 'iteration'."
                f"Valid options are 'iteration'."
            )
            plot_for = "iteration"

        if plot_for == "iteration":
            self._plot_surrogate_by_iteration(
                title,
                filename,
                show,
                up_to,
                iterations,
                true_functions_plot,
                fid_samples_plot,
            )

    def _plot_surrogate_by_iteration(
        self,
        title: str,
        filename: str | None = None,
        show: bool = False,
        up_to: int | str | None = None,
        iterations: list[int] | None = None,
        true_functions_plot: list[int] | None = None,
        fid_samples_plot: list[int] | None = None,
    ) -> None:
        """Plot the surrogate model for 1D and 2D problems by iteration.

        Parameters
        ----------
        title
            The title for the plot.
        filename
            The filename to save the plot to. If None, the plot will not be saved.
            If a relative path is provided, the relative path from the output directory
            will be used.
        show
            Whether to show the plot after creating it.
        up_to
            The iteration up to which to plot. If None, all iterations will be plotted.
        iterations
            The specific iterations to plot. If provided, only these iterations will be
            plotted.
        true_functions_plot
            The list of fidelities of the true functions to plot. If None, no true
            functions will be plotted. Only supported for 1D problems.
        fid_samples_plot
            The list of fidelities of the sample points to plot. If None, sample points
            from all fidelities will be plotted. If no sample points are to be plotted,
            set to an empty list.
        """

        def _iteration_title(iteration_key: int | str) -> str:
            """Return the subplot title for an iteration key.

            Synchronous optimisers use int keys. EgoAsync uses zero-padded str keys
            (a result counter) and stores the time since the start of the run.

            Parameters
            ----------
            iteration_key
                The key of the iteration in the optimisation data.

            Returns
            -------
            str
                The subplot title.
            """
            if isinstance(iteration_key, str):
                time_elapsed = optimisation_data[iteration_key].get("time_since_start")
                if time_elapsed is not None:
                    return f"Time elapsed: {time_elapsed:.1f} sec"
            return f"Iteration {int(iteration_key) + 1}"

        def _contains_point(point: dict, points: list[dict]) -> bool:
            """Check whether a point (by coordinates) is in a list of points.

            Comparing the dicts with `in` fails for multi-element NumPy arrays.

            Parameters
            ----------
            point
                The point to look for.
            points
                The points to search.

            Returns
            -------
            bool
                True if a point with the same coordinates is in points.
            """
            return any(
                np.array_equal(np.asarray(point["coord"]), np.asarray(p["coord"]))
                for p in points
            )

        # Get optimisation data
        optimisation_data = self._optimisation.optimisation_data

        # Set the list of fidelities of which the samples should be plotted.
        if fid_samples_plot is None:
            fid_samples_plot = list(range(self.config.num_fidelities))

        # Get list of iterations to plot
        n_iter = len(
            [
                k
                for k in optimisation_data
                if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
            ]
        )
        iterations_plot = [
            k
            for k in optimisation_data
            if isinstance(k, int) or (isinstance(k, str) and k.isdigit())
        ]
        if up_to is not None:
            iterations_plot = [k for k in iterations_plot if int(k) < int(up_to)]

        # Filter the iterations to be plotted (compare as int, so that both int
        # keys and the zero-padded str keys of EgoAsync can be selected)
        if iterations is not None:
            iterations_selected = {int(i) for i in iterations}
            iterations_plot = [
                k for k in iterations_plot if int(k) in iterations_selected
            ]

        if not iterations_plot:
            self.config.logger.warning(
                "No iterations to plot for the given up_to/iterations; "
                "no plot is made."
            )
            return

        # Determine if the last iteration to be plotted is the last iteration
        if int(iterations_plot[-1]) < n_iter - 1:
            last_iteration_end = False
        else:
            last_iteration_end = True
        n_iter = len(iterations_plot)

        # Get the best sample point
        best_sample = self.optimum

        # Setup the plotting linspace
        if self.config.dimension == 1:
            x_plot = np.linspace(self.config.lb, self.config.ub, 1000).reshape(-1, 1)
        else:
            n_grid = 100
            x1 = np.linspace(self.config.lb[0], self.config.ub[0], n_grid)
            x2 = np.linspace(self.config.lb[1], self.config.ub[1], n_grid)
            x1, x2 = np.meshgrid(x1, x2)
            x_plot = np.vstack([x1.ravel(), x2.ravel()]).T

        # Get the fidelity predictions of the functions for the surrogate model.
        # If required
        if true_functions_plot is not None and self.config.dimension == 1:
            true_function_values = {
                fid: np.array([self.config.functions[fid](x)[0] for x in x_plot])
                for fid in true_functions_plot
                if fid < self.config.num_fidelities
            }

        # Setup list to keep track of prev iteration added samples
        prev_iteration_added = (
            optimisation_data["init"]["sample_points"]
            if "init" in optimisation_data
            else []
        )

        # Save the current training points for the surrogate to restore after plotting
        surrogate_training_points = copy.deepcopy(self._surrogate.samples)

        # Setup the plot
        cols = 2
        rows = (n_iter + cols - 1) // cols if n_iter > 0 else 1
        fig = plt.figure(figsize=(cols * 5, rows * 3.5))

        # Set up tracker for lines
        true_function_lines = {}
        next_sample_locations_plotted = {}
        sample_points_plotted = {}
        new_points_plotted = {}

        # Setup linestyle tracker for each fidelity
        linestyle_fid = [
            style
            for style in lineStyles.keys()
            if style not in ["-", "", " ", "None", "none"]
        ]

        # Add custom linestyles if not enough available for the number of fidelities
        if len(linestyle_fid) < self.config.num_fidelities:
            for i in range(self.config.num_fidelities - len(linestyle_fid)):
                linestyle_fid.append((0, (5, i * 2)))

        # Setup marker tracker for each fidelity
        marker_fid = [
            marker for marker in Line2D.markers if marker not in ["*", ",", "o"]
        ]

        # Add subplot for every iteration
        for k, iteration_key in enumerate(iterations_plot):
            prev_samples = (
                optimisation_data[iterations_plot[k - 1]]["sample_points"]
                if k > 0
                else (
                    optimisation_data["init"]["sample_points"]
                    if "init" in optimisation_data
                    else []
                )
            )
            current_samples = optimisation_data[iteration_key]["sample_points"]
            k_samples = {
                fid: [
                    p
                    for point in current_samples[fid]
                    for p in prev_samples.get(fid, [])
                    if np.all(p["coord"] == point["coord"])
                ]
                for fid in current_samples
            }
            next_location_k = {
                fid: [
                    p
                    for p in current_samples[fid]
                    if not any(
                        np.allclose(p["coord"], prev_point["coord"])
                        for prev_point in prev_samples.get(fid, [])
                    )
                ]
                for fid in current_samples
            }

            # Add the points to the surrogate for plotting
            self._surrogate.samples = k_samples
            self._surrogate._trained = False
            self._surrogate.train()

            # Get the predictions from the surrogate
            surrogate_pred = self._surrogate.predict_values(x_plot)
            if self.config.dimension == 1:
                surrogate_pred = surrogate_pred.flatten()
            if hasattr(self._surrogate, "predict_variance"):
                surrogate_var = self._surrogate.predict_variance(x_plot).flatten()

            # Get the infill values
            try:
                infill_values = [
                    self._infill.calc_infill_value(x, self._surrogate) for x in x_plot
                ]
            # Except for random infill which does not have infill values
            except NotImplementedError:
                infill_values = None

            # Add subplot for this iteration
            ax = fig.add_subplot((n_iter + 1) // 2, 2, k + 1)

            # Plot for 1D
            if self.config.dimension == 1:
                # plot the true function(s) if required with each a different linestyle
                if true_functions_plot is not None:
                    for fid, values in true_function_values.items():
                        (true_function_line,) = ax.plot(
                            x_plot,
                            values,
                            color="tab:green",
                            label=f"True function (fid {fid})",
                            linestyle=linestyle_fid[fid],
                        )
                        true_function_lines[fid] = true_function_line

                # Plot the surrogate mean prediction
                (surrogate_line,) = ax.plot(
                    x_plot,
                    surrogate_pred,
                    color="tab:blue",
                    label="Surrogate prediction",
                )

                # Plot the surrogate variance if available
                if hasattr(self._surrogate, "predict_variance"):
                    surr_var_fill = ax.fill_between(
                        x_plot.flatten(),
                        surrogate_pred - 1.96 * np.sqrt(surrogate_var),
                        surrogate_pred + 1.96 * np.sqrt(surrogate_var),
                        color="tab:blue",
                        alpha=0.35,
                        label="Surrogate 95% CI",
                    )

                # Plot the sample points:
                for fid, points in k_samples.items():
                    if fid not in fid_samples_plot:
                        continue
                    # Flatten the 1-element arrays, so matplotlib gets scalars
                    sample_points_plot = ax.scatter(
                        np.ravel(
                            [
                                p["coord"]
                                for p in points
                                if not _contains_point(
                                    p, prev_iteration_added.get(fid, [])
                                )
                            ]
                        ),
                        np.ravel(
                            [
                                p["value"]
                                for p in points
                                if not _contains_point(
                                    p, prev_iteration_added.get(fid, [])
                                )
                            ]
                        ),
                        label=f"Samples (fid {fid})",
                        color="tab:orange",
                        marker=marker_fid[fid % len(marker_fid)],
                        s=100,
                    )
                    sample_points_plotted[fid] = sample_points_plot

                # Plot the points added in this iteration with a different colour
                for fid, points in prev_iteration_added.items():
                    if fid not in fid_samples_plot:
                        continue
                    new_points_plot = ax.scatter(
                        np.ravel([p["coord"] for p in points]),
                        np.ravel([p["value"] for p in points]),
                        label=f"Samples from previous iterations (fid {fid})",
                        color="k",
                        marker=marker_fid[fid % len(marker_fid)],
                        s=100,
                    )
                    new_points_plotted[fid] = new_points_plot

                # Plot the next sample locations
                if (k != n_iter - 1) or (k == n_iter - 1 and not last_iteration_end):
                    for fid, points in next_location_k.items():
                        next_sample_locations_plot = ax.vlines(
                            np.ravel([p["coord"] for p in points]),
                            ymin=ax.get_ylim()[0],
                            ymax=ax.get_ylim()[1],
                            color="tab:red",
                            label=f"Next sample location (fid {fid})",
                            linestyle=linestyle_fid[fid],
                        )
                        next_sample_locations_plotted[fid] = next_sample_locations_plot

                # Plot the optimum if last iteration
                if k == n_iter - 1 and last_iteration_end:
                    optimum_plot = ax.scatter(
                        np.ravel(best_sample["coord"]),
                        np.ravel(best_sample["value"]),
                        color="tab:purple",
                        label="Optimum found",
                        marker="*",
                        s=200,
                    )

                # Set data for main plot
                ax.grid()
                ax.set_title(_iteration_title(iteration_key))
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")

                # plot the infill value on a secondary y-axis (if not last iteration)
                if (
                    not (k == n_iter - 1 and last_iteration_end)
                    and infill_values is not None
                ):
                    ax2 = ax.twinx()
                    (infill_line,) = ax2.plot(
                        x_plot, infill_values, color="tab:red", label="Infill value"
                    )
                    ax2.set_ylabel("Infill value")

                # Update tracker
                prev_iteration_added = next_location_k

            # Plot for 2D
            elif self.config.dimension == 2:
                # Plot the surrogate
                surrogate_contour = ax.contourf(
                    x1,
                    x2,
                    surrogate_pred.reshape(n_grid, n_grid),
                    levels=50,
                    cmap="viridis",
                    alpha=0.8,
                )

                # Plot the sample points
                # Ensure different symbol per fidelity level
                for fid in k_samples:
                    if fid not in fid_samples_plot:
                        continue
                    sample_coords = [
                        np.asarray(p["coord"]).ravel()
                        for p in k_samples[fid]
                        if not any(
                            np.allclose(p["coord"], prev_point["coord"])
                            for prev_point in prev_iteration_added.get(fid, [])
                        )
                    ]
                    xs = [coord[0] for coord in sample_coords]
                    ys = [
                        coord[1] if coord.size > 1 else np.nan
                        for coord in sample_coords
                    ]
                    sample_points_plot = ax.scatter(
                        xs,
                        ys,
                        label=f"Samples (fid {fid})",
                        color="tab:orange",
                        marker=marker_fid[fid % len(marker_fid)],
                        s=100,
                    )
                    sample_points_plotted[fid] = sample_points_plot

                # Plot the new sample locations
                for fid, points in prev_iteration_added.items():
                    sample_coords = [np.asarray(p["coord"]).ravel() for p in points]
                    xs = [coord[0] for coord in sample_coords]
                    ys = [
                        coord[1] if coord.size > 1 else np.nan
                        for coord in sample_coords
                    ]
                    new_points_plot = ax.scatter(
                        xs,
                        ys,
                        label=f"Samples from previous iterations (fid {fid})",
                        color="k",
                        marker=marker_fid[fid % len(marker_fid)],
                        s=100,
                    )
                    new_points_plotted[fid] = new_points_plot

                # Plot the next sample locations
                if (k != n_iter - 1) or (k == n_iter - 1 and not last_iteration_end):
                    for fid, points in next_location_k.items():
                        sample_coords = [np.asarray(p["coord"]).ravel() for p in points]
                        xs = [coord[0] for coord in sample_coords]
                        ys = [
                            coord[1] if coord.size > 1 else np.nan
                            for coord in sample_coords
                        ]
                        next_sample_locations_plot = ax.scatter(
                            xs,
                            ys,
                            color="tab:red",
                            label=f"Next sample location (fid {fid})",
                            marker=marker_fid[fid % len(marker_fid)],
                            s=100,
                        )
                        next_sample_locations_plotted[fid] = next_sample_locations_plot

                # Plot the optimum if last iteration
                if k == n_iter - 1 and last_iteration_end:
                    xs, ys = best_sample["coord"][0], best_sample["coord"][1]
                    optimum_plot = ax.scatter(
                        xs,
                        ys,
                        color="tab:purple",
                        label="Optimum found",
                        marker="*",
                        s=200,
                    )

                # Add z-legend to the side of the subplot
                ax.set_title(_iteration_title(iteration_key))
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                cbar = plt.colorbar(surrogate_contour, ax=ax)
                cbar.set_label("Surrogate prediction")

                # Update tracker
                prev_iteration_added = next_location_k

        # Add legend to the overall figure
        handles, labels = [], []
        if self.config.dimension == 1:
            if true_functions_plot is not None:
                for fid in true_function_lines:
                    handles.append(true_function_lines[fid])
                    labels.append(f"True function (fid {fid})")
            handles.append(surrogate_line)
            labels.append("Surrogate prediction")
            if hasattr(self._surrogate, "predict_variance"):
                handles.append(surr_var_fill)
                labels.append("Surrogate 95% CI")
            try:
                handles.append(infill_line)
                labels.append("Infill value")
            except UnboundLocalError:
                pass

        for fid in sample_points_plotted:
            handles.append(sample_points_plotted[fid])
            labels.append(f"Samples (fid {fid})")
        for fid in new_points_plotted:
            handles.append(new_points_plotted[fid])
            labels.append(f"Added samples (fid {fid})")
        for fid in next_sample_locations_plotted:
            handles.append(next_sample_locations_plotted[fid])
            labels.append(f"Next sample location (fid {fid})")
        if last_iteration_end:
            handles.append(optimum_plot)
            labels.append("Optimum found")

        ncol_legend = 4
        n_legend_rows = math.ceil(len(handles) / ncol_legend)
        total_height_in = rows * 3.5

        SUPTITLE_H_IN = 0.28
        LEGEND_ROW_H_IN = 0.22
        LEGEND_PAD_IN = 0.12
        GAP_IN = 0.12

        legend_h_in = n_legend_rows * LEGEND_ROW_H_IN + LEGEND_PAD_IN
        header_h_in = SUPTITLE_H_IN + 2 * GAP_IN + legend_h_in
        top_val = max(0.5, 1.0 - header_h_in / total_height_in)

        fig.suptitle(title, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, top_val])

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        sup_bottom_frac = inv.transform(
            (0, fig._suptitle.get_window_extent(renderer).y0)
        )[1]

        legend = fig.legend(
            handles,
            labels,
            loc="center",
            ncol=ncol_legend,
            frameon=True,
            bbox_to_anchor=(0.5, (sup_bottom_frac + top_val) / 2),
            bbox_transform=fig.transFigure,
        )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_bottom_frac = inv.transform((0, legend.get_window_extent(renderer).y0))[1]

        plt.tight_layout(rect=[0, 0, 1, leg_bottom_frac - 0.01])  # was 0.01

        # Save the plot
        if filename is not None:
            if not os.path.isabs(filename):
                filename = os.path.join(self.config.get("save_folder"), filename)
            dirpath = os.path.dirname(filename)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            plt.savefig(filename)

        # Show the plot
        if show:
            plt.show()
        plt.close()

        # restore the surrogate training points to before plotting
        self._surrogate.samples = surrogate_training_points
        self._surrogate._trained = False
