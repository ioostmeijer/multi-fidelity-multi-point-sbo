from typing import Callable

import numpy as np
import scipy.stats.qmc
from scipy.optimize import differential_evolution

from ..surrogate import SurrogateBase
from ..utils import Base


class InfillBase(Base):
    """Base class for the infill strategies.

    This class is the base for the different implemented infill strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the InfillBase class."""
        # Initialise the Base class
        super().__init__(*args, **kwargs)

    def get_new_points(
        self, surrogate: SurrogateBase
    ) -> tuple[list[list[dict]], float]:
        """Generate new points based on the infill strategy.

        Parameters
        ----------
        surrogate
            The surrogate model used to evaluate the infill strategy.

        Returns
        -------
        tuple[list[list[dict]], float]
            The new points and the infill value. The new points are a list of
            tasks; each task is a list of points to evaluate together
            ({'coord': np.ndarray, 'fid': int}, fid 0 is HF). The infill value is
            the value of the infill criterion at the selected point.

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        self.config.logger.error("get_new_points method not implemented in subclass.")
        raise NotImplementedError("Must be implemented in subclass")

    def __call__(self, surrogate: SurrogateBase) -> tuple[list[list[dict]], float]:
        """Call override, which calls the get_new_points method.

        Parameters
        ----------
        surrogate
            The surrogate model used to evaluate the infill strategy.

        Returns
        -------
        tuple[list[list[dict]], float]
            The new points and the infill value. The new points are a list of
            tasks; each task is a list of points to evaluate together
            ({'coord': np.ndarray, 'fid': int}, fid 0 is HF). The infill value is
            the value of the infill criterion at the selected point.
        """
        return self.get_new_points(surrogate=surrogate)

    def _return_nested_samples(self, samples: list[dict]) -> list[dict]:
        """Return a nested list of samples if required.

        A nested list of samples is a list containing sample points at the same location
        but at different fidelity levels. The fidelity levels included for a nested
        sample are all the lower fidelity levels from the initial samples fidelity
        level.

        Parameters
        ----------
        samples
            A list of dicts, each dict containing:
                - 'coord': A numpy ndarray representing the coordinates of the point.
                - 'fid': An integer representing the fidelity level (0=HF -> n=LF).

        Returns
        -------
        list[dict]
            A list of nested samples (if required) in the same format as the input
            samples.
        """
        if not self.config.get("nested"):
            return samples

        number_of_fidelities = self.config.num_fidelities
        for point in samples[:]:
            coord = point["coord"]
            fidelity = point["fid"]
            samples.extend(
                {"coord": coord, "fid": f}
                for f in range(fidelity + 1, number_of_fidelities, 1)
            )
        return samples

    def calc_infill_value(
        self, x: np.ndarray, surrogate: SurrogateBase, *args, **kwargs
    ) -> float:
        """Calculate the infill value for a given point using a flexible context.

        Parameters
        ----------
        x
            The input point where the infill value is to be calculated.
        surrogate
            The surrogate model used for predictions.

        Returns
        -------
        float
            The calculated infill value at the given point.

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        self.config.logger.error(
            "calc_infill_value method not implemented in subclass."
        )
        raise NotImplementedError("Must be implemented in subclass")

    def _select_sample_point_optimisation(
        self, optim_func: Callable, constraints: list | None = None
    ) -> tuple[np.ndarray, float]:
        """Helper to select a point to be evaluated.

        This helper function selects one new sample location by minimising the infill
        value returned by the passed optim_func. If the function is to be maximised,
        pass a function like lambda x: -calc_infill_value(x, ...).
        The optimisation is performed using scipy.differential_evolution, with the
        'infill_opt_*' settings. If the optimiser does not report convergence, its
        best point is still used when it is valid. Only if it returns no valid point,
        the best of 1000 fallback points (a grid in 1D, a Latin hypercube in nD) is
        chosen.

        Parameters
        ----------
        optim_func
            The function to be minimised.
        constraints
            Optional scipy constraints passed to differential evolution (e.g.
            config.scipy_constraints()). None or an empty list means no constraints.

        Returns
        -------
        tuple[np.ndarray, float]
            A tuple containing the coordinates of the selected point and the infill
            value at that point, returned by the passed optim_func.
        """
        de_kwargs = self._differential_evolution_settings()
        if constraints:
            de_kwargs["constraints"] = constraints
        result_optim = differential_evolution(
            func=optim_func,
            bounds=self.config.bounds,
            **de_kwargs,
        )

        if result_optim.success:
            return result_optim.x, result_optim.fun

        # Differential evolution often reports success=False when it stops at the
        # maximum number of generations (e.g. on flat infill surfaces) while its best
        # point is still good. Use that point if it is valid.
        if result_optim.x is not None and np.isfinite(result_optim.fun):
            self.config.logger.info(
                "The infill optimisation did not report convergence "
                f"({result_optim.message}); using its best point "
                f"(infill value: {-result_optim.fun})."
            )
            return result_optim.x, result_optim.fun

        # If the optimiser did not return a valid point, fall back to a search over a
        # fixed set of points.
        self.config.logger.warning(
            "The infill optimisation did not return a valid point. "
            "Falling back to a search over 1000 points for point "
            f"selection. (infill value: {-result_optim.fun}, "
            f"success: {result_optim.success})"
        )
        x_test = self._fallback_points()
        infill_test = np.array([optim_func(x) for x in x_test])
        index_max_ei = np.argmin(infill_test)
        return x_test[index_max_ei], infill_test[index_max_ei]

    def _differential_evolution_settings(self) -> dict:
        """Get the keyword arguments for differential evolution of the infill.

        Returns
        -------
        dict
            The keyword arguments for scipy.optimize.differential_evolution, taken
            from the 'infill_opt_*' settings and the 'random_seed' setting.
        """
        return {
            "maxiter": self.config["infill_opt_max_iterations"],
            "popsize": self.config["infill_opt_population_size"],
            "tol": self.config["infill_opt_tolerance"],
            "mutation": self.config["infill_opt_mutation"],
            "recombination": self.config["infill_opt_recombination"],
            "strategy": self.config["infill_opt_strategy"],
            "polish": self.config["infill_opt_polish"],
            "init": self.config["infill_opt_init"],
            "seed": self.config["random_seed"],
        }

    def _fallback_points(self, n_points: int = 1000) -> np.ndarray:
        """Get the points used when the infill optimisation fails.

        In 1D a regular grid is used. In more dimensions a grid of n_points would
        only cover the diagonal, so a Latin hypercube sample scaled to the bounds is
        used instead.

        Parameters
        ----------
        n_points
            The number of points.

        Returns
        -------
        np.ndarray
            The points, with shape (n_points, n_dimensions).
        """
        if self.config.dimension == 1:
            return np.linspace(self.config.lb, self.config.ub, n_points)
        sampler = scipy.stats.qmc.LatinHypercube(
            d=self.config.dimension, seed=self.config["random_seed"]
        )
        return scipy.stats.qmc.scale(
            sampler.random(n=n_points), self.config.lb, self.config.ub
        )
