from math import comb

import numpy as np
from scipy.stats import norm

from ..surrogate import SurrogateBaseWithVariance
from .Base import InfillBase


class GEI(InfillBase):
    """Generalised Expected Improvement (GEI) infill strategy.

    This infill strategy selects new points to sample based on the Generalised Expected
    Improvement criterion, which extends the traditional Expected Improvement by
    allowing for a parameter 'g' that adjusts the balance between exploration and
    exploitation. The point selected maximises the generalised expected improvement over
    the current
    best-known value.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the Generalised Expected Improvement infill strategy."""
        # Initialise the InfillBase class
        super().__init__(*args, **kwargs)

        # Warn if constraints in config
        # if self.config.equality_constraints or self.config.inequality_constraints:
        #     self.config.logger.warning(
        #         "Constraints are passed to the optimisation, however the selected "
        #         "infill strategy does not support constraints, and will thus be"
        #         "ignored."
        #     )

    def get_new_points(
        self, surrogate: SurrogateBaseWithVariance
    ) -> tuple[list[list[dict]], float]:
        """Generate new points based on the Generalised Expected Improvement strategy.

        Parameters
        ----------
        surrogate
            The surrogate model used to estimate the objective function.

        Returns
        -------
        tuple[list[list[dict]], float]
            The new points and the infill value. The new points are a list of
            tasks; each task is a list of points to evaluate together
            ({'coord': np.ndarray, 'fid': int}, fid 0 is HF). The infill value is
            the value of the infill criterion at the selected point.

        Raises
        ------
        TypeError
            If the surrogate does not provide variance estimates.
        AttributeError
            If the surrogate can not return its optimum (no get_optimum method).
        """
        if not isinstance(surrogate, SurrogateBaseWithVariance):
            self.config.logger.error(
                "surrogate must be of (sub)type SurrogateBaseWithVariance"
            )
            raise TypeError("surrogate must be of (sub)type SurrogateBaseWithVariance")

        if not hasattr(surrogate, "get_optimum"):
            self.config.logger.error(
                "surrogate model does not support optimum retrieval required for GEI."
            )
            raise AttributeError(
                "surrogate model does not support optimum retrieval required for GEI."
            )
        best_value = surrogate.get_optimum()["value"]

        def _optim_function(x: np.ndarray) -> float:
            """The function to minimise to find the new sample location.

            Parameters
            ----------
            x
                The candidate sample location.

            Returns
            -------
            float
                The negative infill value at x.
            """
            return -self.calc_infill_value(
                np.atleast_2d(x), surrogate, best_value=best_value
            )

        sample_coord, infill_value = self._select_sample_point_optimisation(
            _optim_function
        )

        task = [{"coord": sample_coord, "fid": 0}]
        return [self._return_nested_samples(task)], -infill_value

    def calc_infill_value(
        self,
        x: np.ndarray,
        surrogate: SurrogateBaseWithVariance,
        *args,
        best_value: float | None = None,
        fidelity: int = 0,
        **kwargs,
    ) -> float:
        """Calculate the Generalised Expected Improvement (GEI) value at a given point.

        Parameters
        ----------
        x
            The input point where the GEI is to be calculated.
        surrogate
            The surrogate model used to estimate the objective function.
        best_value
            The current best-known value of the objective function. If None, it will be
            retrieved from the surrogate model.
        fidelity
            The fidelity level at which to evaluate the surrogate model.

        Returns
        -------
        float
            The calculated GEI value at the input point.

        Raises
        ------
        TypeError
            If the surrogate does not provide variance estimates.
        """
        if best_value is None:
            # Get the best function value from the surrogate model if not provided
            best_value = surrogate.get_optimum()["value"]

        if not isinstance(surrogate, SurrogateBaseWithVariance):
            msg = "surrogate model must provide variance estimates for GEI calculation."
            self.config.logger.error(msg)
            raise TypeError(msg)

        mu, variance = surrogate(
            np.atleast_2d(x), return_variance=True, fidelity=fidelity
        )
        neg_tol = -1e-5
        if np.any(variance < neg_tol):
            self.config.logger.error(
                "Surrogate returned significant negative variance."
            )
        variance = np.maximum(variance, 0.0)
        sigma = np.sqrt(variance)
        if np.all(np.abs(sigma) <= 1e-12):
            return 0.0

        u = (best_value - mu) / sigma
        t = [norm.cdf(u), -norm.pdf(u)]
        g = self.config.get("infill.g")

        for k in range(2, g + 1):
            tk = -(u ** (k - 1)) * norm.pdf(u) + (k - 1) * t[-2]
            t.append(tk)

        gei = np.zeros_like(mu)
        for k in range(g + 1):
            gei += (-1) ** k * comb(g, k) * (u ** (g - k)) * t[k]
        gei *= sigma**g

        gei = gei.flatten()
        if len(gei) == 1:
            return float(gei[0])
        return gei
