from math import comb

import numpy as np
from scipy.stats import norm

from ..surrogate import SurrogateBaseWithVariance
from ..types import InfillResult, Task
from .Base import InfillBase


class GEI(InfillBase[SurrogateBaseWithVariance]):
    r"""Generalised Expected Improvement (GEI) infill strategy.

    This infill strategy selects new points to sample based on the Generalised Expected
    Improvement criterion, which extends the traditional Expected Improvement by
    allowing for a parameter 'g' that adjusts the balance between exploration and
    exploitation. The point selected maximises the generalised expected improvement over
    the current best-known value:

    $$
    \mathrm{GEI}_g(x) = \hat{s}(x)^g \sum_{k=0}^{g} (-1)^k \binom{g}{k} u^{g-k} T_k,
    \qquad u = \frac{y_{\min} - \hat{y}(x)}{\hat{s}(x)}
    $$

    with the terms $T_k$ calculated recursively:

    $$
    T_0 = \Phi(u), \qquad T_1 = -\phi(u), \qquad
    T_k = -u^{k-1} \phi(u) + (k - 1) \, T_{k-2} \quad \text{for } k \ge 2
    $$

    Here $\hat{y}(x)$ and $\hat{s}^2(x)$ are the predicted mean and variance of the
    surrogate, $y_{\min}$ is the optimum of the surrogate (not the best sample), and
    $\Phi$ and $\phi$ are the cumulative distribution and probability density functions
    of the standard normal distribution. $g = 0$ gives the probability of improvement
    $\Phi(u)$, and $g = 1$ the Expected Improvement; a higher $g$ gives more global
    exploration.
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

    def get_new_points(self, surrogate: SurrogateBaseWithVariance) -> InfillResult:
        """Generate new points based on the Generalised Expected Improvement strategy.

        Parameters
        ----------
        surrogate
            The surrogate model used to estimate the objective function.

        Returns
        -------
        InfillResult
            The new points and the infill value. The new points are a list of
            tasks (Task); each task is a list of points (Point) to evaluate
            together. The infill value is the value of the infill criterion at
            the selected point.

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
            return -float(
                self.calc_infill_value(
                    np.atleast_2d(x), surrogate, best_value=best_value
                )
            )

        sample_coord, infill_value = self._select_sample_point_optimisation(
            _optim_function
        )

        task: Task = [{"coord": sample_coord, "fid": 0}]
        return [self._return_nested_samples(task)], -infill_value

    def calc_infill_value(
        self,
        x: np.ndarray,
        surrogate: SurrogateBaseWithVariance,
        *args,
        best_value: float | None = None,
        fidelity: int = 0,
        **kwargs,
    ) -> float | np.ndarray:
        r"""Calculate the Generalised Expected Improvement (GEI) value at a given point.

        The GEI is calculated with the formulas in the class docstring. A point without
        variance ($\hat{s}(x) \le 10^{-12}$) gets the limit of the formula for
        $\hat{s}(x) \to 0$: $\left(y_{\min} - \hat{y}(x)\right)^g$ if
        $\hat{y}(x) < y_{\min}$, and 0 otherwise (for $g = 0$: 1 or 0).

        Parameters
        ----------
        x
            The points where the GEI is to be calculated.
            One point with shape (n_dim,), or n points with shape (n, n_dim).
        surrogate
            The surrogate model used to estimate the objective function.
        best_value
            The current best-known value of the objective function. If None, it will be
            retrieved from the surrogate model.
        fidelity
            The fidelity level at which to evaluate the surrogate model.

        Returns
        -------
        float | np.ndarray
            The GEI per point: a float for one point, or an array with
            shape (n,) for n points.

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

        improvement = best_value - mu
        has_variance = sigma > 1e-12
        # Avoid dividing by zero; points without variance get u = 0 (overwritten)
        u = np.divide(
            improvement,
            sigma,
            out=np.zeros_like(improvement, dtype=float),
            where=has_variance,
        )
        t = [norm.cdf(u), -norm.pdf(u)]
        g = self.config.get("infill.g")

        for k in range(2, g + 1):
            tk = -(u ** (k - 1)) * norm.pdf(u) + (k - 1) * t[-2]
            t.append(tk)

        gei = np.zeros_like(mu)
        for k in range(g + 1):
            gei += (-1) ** k * comb(g, k) * (u ** (g - k)) * t[k]
        gei *= sigma**g
        # Without variance the improvement is certain: I^g if there is an improvement
        certain = np.where(improvement > 0.0, np.maximum(improvement, 0.0) ** g, 0.0)
        gei[~has_variance] = certain[~has_variance]

        gei = gei.flatten()
        if len(gei) == 1:
            return float(gei[0])
        return gei
