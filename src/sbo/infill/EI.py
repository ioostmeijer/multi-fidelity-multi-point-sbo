import numpy as np
from scipy.stats import norm

from ..surrogate import SurrogateBaseWithVariance, SurrogateBaseWithVariancePerFidelity
from .Base import InfillBase


class EI(InfillBase):
    r"""Expected Improvement (EI) infill strategy.

    This infill strategy selects new points to sample based on the Expected Improvement
    criterion, which balances exploration and exploitation by considering both the
    predicted mean and uncertainty of the surrogate model. The point selected maximises
    the expected improvement over the current best-known value:

    $$
    \mathrm{EI}(x) = \left(y_{\min} - \hat{y}(x)\right) \Phi(z)
        + \hat{s}(x) \, \phi(z),
    \qquad z = \frac{y_{\min} - \hat{y}(x)}{\hat{s}(x)}
    $$

    with $\hat{y}(x)$ the predicted mean and $\hat{s}^2(x)$ the predicted variance of
    the surrogate, $y_{\min}$ the best sampled value at the fidelity used, and $\Phi$
    and $\phi$ the cumulative distribution and probability density functions of the
    standard normal distribution.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the Expected Improvement infill strategy."""
        # Initialise the InfillBase class
        super().__init__(*args, **kwargs)

        # EI only supports one new point, make sure this is set correctly
        if self.config["number_new_points"] != 1:
            self.config.logger.warning(
                "Expected Improvement infill strategy only supports one new point, "
                "multiple new points requested will be generated using KB or CL when "
                "using MPI enabled optimisation algorithm."
            )

        # EI only supports single-fidelity optimisation
        if self.config.num_fidelities != 1:
            self.config.logger.info(
                "Expected Improvement infill strategy only supports single-fidelity "
                "optimisation, using only the high-fidelity (HF) fidelity for sampling "
                "by default. Sampling can be done on lower fidelity by passing "
                "fidelity."
            )

        # Warn if constraints in config
        # if isinstance(self, EI):
        #     if self.config.equality_constraints or self.config.inequality_constraints:
        #         self.config.logger.warning(
        #             "Constraints are passed to the optimisation, however the selected"
        #             " infill strategy does not support constraints, and will thus be"
        #             "ignored."
        #         )

    def get_new_points(
        self,
        surrogate: SurrogateBaseWithVariance | SurrogateBaseWithVariancePerFidelity,
        fidelity: int = 0,
    ) -> tuple[list[list[dict]], float]:
        """Generate new points based on the Expected Improvement strategy.

        Parameters
        ----------
        surrogate
            The surrogate model used to evaluate the infill strategy.
        fidelity
            The fidelity level to sample at (default: 0, high-fidelity). Only possible
            with SurrogateBaseWithVariancePerFidelity

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
            If the surrogate does not provide variance estimates, or a lower fidelity is
            requested from a surrogate without variance per fidelity.
        """
        # EI only supports surrogates with variance estimates
        if not isinstance(
            surrogate, SurrogateBaseWithVariance | SurrogateBaseWithVariancePerFidelity
        ):
            self.config.logger.error(
                "surrogate model must provide variance estimates for EI calculation."
            )
            raise TypeError(
                "surrogate model must provide variance estimates for EI calculation."
            )

        if fidelity != 0 and not isinstance(
            surrogate, SurrogateBaseWithVariancePerFidelity
        ):
            msg = (
                "surrogate does not support variance per fidelity required for "
                "sampling at lower fidelities."
            )
            self.config.logger.error(msg)
            raise TypeError(msg)

        # Calculate the best point to sample (maximising EI)
        best_value = min(p["value"] for p in surrogate.samples[fidelity])

        def _optim_function(
            x: np.ndarray,
        ) -> float:
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
                np.atleast_2d(x), surrogate, best_value=best_value, fidelity=fidelity
            )

        sample_coord, infill_value = self._select_sample_point_optimisation(
            _optim_function
        )
        task = [{"coord": sample_coord, "fid": fidelity}]
        return [self._return_nested_samples(task)], -infill_value

    def calc_infill_value(
        self,
        x: np.ndarray,
        surrogate: SurrogateBaseWithVariance | SurrogateBaseWithVariancePerFidelity,
        *args,
        best_value: float | None = None,
        fidelity: int = 0,
        **kwargs,
    ) -> float:
        r"""Calculate EI value. Accepts optional keyword-only `best_value`.

        `best_value` can be provided by callers of EI; if not provided it can be
        obtained from `context`/`kwargs` or computed from the surrogate.

        The EI is calculated with the formula in the class docstring. A point without
        variance ($\hat{s}(x) \le 10^{-12}$) gets
        $\mathrm{EI}(x) = \max\left(y_{\min} - \hat{y}(x), 0\right)$; if no point in
        `x` has variance, the EI is 0.

        Parameters
        ----------
        x
            The coordinates of the point to evaluate.
        surrogate
            The surrogate model used to evaluate the infill strategy.
        best_value
            The best known objective function value. If None, it will be obtained
            from the surrogate model if possible.
        fidelity
            The fidelity level to sample at (default: 0, high-fidelity). Only possible
            when surrogate is SurrogateBaseWithVariancePerFidelity

        Returns
        -------
        float
            The Expected Improvement value at the given point.

        Raises
        ------
        TypeError
            If the surrogate does not provide variance estimates, or a lower fidelity is
            requested from a surrogate without variance per fidelity.
        ValueError
            If the fidelity level is invalid, or no samples are available to determine
            the best value.
        """
        # Check if surrogate provides variance estimates
        if not isinstance(
            surrogate, SurrogateBaseWithVariance | SurrogateBaseWithVariancePerFidelity
        ):
            self.config.logger.error(
                "surrogate model must provide variance estimates for EI calculation."
            )
            raise TypeError(
                "surrogate model must provide variance estimates for EI calculation."
            )

        if fidelity != 0 and not isinstance(
            surrogate, SurrogateBaseWithVariancePerFidelity
        ):
            msg = (
                "surrogate does not support variance per fidelity required for "
                "sampling at lower fidelities."
            )
            self.config.logger.error(msg)
            raise TypeError(msg)

        if fidelity < 0 or fidelity >= self.config.num_fidelities:
            self.config.logger.error(
                "Invalid fidelity level provided for EI calculation."
            )
            raise ValueError("Invalid fidelity level provided for EI calculation.")

        if best_value is None:
            # Get the best function value from the surrogate model if not provided
            try:
                best_value = min(
                    [p["value"] for p in surrogate.samples.get(fidelity, {})]
                )
            except ValueError as e:
                self.config.logger.error(
                    "No samples found in surrogate model to determine best value for "
                    f"EI calculation. Exit with error: {e}"
                )
                raise ValueError(
                    "No samples found in surrogate model to determine best value for "
                    f"EI calculation. Exit with error: {e}"
                ) from e

        mu, variance = surrogate(
            np.atleast_2d(x), return_variance=True, fidelity=fidelity
        )
        neg_tol = -1e-5
        if np.any(variance < neg_tol):
            self.config.logger.debug(
                "Surrogate returned significant negative variance."
            )
        variance = np.maximum(variance, 0.0)
        sigma = np.sqrt(variance)
        if np.all(np.abs(sigma) <= 1e-12):
            return 0.0

        improvement = best_value - mu
        has_variance = sigma > 1e-12
        # Avoid dividing by zero; points without variance get z = 0 (overwritten)
        z = np.divide(
            improvement,
            sigma,
            out=np.zeros_like(improvement, dtype=float),
            where=has_variance,
        )

        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        # Without variance the expected improvement is the (positive) improvement
        ei = np.where(has_variance, ei, np.maximum(improvement, 0.0))
        ei = ei.flatten()

        if len(ei) == 1:
            return float(ei[0])
        return ei
