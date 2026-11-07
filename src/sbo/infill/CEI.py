import numpy as np

from ..surrogate import SurrogateBaseWithVariance, SurrogateBaseWithVariancePerFidelity
from .EI import EI


class CEI(EI):
    """Constrained Expected Improvement (CEI) infill strategy.

    This infill strategy selects new points to sample based on the Constrained
    Expected Improvement criterion, which balances exploration and exploitation by
    considering both the predicted mean and uncertainty of the surrogate model.
    In addition, the feasibility of the new sample point is taken into account.
    The point selected maximises the constrained expected improvement over the
    current best-known value.

    Notes
    -----
    This infill strategy is not finished
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the CEI class.

        Raises
        ------
        NotImplementedError
            Always: CEI is not implemented yet.
        """
        super().__init__(*args, **kwargs)
        raise NotImplementedError("CEI is not yet implemented.")

    def get_new_points(
        self,
        surrogate: SurrogateBaseWithVariance | SurrogateBaseWithVariancePerFidelity,
        fidelity: int = 0,
    ) -> tuple[list[list[dict]], float]:
        """Generate new points based on the Constrained Expected Improvement strategy.

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
                "surrogate model must provide variance estimates for CEI calculation."
            )
            raise TypeError(
                "surrogate model must provide variance estimates for CEI calculation."
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

        # Calculate the best point to sample (maximising CEI) with the shared
        # infill optimisation of the base class
        best_value = min(p["value"] for p in surrogate.samples[fidelity])
        coord, neg_cei_value = self._select_sample_point_optimisation(
            lambda x: -self.calc_infill_value(
                np.atleast_2d(x), surrogate, best_value=best_value, fidelity=fidelity
            ),
            constraints=self.config.scipy_constraints(),
        )

        task = [{"coord": coord, "fid": 0}]
        return [self._return_nested_samples(task)], -neg_cei_value

    def calc_infill_value(
        self,
        x: np.ndarray,
        surrogate: SurrogateBaseWithVariance | SurrogateBaseWithVariancePerFidelity,
        *args,
        best_value: float | None = None,
        fidelity: int = 0,
        **kwargs,
    ) -> float:
        """Calculate the CEI infill values.

        Use the super().calc_infill_values to calculate the EI value at the passed x.
        This value is used to calculate the CEI at this point.

        Parameters
        ----------
        x
            The x-coordinate for which to calculate the CEI value
        surrogate
            The surrogate of the objective function to calculate the CEI value for.
            The surrogates of the expensive constraints are retrieved from the config.
            The inexpensive constraints are taken into account by the optimiser for the
            EI value in the get_new_points_method.
        best_value
            The best seen value used in the calculation. If None is passed, the best
            value will be retrieved from the samples of the passed surrogate.
        fidelity
            The fidelity level at which the CEI value is calculated. Default: 0 (HF)

        Returns
        -------
        float
            The constrained expected improvement at x: the EI multiplied by the
            probability that all constraints are satisfied.

        Raises
        ------
        TypeError
            If the surrogate does not provide variance estimates, or a lower fidelity is
            requested from a surrogate without variance per fidelity.
        ValueError
            If the fidelity level is invalid, or no samples are available to determine
            the best value.
        NotImplementedError
            Always: CEI is not implemented yet.
        """
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
                )

        # Get the EI values
        ei_value = super().calc_infill_value(
            x, surrogate, *args, best_value=best_value, fidelity=fidelity, **kwargs
        )

        # TODO: multiply ei_value by the probability of feasibility predicted by the
        # constraint surrogates, i.e. CEI(x) = EI(x) * prod_i P(g_i(x) <= 0).
        raise NotImplementedError(
            f"CEI is not yet implemented (EI value at x: {ei_value})."
        )
