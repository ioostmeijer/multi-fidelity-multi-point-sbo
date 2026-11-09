import numpy as np

from ..surrogate import CoKRG
from ..utils import Config, get_logger
from .GEI import GEI


class FilteredGEI(GEI):
    r"""Filtered Generalised Expected Improvement infill strategy.

    Implements a filtered version of the Generalised Expected Improvement (GEI)
    infill strategy for multi-fidelity optimisation introduced in [1].
    This strategy selects new points to sample based on the GEI criterion, while also
    determining the appropriate fidelity level (high-fidelity or low-fidelity) for
    each point based on a calculated threshold.

    The sample location $x$ maximises the GEI (see `GEI`). The point is sampled at the
    high fidelity if the prediction there is below the threshold $T(x)$, and at the low
    fidelity otherwise:

    $$
    l(x) = \begin{cases}
        0 \text{ (HF)} & \text{if } \hat{y}(x) < T(x) \\
        1 \text{ (LF)} & \text{otherwise}
    \end{cases}
    $$

    The threshold is calculated in `_calc_threshold`.

    Notes
    -----
    This infill strategy is not functional yet: it needs the CoKRG surrogate, which
    is not implemented. Creating it for more than one fidelity raises a
    NotImplementedError. With one fidelity it is replaced by GEI. It has not been
    verified.

    References
    ----------
    [1] Z. Guo, Q. Wang, L. Song, and J. Li, “Parallel multi-fidelity expected
    improvement method for efficient global optimization,” Structural and
    Multidisciplinary Optimization, vol. 64, no. 3, pp. 1457–1468, Sep. 2021,
    ISSN: 16151488. DOI: 10.1007/S00158-021-02931-1. [Online].
    Available: https://link.springer.com/article/10.1007/s00158-021-02931-1.
    """

    def __new__(cls, *args, **kwargs) -> "FilteredGEI | GEI":
        """Create a new instance of FilteredGEI or GEI based on number of fidelities.

        If the number of fidelities is 1, return an instance of GEI. Otherwise, return
        an instance of FilteredGEI.

        Returns
        -------
        FilteredGEI | GEI
            The FilteredGEI object or GEI object.

        Raises
        ------
        ValueError
            If no config is passed as keyword argument.
        """
        config: Config = kwargs.get("config")
        if config is None:
            (kwargs.get("logger") or get_logger()).error(
                "Config object must be provided to FilteredGEI."
            )
            raise ValueError("Config object is required for FilteredGEI.")
        num_fid = config.num_fidelities
        if num_fid == 1:
            # update the settings to be updated
            config.update_used_algorithms({**config.used_algorithms, "infill": "GEI"})
            config.update_settings({"infill": config.settings.get("infill", {})})
            return GEI(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the Filtered Generalised Expected Improvement infill strategy.

        Raises
        ------
        NotImplementedError
            Always: FilteredGEI needs the CoKRG surrogate, which is not implemented.
        """
        # initialise the GEI class
        super().__init__(*args, **kwargs)

        msg = (
            "FilteredGEI is not functional yet: it needs the CoKRG surrogate, which "
            "is not implemented."
        )
        self.config.logger.error(msg)
        raise NotImplementedError(msg)

        if self.config.num_fidelities > 2:
            self.config.logger.error("FilteredGEI only supports 2 fidelity levels.")
            raise ValueError("FilteredGEI only supports 2 fidelity levels.")

        # Warn if constraints in config
        # if self.config.equality_constraints or self.config.inequality_constraints:
        #     self.config.logger.warning(
        #         "Constraints are passed to the optimisation, however the selected "
        #         "infill strategy does not support constraints, and will thus be"
        #         "ignored."
        #     )

    def get_new_points(
        self,
        surrogate: CoKRG,
        *args,
        samples_per_iteration: None | dict[int, list[float]] = None,
        **kwargs,
    ) -> tuple[list[list[dict]], float]:
        """Generate new points based on the Filtered GEI strategy.

        Parameters
        ----------
        surrogate
            The CoKriging surrogate model.
        samples_per_iteration
            A dictionary where keys are iteration indices and values are lists of
            sampled high-fidelity values in that iteration. If None, the mean of all
            high-fidelity samples will be used for threshold calculation.

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
            If the surrogate is not a CoKRG surrogate.
        ValueError
            If the surrogate has more than 2 fidelity levels.
        """
        if not isinstance(surrogate, CoKRG):
            self.config.logger.error("surrogate must be of type CoKRG.")
            raise TypeError("surrogate must be of type CoKRG")
        if surrogate.config.num_fidelities > 2:
            self.config.logger.error("FilteredGEI only supports 2 fidelity levels.")
            raise ValueError("FilteredGEI only supports 2 fidelity levels.")

        # Determine the sample location using GEI
        best_value = surrogate.get_optimum()["value"]
        x_location, neg_gei_value = self._select_sample_point_optimisation(
            lambda x: -self.calc_infill_value(x, surrogate, best_value=best_value)
        )
        x_location = np.asarray(x_location)
        gei_value = -neg_gei_value

        # Determine whether to sample HF or LF based on threshold
        threshold = self._calc_threshold(
            surrogate, x_location.reshape(1, -1), best_value, samples_per_iteration
        )[0]

        if surrogate.predict(x_location, return_variance=False)[0] < threshold:
            fid = 0  # Sample HF
        else:
            fid = 1

        task = [{"coord": x_location, "fid": fid}]
        return [self._return_nested_samples(task)], gei_value

    def _calc_threshold(
        self,
        surrogate: CoKRG,
        x: np.ndarray,
        best_value: float | None = None,
        samples_per_iteration: None | dict[int, list[float]] = None,
    ) -> np.ndarray:
        r"""Calculate the threshold value for fidelity selection at point x.

        The threshold is a weighted average of the optimum $y_{\min}$ and the mean
        $\bar{y}_h$ of the high-fidelity samples:

        $$
        T(x) = \omega(x) \, y_{\min} + \left(1 - \omega(x)\right) \bar{y}_h,
        \qquad \omega(x) = \frac{1}{1 + \sqrt{\hat{s}^2(x) / \hat{s}_l^2(x)}}
        $$

        with $\hat{s}^2(x)$ the variance of the co-Kriging discrepancy model and
        $\hat{s}_l^2(x)$ the variance of the low-fidelity model ($\hat{s}_l^2$ is
        clipped to at least $10^{-12}$). With `samples_per_iteration` (the HF values
        $y_{h,i,j}$, $j = 1, \dots, n_i$, sampled in iteration $i = 0, \dots, t - 1$),
        the mean is calculated per iteration:

        $$
        \bar{y}_h = \frac{1}{t + 1} \sum_{i=0}^{t-1} \frac{1}{n_i}
            \sum_{j=1}^{n_i} y_{h,i,j}
        $$

        Without it, $\bar{y}_h$ is the mean of all HF samples. Not verified against
        [1] yet.

        Parameters
        ----------
        surrogate
            The CoKriging surrogate model.
        x
            The input point(s) where the threshold is to be calculated.
        best_value
            The current best-known value of the objective function.
        samples_per_iteration
            A dictionary where keys are iteration indices and values are lists of
            sampled high-fidelity values in that iteration.
            If None, the mean of all high-fidelity samples will be used.

        Returns
        -------
        np.ndarray
            The calculated threshold value(s) at point x.

        Raises
        ------
        TypeError
            If the surrogate is not a CoKRG surrogate.
        ValueError
            If no best value is given.
        """
        if not isinstance(surrogate, CoKRG):
            self.config.logger.error("surrogate must be of type CoKRG.")
            raise TypeError("surrogate must be of type CoKRG")

        if best_value is None:
            if hasattr(surrogate, "get_optimum"):
                best_value = surrogate.get_optimum()["value"]
            else:
                self.config.logger.error(
                    "Best value must be provided or obtainable from surrogate."
                )
                raise ValueError("Best value is required for threshold calculation.")
        y_hf_min = best_value

        # Note: unverified. Without per-iteration samples (e.g. in asynchronous
        # optimisation) the mean over all HF samples is used instead of the
        # per-iteration average from [1].
        if samples_per_iteration is None:
            y_hf_mean = np.mean([point["value"] for point in surrogate.samples[0]])
        else:
            current_iter = len(samples_per_iteration) + 1
            sum_val = sum(
                (1 / len(samples_per_iteration[i]) * sum(samples_per_iteration[i]))
                for i in range(len(samples_per_iteration))
            )
            y_hf_mean = 1 / current_iter * sum_val

        # Variance of the coKRG discrepancy model
        variance_full = surrogate.predict_variance(x)

        # Variance of the LF model
        variance_lf = surrogate.predict_variance_fidelity(x, fidelity=1)

        # avoid division by zero
        eps = 1e-12
        safe_variance_lf = np.maximum(variance_lf, eps)
        if np.any(variance_lf < eps):
            self.config.logger.warning(
                f"variance_lf is zero or very small (min={np.min(variance_lf)}). "
                f"Clipping to {eps} to avoid division by zero."
            )

        # Weighting factor (note: implementation not yet verified against [1])
        omega = 1 / (1 + np.sqrt(variance_full / safe_variance_lf))

        return omega * y_hf_min + (1 - omega) * y_hf_mean
