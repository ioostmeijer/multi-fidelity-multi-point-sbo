import importlib
import inspect

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm

from ..surrogate import SurrogateBaseWithVariancePerFidelity
from .Base import InfillBase


class TwoStep(InfillBase):
    r"""Two-step infill strategy for multi-fidelity optimisation.

    This method utilises a two-step approach to determine the next sampling point:

    1. Determine the sample location $x$ based on a single-fidelity infill strategy.
    2. Determine the cheapest accurate enough fidelity to sample at that location,
        based on the Jensen-Shannon distance (JSD) between the high-fidelity (HF)
        surrogate and each lower-fidelity surrogate at the selected location.

    In step 2, the prediction of fidelity $l$ at $x$ is the normal distribution
    $P_l = \mathcal{N}\left(\hat{y}_l(x), \hat{s}_l^2(x)\right)$, with $l = 0$ the HF.
    The fidelity sampled is the cheapest one (highest index) that is close enough to
    the HF prediction:

    $$
    l^* = \max \left\{ l : \mathrm{JSD}(P_0, P_l) < \tau \right\}
    $$

    with $\tau$ the `jsd_threshold` setting. The HF ($l = 0$) always qualifies.

    The method is introduced by Garbo et al. [1].


    Attributes
    ----------
    single_fid_infill : InfillBase
        An instance of a single-fidelity infill strategy used to determine the
        sample location.

    Notes
    -----
    This infill strategy is designed to work with a surrogate model that provides
    both predictions of the mean and uncertainty (variance) of the function values.

    References
    ----------
    [1] A. Garbo, J. Parekh, T. Rischmann, and P. Bekemeyer, “Multi-Fidelity Adaptive
        Sampling for Surrogate-Based Optimization and Uncertainty Quantification,”
        Aerospace, vol. 11, no. 6, p. 448, 2024, ISSN: 2226-4310.
        DOI: 10.3390/aerospace11060448. [Online].
        Available: https://www.mdpi.com/2226-4310/11/6/448.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the TwoStep infill strategy.

        The single-fidelity infill strategy used to determine the sample location is
        read from the ``infill.single_fid_infill`` setting of the config.

        Raises
        ------
        ImportError
            If the single-fidelity infill strategy can not be found or created.
        """
        # Initialise the InfillBase class
        super().__init__(*args, **kwargs)

        # Configure the single fidelity infill strategy
        single_fid_name = self.config.get("infill.single_fid_infill")

        # Load the single-fidelity infill strategy dynamically
        base_module = importlib.import_module("..infill", package=__package__)
        attr = getattr(base_module, single_fid_name, None)

        # If attribute is a module, import the submodule and search for a suitable class
        if inspect.ismodule(attr):
            submodule = importlib.import_module(
                f"..infill.{single_fid_name}", package=__package__
            )
            cls = getattr(submodule, single_fid_name, None)
        else:
            cls = attr

        if cls is None or not inspect.isclass(cls):
            msg = (
                f"Single-fidelity infill strategy class '{single_fid_name}' "
                f"not found or is not a class in module '{base_module.__name__}'"
            )
            self.config.logger.error(msg)
            raise ImportError(msg)

        # Instantiate the infill strategy
        try:
            self.single_fid_infill = cls(config=self.config)
        except (TypeError, AttributeError) as e:
            msg = (
                f"Error initialising single-fidelity infill strategy "
                f"'{single_fid_name}': {e}"
            )
            self.config.logger.error(msg)
            raise ImportError(msg) from e

        # Warn if constraints in config
        # if self.config.equality_constraints or self.config.inequality_constraints:
        #     self.config.logger.warning(
        #         "Constraints are passed to the optimisation, however the selected "
        #         "infill strategy does not support constraints, and will thus be"
        #         "ignored."
        #     )

    def get_new_points(
        self, surrogate: SurrogateBaseWithVariancePerFidelity
    ) -> tuple[list[list[dict]], float]:
        """Generate new points based on the two-step approach.

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
        ValueError
            If the surrogate does not provide variance estimates per fidelity.
        """
        if not isinstance(surrogate, SurrogateBaseWithVariancePerFidelity):
            self.config.logger.error(
                "TwoStep infill strategy requires a surrogate model that provides "
                "uncertainty estimates per fidelity."
            )
            raise ValueError(
                "TwoStep infill strategy requires a surrogate model that provides "
                "uncertainty estimates per fidelity."
            )

        # Step 1: Determine the sample location using the single-fidelity infill
        samples, infill_value = self.single_fid_infill(surrogate)
        x_new = samples[0][0]["coord"]  # Extract the coordinates of the new point

        # Step 2: Determine the cheapest accurate enough fidelity at that location
        accurate_fidelities = self._get_accurate_enough_fidelities(x_new, surrogate)
        cheapest_fidelity = max(accurate_fidelities)  # LF has higher index

        new_samples = [{"coord": x_new, "fid": cheapest_fidelity}]
        return [self._return_nested_samples(new_samples)], infill_value

    def _get_accurate_enough_fidelities(
        self, x: np.ndarray, surrogate: SurrogateBaseWithVariancePerFidelity
    ) -> list[int]:
        r"""Get the list of fidelities which are accurate enough based on the JSD.

        This method computes the Jensen-Shannon distance (JSD) between the
        high-fidelity (HF) surrogate and each lower-fidelity surrogate at the
        selected location. It returns a list of fidelity levels that are considered
        accurate enough based on a predefined JSD threshold.

        The two normal distributions $P_0$ (HF) and $P_l$ (fidelity $l$) are evaluated
        at `num_jsd_points` points between
        $\min_i \left(\hat{y}_i(x) - 3 \hat{s}_i(x)\right)$ and
        $\max_i \left(\hat{y}_i(x) + 3 \hat{s}_i(x)\right)$, $i \in \{0, l\}$, and
        normalised to sum to 1. The distance (base 2, so between 0 and 1) is

        $$
        \mathrm{JSD}(P_0, P_l) = \sqrt{\frac{1}{2} D_{\mathrm{KL}}(P_0 \parallel M)
            + \frac{1}{2} D_{\mathrm{KL}}(P_l \parallel M)},
        \qquad M = \frac{1}{2} (P_0 + P_l)
        $$

        with $D_{\mathrm{KL}}$ the Kullback-Leibler divergence. Fidelity $l$ is
        accurate enough if $\mathrm{JSD}(P_0, P_l) < \tau$ (`jsd_threshold`). A zero
        predicted variance is replaced by $10^{-10}$.

        Parameters
        ----------
        x
            The coordinates of the point to evaluate.
        surrogate
            The surrogate model used to evaluate the infill strategy.

        Returns
        -------
        list[int]
            A list of fidelity levels that are accurate enough.
        """
        # Ensure x is 2D for prediction
        x_row = np.asarray(x).reshape(1, -1)

        mu_hf = surrogate.predict_values(x_row, fidelity=0)
        var_hf = surrogate.predict_variance_fidelity(x_row, fidelity=0)

        # Extract scalar values from arrays
        mu_hf = float(np.ravel(mu_hf)[0])
        var_hf = float(np.ravel(var_hf)[0])

        if var_hf == 0.0:
            var_hf = 1e-10  # Prevent division by zero

        # HF is always accurate enough, since it is taken as true
        accurate_fidelities = [0]

        for fid in range(1, self.config.num_fidelities):
            mu_fid = surrogate.predict_values(x_row, fidelity=fid)
            var_fid = surrogate.predict_variance_fidelity(x_row, fidelity=fid)
            mu_fid = float(np.ravel(mu_fid)[0])
            var_fid = float(np.ravel(var_fid)[0])
            if var_fid == 0.0:
                var_fid = 1e-10  # Prevent division by zero

            # Calculate 3-sigma bounds
            lb = min(mu_hf - 3 * np.sqrt(var_hf), mu_fid - 3 * np.sqrt(var_fid))
            ub = max(mu_hf + 3 * np.sqrt(var_hf), mu_fid + 3 * np.sqrt(var_fid))

            # Create a linspace between the bounds
            x_linspace = np.linspace(lb, ub, self.config.get("infill.num_jsd_points"))

            # Calculate the probability distributions
            p_hf = norm.pdf(x_linspace, loc=mu_hf, scale=np.sqrt(var_hf))
            p_fid = norm.pdf(x_linspace, loc=mu_fid, scale=np.sqrt(var_fid))

            # Normalise the distributions
            s_hf = np.sum(p_hf)
            if not np.isfinite(s_hf) or s_hf <= 0:
                p_hf = np.full_like(p_hf, 1.0 / p_hf.size)
            else:
                p_hf /= s_hf

            s_fid = np.sum(p_fid)
            if not np.isfinite(s_fid) or s_fid <= 0:
                p_fid = np.full_like(p_fid, 1.0 / p_fid.size)
            else:
                p_fid /= s_fid

            # Calculate the Jensen-Shannon distance (base 2, so in [0, 1])
            jsd = jensenshannon(p_hf, p_fid, base=2)

            # Check if the fidelity is accurate enough
            if jsd < self.config.get("infill.jsd_threshold"):
                accurate_fidelities.append(fid)

        return accurate_fidelities

    def calc_infill_value(
        self,
        x: np.ndarray,
        surrogate: SurrogateBaseWithVariancePerFidelity,
        *args,
        **kwargs,
    ) -> float:
        """Calculate the infill value for a point using the single-fidelity infill.

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
        """
        # Use the single-fidelity infill to calculate the infill value at the point
        return self.single_fid_infill.calc_infill_value(x, surrogate)
