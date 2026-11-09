import numpy as np
from scipy.linalg import LinAlgError, cholesky, qr, solve_triangular
from scipy.optimize import differential_evolution
from smt.surrogate_models import KRG as SMT_KRG

from ..DefaultSettings.surrogate.KRIGING import SMT_DEFAULT_NUGGET
from ..utils import Config, get_logger
from .Base import SurrogateBaseWithVariancePerFidelity


class HK(SurrogateBaseWithVariancePerFidelity):
    r"""Hierarchical Kriging (HK) surrogate model for multi-fidelity.

    This surrogate model implements the Hierarchical Kriging approach for
    multi-fidelity surrogate modelling. It builds a hierarchy of Kriging models,
    where the lowest fidelity model is an SMT KRG model, and higher fidelity models
    build upon the lower fidelity models.
    The implemented method was introduced by Han and Görtz [1].

    Each fidelity $l$ above the lowest one (fidelity 0 is the HF) uses the prediction
    of the next lower fidelity $\hat{y}_{l+1}$ as its trend, scaled by $\beta_0$:

    $$
    \hat{y}_l(x) = \beta_0 \, \hat{y}_{l+1}(x)
        + \mathbf{r}(x)^\top \mathbf{R}^{-1} (\mathbf{y} - \beta_0 \mathbf{F})
    $$

    with $\mathbf{y}$ the $n$ samples of fidelity $l$, $\mathbf{F}$ the prediction
    $\hat{y}_{l+1}$ at these samples, $\mathbf{R}$ their correlation matrix and
    $\mathbf{r}(x)$ the correlations between $x$ and the samples. The scaling factor
    and the process variance follow from generalised least squares:

    $$
    \beta_0 = \left(\mathbf{F}^\top \mathbf{R}^{-1} \mathbf{F}\right)^{-1}
        \mathbf{F}^\top \mathbf{R}^{-1} \mathbf{y},
    \qquad
    \hat{\sigma}^2 = \frac{1}{n} (\mathbf{y} - \beta_0 \mathbf{F})^\top
        \mathbf{R}^{-1} (\mathbf{y} - \beta_0 \mathbf{F})
    $$

    With a cubic spline or Gaussian exponential correlation function by Lophaven et al.
    [2] (see `_vectorised_correlation`).

    Raises
    ------
        ValueError:
            If the Config object is not provided during initialisation.

    References
    ----------
    [1] Zhong Hua Han and Stefan Görtz.
    “Hierarchical kriging model for variable-fidelity surrogate modeling”.
    In: AIAA Journal 50.9 (Sept. 2012), pp. 1885–1896. ISSN: 00011452.
    DOI: 10.2514/1.J051354. URL: https://doi.org/10.2514/1.J051354.

    [2] Søren Nymand Lophaven, Hans Bruun Nielsen, and Jacob Søndergaard.
    DACE - A Matlab Kriging Toolbox, Version 2.0. Vol. IMM-TR-2002-12.
    Informatics and Mathematical Modelling, Technical University of Denmark, DTU, 2002.
    URL: https://orbit.dtu.dk/en/publications/dace-a-matlab-kriging-toolbox-version-20/.
    """

    supports_fidelity_prediction = True

    def __new__(cls, *args, **kwargs) -> "HK | SMT_KRG":
        """Create a new instance of the HK surrogate model.

        If the number of fidelities is 1, a KRG instance is returned instead, since the
        HK would be an KRG model.

        Returns
        -------
        HK | SMT_KRG
            An instance of the HK surrogate model or KRG if num_fidelities is 1

        Raises
        ------
        ValueError
            If no config is passed.
        """
        config: Config | None = None
        if "config" in kwargs:
            config = kwargs.get("config", None)
        else:
            for arg in args:
                if isinstance(arg, Config):
                    config = arg
                    break
        if config is not None:
            if config.num_fidelities == 1:
                # return a KRG instance
                from .KRG import KRG

                krg_settings = cls._get_krg_settings(config)
                config.update_used_algorithms(
                    {**config.used_algorithms, "surrogate": "KRG"}
                )
                config.update_settings({"surrogate": krg_settings})
                return KRG(*args, **kwargs)
            return super().__new__(cls)

        msg = "Config object must be provided to determine number of fidelities."
        (kwargs.get("logger") or get_logger()).warning(msg)
        raise ValueError(msg)

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the HK surrogate model."""
        super().__init__(*args, **kwargs)

    def _initialise_model(self, *args, **kwargs) -> dict:
        """Initialise the HK surrogate model.

        Returns
        -------
        dict
            An empty dictionary to hold the trained model parameters.
        """
        # Set the LF model to a Kriging model
        model = {fid: {} for fid in range(self.config.num_fidelities)}
        krg_instance = SMT_KRG(**self._get_krg_settings(self.config))
        model[self.config.num_fidelities - 1] = krg_instance

        self._trained = False
        return model

    def train(self) -> None:
        r"""Train the HK surrogate model using the provided training data.

        The lowest fidelity model is trained first, followed by higher fidelity models.
        The training process involves optimising hyperparameters using differential
        evolution and calculating necessary matrices for predictions.

        The hyperparameters $\boldsymbol{\theta}$ maximise the log-likelihood (see
        `_log_likelihood_hk`), searched in $\ln \boldsymbol{\theta}$ within the
        `theta_bounds` setting. The matrices are computed with the Cholesky
        decomposition $\mathbf{R} = \mathbf{C} \mathbf{C}^\top$ and the QR
        decomposition $\mathbf{C}^{-1} \mathbf{F} = \mathbf{Q} \mathbf{G}$, so that
        $\beta_0 = \mathbf{G}^{-1} \mathbf{Q}^\top \mathbf{C}^{-1} \mathbf{y}$, and
        $\boldsymbol{\gamma} = \mathbf{R}^{-1} (\mathbf{y} - \beta_0 \mathbf{F})$ is
        stored for the mean prediction.

        Raises
        ------
        ValueError
            If a fidelity level has fewer than 2 samples.
        LinAlgError
            If the regression matrix is ill-conditioned at the optimal hyperparameters.
        """
        # Check if there are enough samples for each fidelity
        fidelities_with_insufficient_samples = [
            fid
            for fid in range(self.config.num_fidelities)
            if len(self.samples[fid]) < 2
        ]
        if fidelities_with_insufficient_samples:
            msg = (
                f"Not enough samples to train HK surrogate at fidelities: "
                f"{fidelities_with_insufficient_samples}. "
                f"At least 2 samples are required per fidelity."
            )
            self.config.logger.error(msg)
            raise ValueError(msg)

        for fidelity in reversed(range(self.config.num_fidelities)):
            x_samples = np.array([p["coord"] for p in self.samples[fidelity]])
            x_samples = self._correct_x_shape(x_samples)
            y_samples = np.array([p["value"] for p in self.samples[fidelity]]).reshape(
                -1, 1
            )

            if fidelity == self.config.num_fidelities - 1:
                # Train the base KRG model
                krg_model = self._model[fidelity]

                krg_model.set_training_values(x_samples, y_samples)
                krg_model.train()
                self._model[fidelity] = krg_model
                continue

            # Train the HK model for the current fidelity
            # 1. Calculate F, y_samples and n
            n = len(x_samples)
            f = self._predict_mean(x_samples, fidelity=fidelity + 1)
            self._model[fidelity] = {
                "y_samples": y_samples,
                "f": f,
                "x": x_samples,
            }

            # 2. Optimise the negative log-likelihood function for theta using
            # differential evolution
            theta_bounds = self.config.get("surrogate.theta_bounds")

            # Convert to log-space bounds for optimisation
            bound_tuple = (np.log(theta_bounds[0]), np.log(theta_bounds[1]))
            bounds = [
                bound_tuple
                for _ in range(np.array(self.samples[fidelity][0]["coord"]).shape[0])
            ]
            dynamic_popsize = max(15, int(40 / self.config.dimension))

            # define a wrapper for the de optimisation
            def _de_wrapper(
                theta_log: float, n_in: int = n, fidelity_in: int = fidelity
            ) -> float:
                """A wrapper function for the objective of DE optimisation.

                Parameters
                ----------
                theta_log
                    The natural logarithm of the hyperparameters theta.
                n_in
                    The number of samples at this fidelity.
                fidelity_in
                    The fidelity level of the model being trained.

                Returns
                -------
                float
                    The negative log-likelihood, or 1e12 for invalid theta values.
                """
                theta = np.exp(theta_log)  # Convert back to original space
                log_likelihood = self._log_likelihood_hk(theta, n_in, fidelity_in)

                if log_likelihood is None or np.isnan(log_likelihood):
                    return (
                        1e12  # Return a large number to penalise invalid theta values
                    )
                return -1 * log_likelihood

            # Start a multi-start to find the optimal theta
            best_neg_ll = np.inf
            n_starts = self.config.get("surrogate.n_theta_optim_starts")
            theta_opt = None
            base_seed = self.config.get("random_seed") + 1
            for i in range(n_starts):
                result = differential_evolution(
                    _de_wrapper,
                    bounds,
                    args=(n, fidelity),
                    strategy="rand1bin",
                    popsize=dynamic_popsize,
                    init="latinhypercube",
                    recombination=0.9,
                    mutation=(0.5, 1.2),
                    tol=1e-8,
                    polish=True,
                    seed=base_seed + i,
                )

                if result.fun < best_neg_ll:
                    best_neg_ll = result.fun
                    theta_opt = np.exp(result.x)  # Convert back to original space
                    self.config.logger.debug(
                        f"Start {i}: New best Neg-LL found: {best_neg_ll}"
                    )

            # 3. Compute the trained model parameters via Cholesky + QR
            try:
                corr_matrix = self._calculate_corr_matrix(x_samples, theta=theta_opt)
                chol = cholesky(corr_matrix, lower=True)
                ft = solve_triangular(chol, f, lower=True)
                q, g = qr(ft, mode="economic")
                yt = solve_triangular(chol, y_samples, lower=True)

                sv = np.linalg.svd(g, compute_uv=False)
                rcond_g = sv.min() / sv.max() if sv.max() > 0 else 0.0
                if rcond_g < 1e-10:
                    raise LinAlgError(
                        f"Ft is ill-conditioned (rcond={rcond_g:.2e}) at "
                        f"theta_opt={theta_opt}."
                    )

                # beta_0 = (F^T R^-1 F)^-1 F^T R^-1 y solved as G^-1 Q^T Yt
                beta_0_vec = solve_triangular(g, q.T @ yt, lower=False)
                beta_0 = beta_0_vec.item() if beta_0_vec.size == 1 else beta_0_vec

                # sigma^2 = ||C^-1 (y - beta_0 F)||^2 / n
                rho = yt - ft @ np.atleast_2d(beta_0_vec)
                sigma_sq = np.array([[(rho.T @ rho).item() / n]])

                # Precompute gamma = R^-1 (y - beta_0 F) for cheap mean prediction
                residual = y_samples - beta_0 * f
                gamma = solve_triangular(
                    chol.T,
                    solve_triangular(chol, residual, lower=True),
                    lower=False,
                )
            except (LinAlgError, ValueError) as e:
                self.config.logger.warning(
                    f"Unstable matrix at fidelity {fidelity} during training "
                    f"({e}); falling back to a degenerate HK fit at this level."
                )

                chol = np.eye(n)
                ft = f.copy()
                q, g = (
                    qr(ft, mode="economic")
                    if np.any(f)
                    else (
                        np.eye(n, 1),
                        np.array([[1.0]]),
                    )
                )
                beta_0 = 1.0
                sigma_sq = np.array([[1.0]])
                gamma = y_samples - beta_0 * f

            # 4. Store the trained model parameters
            self._model[fidelity].update(
                {
                    "beta_0": beta_0,
                    "chol": chol,
                    "ft": ft,
                    "q": q,
                    "g": g,
                    "gamma": gamma,
                    "theta": theta_opt,
                    "sigma_sq": sigma_sq,
                }
            )

        # Set the trained flag to True
        self._trained = True
        num_samples = [
            (fid, len(self.samples[fid])) for fid in range(self.config.num_fidelities)
        ]
        samples_summary = "; ".join(
            f"fidelity {fid}: {count} sample{'s' if count != 1 else ''}"
            for fid, count in num_samples
        )
        total = sum(count for _, count in num_samples)
        self.config.logger.debug(
            f"HK surrogate model training completed; total {total} "
            f"sample{'s' if total != 1 else ''}; {samples_summary}"
        )

    def predict_values(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict the mean at a given fidelity.

        Parameters
        ----------
        x
            The input points for which the mean is predicted.
        fidelity
            The fidelity level for which the mean is predicted.

        Returns
        -------
        np.ndarray
            The predicted mean at the given fidelity.
        """
        if not self._trained:
            self.train()
        return self._predict_mean(x, fidelity=fidelity)

    def _predict_mean(self, x: np.ndarray, fidelity: int = 0) -> np.ndarray:
        r"""Calculate the mean prediction at a given fidelity.

        The lowest fidelity is predicted by its SMT KRG model; every higher fidelity
        recursively as

        $$
        \hat{y}_l(x) = \beta_0 \, \hat{y}_{l+1}(x)
            + \mathbf{r}(x)^\top \boldsymbol{\gamma},
        \qquad \boldsymbol{\gamma} = \mathbf{R}^{-1} (\mathbf{y} - \beta_0 \mathbf{F})
        $$

        Parameters
        ----------
        x
            The input points for which the mean is predicted.
        fidelity
            The fidelity level for which the mean is predicted.

        Returns
        -------
        np.ndarray
            The predicted mean at the given fidelity.

        Raises
        ------
        ValueError
            If the fidelity level is out of range.
        """
        if fidelity < 0 or fidelity >= self.config.num_fidelities:
            msg = f"Fidelity {fidelity} is out of bounds for the HK model."
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Check x shape
        x = self._correct_x_shape(x)

        if fidelity == self.config.num_fidelities - 1:
            return self._model[fidelity].predict_values(x)

        beta_0 = self._model[fidelity]["beta_0"]
        gamma = self._model[fidelity]["gamma"]
        theta = self._model[fidelity]["theta"]

        corr_vector = self._calculate_corr_vector(x, fidelity=fidelity, theta=theta)
        lower_fidelity_predictions = self._predict_mean(x, fidelity=fidelity + 1)

        # ŷ(x) = beta_0 * ŷ_lf(x) + r(x)^T @ gamma
        # with gamma = R^-1 (y - beta_0 F) precomputed during training.
        term1 = beta_0 * lower_fidelity_predictions
        term2 = corr_vector.T @ gamma
        return term1 + term2

    def predict_variance(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict the variance at a given fidelity.

        Parameters
        ----------
        x
            The input points for which the variance is predicted.
        fidelity
            The fidelity level for which the variance is predicted.

        Returns
        -------
        np.ndarray
            The predicted variance at the given fidelity.
        """
        return self.predict_variance_fidelity(x, *args, fidelity=fidelity, **kwargs)

    def predict_variance_fidelity(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict the variance at a given fidelity.

        Parameters
        ----------
        x
            The input points for which the variance is predicted.
        fidelity
            The fidelity level for which the variance is predicted.

        Returns
        -------
        np.ndarray
            The predicted variance at the given fidelity.
        """
        if not self._trained:
            self.train()
        return self._predict_variance(x, fidelity=fidelity)

    def _predict_variance(self, x: np.ndarray, fidelity: int = 0) -> np.ndarray:
        r"""Calculate the variance prediction at a given fidelity.

        The lowest fidelity is predicted by its SMT KRG model; every higher fidelity
        as (Han and Görtz [1] in the class docstring)

        $$
        \hat{s}_l^2(x) = \hat{\sigma}^2 \left[ 1 - \mathbf{r}^\top \mathbf{R}^{-1}
            \mathbf{r} + \mathbf{a}^\top \left(\mathbf{F}^\top \mathbf{R}^{-1}
            \mathbf{F}\right)^{-1} \mathbf{a} \right],
        \qquad \mathbf{a} = \mathbf{F}^\top \mathbf{R}^{-1} \mathbf{r}
            - \hat{y}_{l+1}(x)
        $$

        with $\mathbf{r} = \mathbf{r}(x)$. Small negative values from rounding errors
        are set to 0.

        Parameters
        ----------
        x
            The input points for which the variance is predicted.
        fidelity
            The fidelity level for which the variance is predicted.

        Returns
        -------
        np.ndarray
            The predicted variance at the given fidelity.

        Raises
        ------
        ValueError
            If the fidelity level is out of range.
        """
        if fidelity < 0 or fidelity >= self.config.num_fidelities:
            msg = f"Fidelity {fidelity} is out of bounds for the HK model."
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Check x shape
        x = self._correct_x_shape(x)

        if fidelity == self.config.num_fidelities - 1:
            return self._model[fidelity].predict_variances(x)

        sigma_sq = self._model[fidelity]["sigma_sq"]
        chol = self._model[fidelity]["chol"]
        ft = self._model[fidelity]["ft"]
        g = self._model[fidelity]["g"]
        theta = self._model[fidelity]["theta"]

        corr_vector = self._calculate_corr_vector(x, fidelity, theta=theta)
        predictions_lf = self._predict_mean(x, fidelity=fidelity + 1)

        # rt = C^-1 r, shape (n_train, n_query).
        # Then diag(r^T R^-1 r) = (rt**2).sum(axis=0).
        rt = solve_triangular(chol, corr_vector, lower=True)
        term_rRr = np.sum(rt**2, axis=0).reshape(-1, 1)

        # The β-correction term:
        #   (ŷ_lf(x) - r^T R^-1 F)^T (F^T R^-1 F)^-1 (ŷ_lf(x) - r^T R^-1 F)
        # is computed as ||u||^2 with
        #   u = G^-T (Ft^T rt - ŷ_lf(x)^T)
        ft_T_rt = ft.T @ rt
        a = ft_T_rt - predictions_lf.T
        u = solve_triangular(g.T, a, lower=True)
        term_F = np.sum(u**2, axis=0).reshape(-1, 1)

        variance = sigma_sq * (1.0 - term_rRr + term_F)

        # Variance might be slightly negative due to floating-point cancellation;
        # force to zero.
        variance[variance < 0.0] = 0.0
        return variance

    def _log_likelihood_hk(
        self, theta: np.ndarray, n: int, fidelity: int
    ) -> float | None:
        r"""Calculate the log-likelihood for the HK model at a given fidelity.

        The concentrated log-likelihood, without constant terms and multiplied by 2:

        $$
        \ln L(\boldsymbol{\theta}) = -n \ln \hat{\sigma}^2(\boldsymbol{\theta})
            - \ln \lvert \mathbf{R}(\boldsymbol{\theta}) \rvert
        $$

        with $\ln \lvert \mathbf{R} \rvert = 2 \sum_i \ln C_{ii}$ from the Cholesky
        factor $\mathbf{C}$.

        Parameters
        ----------
        theta
            The hyperparameter theta for the correlation function.
            With shape Mx1, where M is the number of input dimensions.
        n
            The number of training points at fidelity fid.
        fidelity
            The fidelity level for which the log-likelihood is calculated.

        Returns
        -------
        float | None
            The log-likelihood value for the HK model at the given fidelity. None if
            calculation fails due to numerical issues.
        """
        y_samples = self._model[fidelity]["y_samples"]
        x = self._model[fidelity]["x"]
        f = self._model[fidelity]["f"]

        if y_samples.ndim == 1:
            y_samples = y_samples.reshape(-1, 1)
        if f.ndim == 1:
            f = f.reshape(-1, 1)

        try:
            corr_matrix = self._calculate_corr_matrix(x, theta=theta)

            chol = cholesky(corr_matrix, lower=True)
            log_det_r = 2 * np.sum(np.log(np.diag(chol)))

            ft = solve_triangular(chol, f, lower=True)
            q, g = qr(ft, mode="economic")
            yt = solve_triangular(chol, y_samples, lower=True)

            # Reject theta values that drive Ft towards singular
            sv = np.linalg.svd(g, compute_uv=False)
            if sv.max() <= 0 or sv.min() / sv.max() < 1e-10:
                return None

            # beta_0 = G^-1 Q^T Yt  (equivalent to (F^T R^-1 F)^-1 F^T R^-1 y)
            beta_0_vec = solve_triangular(g, q.T @ yt, lower=False)
            # sigma^2 = ||Yt - ft beta_0||^2 / n  (equivalent to the standard
            # (y - beta_0 F)^T R^-1 (y - beta_0 F) / n).
            rho = yt - ft @ np.atleast_2d(beta_0_vec)
            sigma_sq_val = (rho.T @ rho).item() / n

            if sigma_sq_val <= 0:
                return None

            log_likelihood = -1 * n * np.log(sigma_sq_val) - log_det_r
            return log_likelihood

        except (LinAlgError, ValueError):
            self.config.logger.warning(
                f"Unstable matrix at fidelity {fidelity} during log-likelihood "
                f"calculation."
            )
            return None

    def _calculate_corr_vector(
        self, x_in: np.ndarray, fidelity: int, theta: np.ndarray = None
    ) -> np.ndarray:
        """Calculate the correlation vector between input points and training points.

        Parameters
        ----------
        x_in
            The input points.
        fidelity
            The fidelity level for which the correlation vector is calculated.
        theta
            The hyperparameter theta for the correlation function.

        Returns
        -------
        np.ndarray
            The correlation vector between input points and training points.
        """
        if theta is None:
            theta = self._model[fidelity]["theta"]

        x_train = self._model[fidelity]["x"]
        return self._vectorised_correlation(x_train, x_in, theta)

    def _calculate_corr_matrix(
        self, x_train: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Calculate the correlation matrix for the training points.

        Parameters
        ----------
        x_train
            The training points.
        theta
            The hyperparameter theta for the correlation function.

        Returns
        -------
        np.ndarray
            The correlation matrix for the training points.
        """
        return self._vectorised_correlation(x_train, x_train, theta)

    def _vectorised_correlation(
        self, x1: np.ndarray, x2: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        r"""Calculate the correlation matrix or vector between two sets of points.

        The correlation between two points $x$ and $x'$ with $d$ dimensions, set by the
        `corr` setting. For `squar_exp` ($p = 2$) and `pow_exp` ($p$ = `pow_exp_power`):

        $$
        R(x, x') = \exp\left(-\sum_{k=1}^{d} \theta_k \lvert x_k - x'_k \rvert^p\right)
        $$

        For `cubic_spline`, with $\xi_k = \theta_k \lvert x_k - x'_k \rvert$:

        $$
        R(x, x') = \prod_{k=1}^{d} \zeta(\xi_k), \qquad
        \zeta(\xi) = \begin{cases}
            1 - 15 \xi^2 + 30 \xi^3 & \text{for } 0 \le \xi \le 0.2 \\
            1.25 (1 - \xi)^3 & \text{for } 0.2 < \xi < 1 \\
            0 & \text{for } \xi \ge 1
        \end{cases}
        $$

        For the correlation matrix of a set of points with itself, the `nugget` is
        added to the diagonal.

        Parameters
        ----------
        x1
            The first set of points.
        x2
            The second set of points.
        theta
            The hyperparameter theta for the correlation function.

        Returns
        -------
        np.ndarray
            The correlation matrix or vector between the two sets of points.

        Raises
        ------
        ValueError
            If the correlation type is not supported.
        """
        corr_type = self.config.get("surrogate.corr")

        # reshape x1 or x2 if one of them is 1D
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)

        if corr_type == "cubic_spline":
            diff = np.abs(x1[:, None, :] - x2[None, :, :])
            xi = theta * diff

            r = np.zeros_like(xi)

            mask1 = (xi >= 0.0) & (xi <= 0.2)
            mask2 = (xi > 0.2) & (xi < 1.0)

            r[mask1] = 1.0 - 15.0 * xi[mask1] ** 2 + 30.0 * xi[mask1] ** 3
            r[mask2] = 1.25 * (1.0 - xi[mask2]) ** 3

            r = np.prod(r, axis=2)

        elif corr_type == "pow_exp" or corr_type == "squar_exp":
            if corr_type == "pow_exp":
                p = self.config.get("surrogate.pow_exp_power")
            else:
                p = 2

            diff = np.abs(x1[:, None, :] - x2[None, :, :])
            weighted_diff = theta * diff**p

            r = np.exp(-np.sum(weighted_diff, axis=2))

        else:
            msg = f"Unsupported correlation type: {corr_type}"
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Add nugget
        if x1 is x2 or (x1.shape == x2.shape and np.allclose(x1, x2)):
            nugget = self.config.get(
                "surrogate.nugget", SMT_DEFAULT_NUGGET, strict=False
            )
            if nugget > 0.0:
                r += nugget * np.eye(r.shape[0])

        return r

    @staticmethod
    def _get_krg_settings(config: Config) -> dict:
        """Extract the settings relevant for the KRG surrogate model.

        Parameters
        ----------
        config
            Config object containing the overall settings.

        Returns
        -------
        dict
            Dictionary with settings relevant for the KRG surrogate model.
        """
        krg_settings = config.settings.get("surrogate", {}).copy()

        # Correct the correlation type
        if krg_settings.get("corr", None) == "cubic_spline":
            krg_settings["corr"] = "squar_exp"
            config.logger.info(
                "Note that the cubic_spline correlation is not supported in the base "
                "KRG model. Using squar_exp instead for the (base LF) KRG model."
            )

        # Check if valid settings for SMT_KRG
        for key in config.settings.get("surrogate", {}):
            if not SMT_KRG().options.is_declared(key):
                config.logger.debug(
                    f"Removing unsupported setting '{key}' for KRG surrogate."
                )
                krg_settings.pop(key)
        return krg_settings
