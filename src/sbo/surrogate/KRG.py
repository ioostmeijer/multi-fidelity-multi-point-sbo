import numpy as np
from smt.surrogate_models import KRG as SMT_KRG

from .Base import SurrogateBaseWithVariance


class KRG(SurrogateBaseWithVariance):
    r"""Kriging Regression (KRG) surrogate model.

    This class wraps around the SMT KRG model, providing functionality to
    initialise, train, and predict using the KRG model. It inherits from the
    SurrogateBaseWithVariance class and implements the necessary methods for
    training and prediction.

    The prediction is a regression trend plus a correlation term fitted to the
    high-fidelity samples $\mathbf{y}$:

    $$
    \hat{y}(x) = \mathbf{f}(x)^\top \boldsymbol{\beta}
        + \mathbf{r}(x)^\top \mathbf{R}^{-1}
        (\mathbf{y} - \mathbf{F} \boldsymbol{\beta})
    $$

    with $\mathbf{f}(x)$ the regression functions (the `poly` setting),
    $\mathbf{F}$ these functions at the samples, $\boldsymbol{\beta}$ their
    coefficients, $\mathbf{R}$ the correlation matrix of the samples (the `corr`
    setting) and $\mathbf{r}(x)$ the correlations between $x$ and the samples. See
    the SMT documentation for the variance.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the KRG surrogate model."""
        super().__init__(*args, **kwargs)

        if self.config.num_fidelities > 1:
            self.config.logger.warning(
                "KRG model does not support multiple fidelities. "
                "Using only the HF function."
            )

    def _initialise_model(self, *args, **kwargs) -> SMT_KRG:
        """Initialise the KRG model with the given parameters.

        Returns
        -------
        SMT_KRG
            An instance of the KRG model.
        """
        params = {**self.config.settings["surrogate"]}

        if params.get("theta0", None) is None:
            params["theta0"] = [0.01] * self.config.dimension
        if params.get("theta_bounds", None) is None:
            params["theta_bounds"] = [1e-6, 20.0]
        if params.get("noise0", None) is None:
            params["noise0"] = [0.0]
        if params.get("noise_bounds", None) is None:
            params["noise_bounds"] = [np.finfo(float).eps, 1e10]

        # Add the seed to the parameters if specified
        params["seed"] = self.config.get("random_seed", None)

        model = SMT_KRG(**params)

        self._trained = False
        return model

    def train(self) -> None:
        """Train the KRG model with the current training data."""
        if not self._trained:
            hf_points = self.samples.get(0, [])
            if hf_points:
                x = np.vstack([p["coord"] for p in hf_points])
                y = np.asarray([p["value"] for p in hf_points])
                self._model.set_training_values(x, y)
                self._model.train()
                self._trained = True
                self.config.logger.debug(
                    f"KRG model trained with {len(hf_points)} HF samples."
                )
            else:
                self.config.logger.warning(
                    "No samples available to train the KRG model."
                )
        return

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict values using the KRG model.

        Parameters
        ----------
        x
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted values as a numpy array.
        """
        if not self._trained:
            self.train()

        # Check x shape
        x = self._correct_x_shape(x)
        return self._model.predict_values(x)

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict variances using the KRG model.

        Parameters
        ----------
        x
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted variances as a numpy array.
        """
        if not self._trained:
            self.train()

        # Check x shape
        x = self._correct_x_shape(x)
        return self._model.predict_variances(x)
