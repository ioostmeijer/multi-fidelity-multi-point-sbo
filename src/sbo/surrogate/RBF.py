import numpy as np
from smt.surrogate_models import RBF as SMT_RBF

from .Base import SurrogateBase


class RBF(SurrogateBase):
    r"""Radial Basis Function (RBF) surrogate model.

    This class wraps around the SMT RBF model, providing functionality to
    initialise, train, and predict using the RBF model. It inherits from the
    SurrogateBase class and implements the necessary methods for training and
    prediction.

    The prediction is a weighted sum of Gaussian basis functions centred at the $n$
    high-fidelity samples $x_i$, plus an optional polynomial $p(x)$ (the
    `poly_degree` setting):

    $$
    \hat{y}(x) = \sum_{i=1}^{n} w_i
        \exp\left(-\frac{\lVert x - x_i \rVert^2}{d_0^2}\right) + p(x)
    $$

    with $d_0$ the `d0` setting. The weights $w_i$ are fitted to the samples, with
    the `reg` setting as regularisation.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the RBF surrogate model."""
        super().__init__(*args, **kwargs)

        if self.config.num_fidelities > 1:
            msg = (
                "RBF model does not support multiple fidelities, will only be using HF."
            )
            self.config.logger.warning(msg)

    def _initialise_model(self, *args, **kwargs) -> SMT_RBF:
        """Initialise the RBF model with the given parameters.

        Returns
        -------
        SMT_RBF
            An instance of the RBF model.
        """
        params = self.config.settings["surrogate"].copy()

        if params.get("data_dir") is None:
            params.pop("data_dir")

        model = SMT_RBF(**params)

        self._trained = False
        return model

    def train(self) -> None:
        """Train the RBF model with the current training data."""
        if not self._trained:
            if self.samples.get(0, []):
                hf_points = self.samples.get(0, [])
                x = np.vstack([p["coord"] for p in hf_points])
                y = np.asarray([p["value"] for p in hf_points])
                self._model.set_training_values(x, y)
                self._model.train()
                self._trained = True
                self.config.logger.debug(
                    f"RBF model trained with {len(hf_points)} HF samples."
                )
            else:
                self.config.logger.warning(
                    "No samples available to train the RBF model."
                )
        return

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict values using the RBF model.

        Parameters
        ----------
        x
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted values as a numpy array.
        """
        if isinstance(x, list):
            x = np.array(x)
        if not self._trained:
            self.train()

        # Check x shape
        x = self._correct_x_shape(x)

        return self._model.predict_values(x)
