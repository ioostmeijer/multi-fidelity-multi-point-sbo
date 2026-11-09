import numpy as np
from smt.applications.mfk import MFK as SMT_MFK

from .Base import SurrogateBaseWithVariance


class MFK(SurrogateBaseWithVariance):
    r"""Multi-Fidelity Kriging (MFK) surrogate model.

    This class wraps around the SMT MFK model, providing functionality to
    initialise, train, and predict using the MFK model. It inherits from the
    SurrogateBaseWithVariance class and implements the necessary methods for
    training and prediction.

    SMT's MFK is a recursive co-Kriging model: each fidelity $l$ is the prediction of
    the next lower fidelity, scaled, plus a Kriging model of the difference:

    $$
    \hat{y}_l(x) = \rho_l(x) \, \hat{y}_{l+1}(x) + \hat{\delta}_l(x)
    $$

    with the scaling $\rho_l(x)$ a regression (the `rho_regr` setting).

    The fidelity argument of predict_values and predict_variance selects the level:
    fidelity 0 is the HF level, fidelity n_fidelities - 1 the lowest. SMT orders its
    levels the other way round (level 1 is the lowest, level n the HF level).

    Note this class has not been verified.
    """

    supports_fidelity_prediction = True

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the MFK surrogate model."""
        super().__init__(*args, **kwargs)

        if not self.config.get("nested", True):
            self.config.logger.error(
                "MFK model does not support non nested samples, setting "
                "to nested samples."
            )
            self.config.set_setting("nested", True)

    def _initialise_model(self, *args, **kwargs) -> SMT_MFK:
        """Initialise the MFK model with the given parameters.

        Returns
        -------
        SMT_MFK
            An instance of the MFK model.
        """
        params = {**self.config.settings["surrogate"]}

        if params["theta0"] is None:
            params["theta0"] = [0.01 for _ in range(len(self.samples[0][0]["coord"]))]
        if params["theta_bounds"] is None:
            params["theta_bounds"] = [1e-6, 20.0]
        if params["noise0"] is None:
            params["noise0"] = [0.0 for _ in range(self.config.num_fidelities)]
        if params["noise_bounds"] is None:
            params["noise_bounds"] = [2.220446049250313e-14, 10000000000.0]

        # Add the seed to the parameters if specified
        params["seed"] = self.config.get("random_seed", None)

        model = SMT_MFK(**params)

        self._trained = False
        return model

    def train(self) -> None:
        """Train the MFK model with the current training data."""
        if not self._trained:
            if self.samples:
                # Add samples to the model
                for fidelity in range(self.config.num_fidelities):
                    samples = self.samples.get(fidelity, [])
                    x = np.vstack([sample["coord"] for sample in samples])
                    y = np.asarray([sample["value"] for sample in samples])

                    if fidelity == 0:
                        self._model.set_training_values(x, y)
                    else:
                        self._model.set_training_values(
                            x, y, name=self.config.num_fidelities - fidelity - 1
                        )
                self._model.train()
                self._trained = True
        return

    def _check_fidelity(self, fidelity: int) -> None:
        """Raise a ValueError if the fidelity level does not exist.

        Parameters
        ----------
        fidelity
            The fidelity level (0 = HF).

        Raises
        ------
        ValueError
            If the fidelity level is out of range.
        """
        if not 0 <= fidelity < self.config.num_fidelities:
            msg = (
                f"Fidelity {fidelity} is out of range for "
                f"{self.config.num_fidelities} fidelities."
            )
            self.config.logger.error(msg)
            raise ValueError(msg)

    def predict_values(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict values using the MFK model.

        Parameters
        ----------
        x
            Input coordinates for prediction.
        fidelity
            The fidelity level to predict (0 = HF).

        Returns
        -------
        np.ndarray
            Predicted values as a numpy array.
        """
        if not self._trained:
            self.train()

        # Check x shape
        x = self._correct_x_shape(x)

        self._check_fidelity(fidelity)
        if fidelity == 0:
            return self._model.predict_values(x)

        # SMT level (1 = lowest fidelity, n = HF) of this fidelity
        smt_level = self.config.num_fidelities - fidelity
        return self._model._predict_intermediate_values(x, smt_level)

    def predict_variance(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict variances using the MFK model.

        Parameters
        ----------
        x
            Input coordinates for prediction.
        fidelity
            The fidelity level to predict (0 = HF).

        Returns
        -------
        np.ndarray
            Predicted variances as a numpy array.
        """
        if not self._trained:
            self.train()

        # Check x shape
        x = self._correct_x_shape(x)

        self._check_fidelity(fidelity)
        if fidelity == 0:
            return self._model.predict_variances(x)

        # Column of this fidelity in SMT's levels (0 = lowest fidelity)
        smt_column = self.config.num_fidelities - fidelity - 1
        variances_all_levels = self._model.predict_variances_all_levels(x)[0]
        return variances_all_levels[:, smt_column].reshape(-1, 1)
