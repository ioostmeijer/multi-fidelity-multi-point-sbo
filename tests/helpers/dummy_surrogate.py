import os
import pickle

import numpy as np

from sbo import Config
from sbo.surrogate import (
    SurrogateBase,
    SurrogateBaseWithVariance,
    SurrogateBaseWithVariancePerFidelity,
)


class DummySurrogate(SurrogateBase):
    """A dummy surrogate model for testing purposes."""

    def __init__(
        self,
        config: Config,
        mu: float = 0.0,
        optimum_value: dict | None = None,
        samples: dict | None = None,
    ) -> None:
        """Initialise the DummySurrogate with necessary attributes.

        Parameters
        ----------
        config
            Configuration object containing settings for the surrogate.
        mu
            The mean prediction value for the surrogate.
        optimum_value| None
            The optimum value and coordinates for the surrogate.
        samples| None
            Sampled data points for the surrogate.
        """
        self._mu = mu
        self._optimum = (
            optimum_value
            if optimum_value is not None
            else {"coord": np.array([0.0, 1.0]), "value": [1.0]}
        )
        self.samples = samples if samples is not None else {}
        self._trained = True
        self.config = config
        self._model = "some model representation"

    def get_optimum(self, fidelity: int = 0) -> dict:
        """Get the optimum value and coordinates.

        Parameters
        ----------
        fidelity
            The fidelity level for which to get the optimum (default: 0).
        """
        return self._optimum

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the mean values at point x.

        Parameters
        ----------
        x
            Input points where predictions are to be made.

        Returns
        -------
        Predicted mean values at the input points.
        """
        return np.array([self._mu] * len(x))

    def train(self) -> None:
        """Train the surrogate model (dummy implementation)."""
        self._trained = True

    def save_model_to_file(self, file_path: str) -> None:
        """Save the surrogate model to a file using pickle."""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def load_model_from_file(self, file_path: str) -> None:
        """Load the surrogate model from a file using pickle."""
        with open(file_path, "rb") as f:
            loaded_model = pickle.load(f)  # noqa: S301 (file written by the test)
        self._mu = loaded_model._mu
        self._optimum = loaded_model._optimum
        self.samples = loaded_model.samples
        self._trained = loaded_model._trained

    def add_training_data(self, new_data: list[dict]) -> None:
        """Mock method to add new training data to the surrogate model.

        Mock method to simulate adding new training data to the surrogate model.
        Without overriding the _trained flag, used for testing purposes.

        Parameters
        ----------
        new_data
            A list of dicts, each dict containing:
            - 'coord': A numpy ndarray representing the coordinates of the point.
            - 'fid': An integer representing the fidelity level (0=HF -> n=LF).
        """
        for point in new_data:
            fid = point["fid"]
            if fid not in self.samples:
                self.samples[fid] = []
            self.samples[fid].append(point)
        return


class DummySurrogateWithVariance(DummySurrogate, SurrogateBaseWithVariance):
    """A dummy surrogate model with variance for testing purposes."""

    def __init__(self, config: Config, *args, var: float = 1.0, **kwargs) -> None:
        """Initialise the DummySurrogateWithVariance with necessary attributes.

        Parameters
        ----------
        config
            Configuration object containing settings for the surrogate.
        var
            The variance prediction value for the surrogate.
        """
        self._var = var
        super().__init__(config, *args, **kwargs)

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the mean values at point x.

        Parameters
        ----------
        x
            Input points where predictions are to be made.

        Returns
        -------
        Predicted mean values at the input points.
        """
        return np.array([self._mu] * len(x))

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the variance values at point x.

        Parameters
        ----------
        x
            Input points where variance predictions are to be made.

        Returns
        -------
        Predicted variance values at the input points.
        """
        return np.array([self._var] * len(x))


class DummySurrogateWithVariancePerFidelity(
    DummySurrogateWithVariance, SurrogateBaseWithVariancePerFidelity
):
    """A dummy surrogate model with variance per fidelity for testing purposes."""

    def __init__(
        self, config: Config, *args, var_per_fidelity: dict | None = None, **kwargs
    ) -> None:
        """Initialise an instance with necessary attributes.

        Parameters
        ----------
        config
            Configuration object containing settings for the surrogate.
        var_per_fidelity
            A dictionary mapping fidelity levels to their respective variance values.
        """
        super().__init__(config, *args, **kwargs)
        if var_per_fidelity is None:
            self._var_per_fidelity = {
                level: kwargs.get("var", 1.0)
                for level in range(self.config.num_fidelities)
            }
            self._var_per_fidelity[2] = (
                kwargs.get("var", 1.0) * 100
            )  # Make fidelity 2 less accurate
        else:
            self._var_per_fidelity = var_per_fidelity

    def predict_variance_fidelity(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict the variance at point x for a given fidelity level.

        Parameters
        ----------
        x
            Input points where variance predictions are to be made.
        fidelity
            The fidelity level for which variance predictions are to be made.

        Returns
        -------
        Predicted variance values at the input points for the specified fidelity level.
        """
        return np.array([self._var_per_fidelity[fidelity]] * len(x))

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the variance values at point x.

        Parameters
        ----------
        x
            Input points where variance predictions are to be made.

        Returns
        -------
        Predicted variance values at the input points.
        """
        if "fidelity" in kwargs:
            return self.predict_variance_fidelity(
                x, kwargs["fidelity"], *args, **kwargs
            )
        return np.array([self._var] * len(x))
