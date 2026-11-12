import copy
import os
import pathlib
import pickle
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

from ..utils import Base


class SurrogateBase(Base):
    """Base class for surrogate models.

    Attributes
    ----------
    samples
        A dictionary storing the training data.
    """

    # Whether predict(..., fidelity=f) gives the prediction at fidelity f. Surrogates
    # that only model the HF function (or ignore the fidelity argument) keep False.
    supports_fidelity_prediction: bool = False

    def __init__(
        self, *args, initial_samples: list[dict] | None = None, **kwargs
    ) -> None:
        """Initialise the SurrogateBase class.

        Parameters
        ----------
        initial_samples
            A list of dictionaries containing initial training data points.
            ({'coord': np.ndarray, 'fidelity': int, 'value': float})
        """
        super().__init__(*args, **kwargs)

        self._trained: bool = False
        self._optimum: dict = dict.fromkeys(range(self.config.num_fidelities), None)
        self.samples: dict = {fid: [] for fid in range(self.config.num_fidelities)}
        if initial_samples is not None:
            self.set_training_data(initial_samples)

        self._model = self._initialise_model(*args, **kwargs)

    def _initialise_model(self, *args, **kwargs) -> Any:  # noqa: ANN401
        """Initialise the surrogate model.

        Returns
        -------
        Any
            Initialised surrogate model.

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        msg = (
            "The _initialise_model method must be implemented by subclasses of "
            "SurrogateBase."
        )
        self.config.logger.error(msg)
        raise NotImplementedError(msg)

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the output for given input points.

        Parameters
        ----------
        x
            Input points for prediction.

        Returns
        -------
        np.ndarray
            Predicted values at the input points, with shape (n_points, 1).
        """
        # Check x shape
        x = self._correct_x_shape(x)

        results = np.asarray(self.predict_values(x, *args, **kwargs))
        results = self._reshape_results(results, x.shape[0])

        return results

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the output values for given input points.

        Parameters
        ----------
        x
            Input points for prediction.

        Returns
        -------
        np.ndarray
            Predicted values at the input points, with shape (n_points, 1).

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        msg = (
            "The predict_values method must be implemented by subclasses of "
            "SurrogateBase."
        )
        self.config.logger.error(msg)
        raise NotImplementedError(msg)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Make the surrogate callable to predict outputs.

        Parameters
        ----------
        x
            Input points for prediction.

        Returns
        -------
        np.ndarray
            The result of predict: the predicted values with shape (n_points, 1), or for
            surrogates with variance a tuple (values, variances) if requested.
        """
        return self.predict(x, *args, **kwargs)

    def get_optimum(self, fidelity: int = 0) -> dict:
        """Retrieve the current optimum from the surrogate model.

        Parameters
        ----------
        fidelity
            The fidelity level of the optimum (0 = HF).

        Returns
        -------
        dict
            A dictionary containing the optimum information.

        Raises
        ------
        ValueError
            If the fidelity level is out of range.
        """
        # Check if fidelity is valid
        if fidelity < 0 or fidelity >= self.config.num_fidelities:
            msg = f"Fidelity {fidelity} is out of bounds."
            self.config.logger.error(msg)
            raise ValueError(msg)

        # Check if the optimum of the current model has been calculated before
        if self._trained and self._optimum[fidelity] is not None:
            return self._optimum[fidelity]

        # Check if the model is trained
        if not self._trained:
            self.train()

        # Get the constraints
        # scipy_constr = self.config.scipy_constraints(
        #     return_inexpensive=True, return_surrogate=True
        # )

        result = differential_evolution(
            lambda x: self.predict_values(np.array([x]), fidelity=fidelity),
            self.config.bounds,
            # constraints=scipy_constr,
            strategy=self.config.get("surrogate_opt_strategy", "best1bin"),
            maxiter=self.config.get("surrogate_opt_max_iterations", 1000),
            popsize=self.config.get("surrogate_opt_population_size", 15),
            tol=self.config.get("surrogate_opt_tolerance", 0.01),
            mutation=self.config.get("surrogate_opt_mutation", (0.5, 1)),
            recombination=self.config.get("surrogate_opt_recombination", 0.7),
            disp=self.config.get("surrogate_opt_verbose", False),
            polish=self.config.get("surrogate_opt_polish", True),
            init=self.config.get("surrogate_opt_init", "latinhypercube"),
            atol=self.config.get("surrogate_opt_atol", 0),
            seed=self.config.get("random_seed"),
        )
        self._optimum[fidelity] = {"coord": result.x, "value": result.fun}
        return self._optimum[fidelity]

    def train(self) -> None:
        """Train the surrogate model using the provided training data.

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        if self._trained:
            return

        msg = "The train method must be implemented by subclasses of SurrogateBase."
        self.config.logger.error(msg)
        raise NotImplementedError(msg)

    def set_training_data(self, points: list[dict]) -> None:
        """Set the training data for the surrogate model.

        This replaces the current samples. If you want to add samples, use
        add_training_data instead.

        Parameters
        ----------
        points
            A list of dictionaries containing the training data points.
            ({'coord': np.ndarray, 'fidelity': int, 'value': float})
        """
        self.samples = {fid: [] for fid in range(self.config.num_fidelities)}

        self.add_training_data(points)
        return

    def add_training_data(self, points: list[dict] | dict) -> None:
        """Add training data to the surrogate model.

        This adds the provided points to the current samples. If you want to
        replace the current samples, use set_training_data instead.

        Parameters
        ----------
        points
            A list of dictionaries containing the training data points.
            ({'coord': np.ndarray, 'fid': int, 'value': float})
            or a dictionary containing the training data points, with fidelity levels as
            keys.
            ({fid: list of dict, ...})
        """
        if isinstance(points, dict):
            points_list = []
            for fid_points in points.values():
                points_list.extend(fid_points)
        else:
            points_list = points

        for point in points_list:
            # Copy the point so the caller's data is not changed
            point = dict(point)
            fidelity = point.get("fid")

            if isinstance(point["value"], list) or isinstance(
                point["value"], np.ndarray
            ):
                point["value"] = float(np.ravel(point["value"])[0])

            self.samples.setdefault(fidelity, [])

            try:
                if any(
                    np.array_equal(point["coord"], existing_point["coord"])
                    for existing_point in self.samples[fidelity]
                ):
                    self.config.logger.info(
                        f"Point {point} already exists in samples for fidelity "
                        f"{fidelity}, not adding."
                    )
                    continue
            except ValueError as e:
                msg = (
                    f"Error checking if point {point} exists in samples for "
                    f"fidelity {fidelity}, skipping point (error: {e})."
                )
                self.config.logger.error(msg)
                continue
            self.samples[fidelity].append(point)

        self._trained = False
        self._optimum = dict.fromkeys(range(self.config.num_fidelities), None)

    def save_model(self) -> dict:  # noqa: ANN401
        """Serialise the surrogate model.

        Returns
        -------
        dict
            The model state: {'model': the underlying model, 'trained': bool, 'optimum':
            the cached optimum per fidelity, 'samples': the training samples per
            fidelity}. Can be restored with load_model.
        """
        return {
            "model": copy.deepcopy(self._model),
            "trained": self._trained,
            "optimum": copy.deepcopy(self._optimum),
            "samples": copy.deepcopy(self.samples),
        }

    def load_model(self, model_data: dict) -> None:  # noqa: ANN401
        """Load the surrogate model from serialised data.

        Parameters
        ----------
        model_data
            dictionary containing the model data.
        """
        self._model = copy.deepcopy(model_data.get("model"))
        self._trained = model_data.get("trained", False)
        self._optimum = copy.deepcopy(model_data.get("optimum"))
        self.samples = copy.deepcopy(model_data.get("samples", {}))
        return

    def save_model_to_file(self, file_path: str | pathlib.Path) -> None:
        """Save the surrogate model to a file.

        Parameters
        ----------
        file_path
            The path to the file where the model should be saved.
        """
        model_data = self.save_model()

        path = pathlib.Path(file_path)

        if path.name == "":
            self.config.logger.error(
                "No file name provided in file path for saving model, "
                "saving to surrogate_model.pkl."
            )
            path = path / "surrogate_model.pkl"

        if path.suffix.lower() != ".pkl":
            if path.suffix != "":
                self.config.logger.warning(
                    f"File extension not .pkl. Overwriting extension in file name "
                    f"{path.name}."
                )
            else:
                self.config.logger.warning(
                    f"No file extension provided. Adding .pkl to file name, new file "
                    f"name: {path}."
                )
            path = path.with_suffix(".pkl")

        parent = path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model_from_file(self, file_path: str | pathlib.Path) -> None:
        """Load the surrogate model from a file.

        Parameters
        ----------
        file_path
            The path to the file from which the model should be loaded.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.

        Warnings
        --------
        The model is loaded with `pickle`, which can run arbitrary code while loading.
        Only load model files that you created yourself or otherwise trust. Version
        2.0.0 replaces pickle with a data-only format (issue #54).
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            self.config.logger.error(f"Model file {file_path} does not exist.")
            raise FileNotFoundError(f"Model file {file_path} does not exist.")

        # Trusted files only: see the Warnings section of the docstring.
        with open(file_path, "rb") as f:
            model_data = pickle.load(f)  # noqa: S301
        self.load_model(model_data)
        return

    def _correct_x_shape(self, x: np.ndarray) -> np.ndarray:
        """Ensure that x is 2D with shape (n_points, n_dimensions).

        A 1D array is read as a single point when n_dimensions > 1, and as a
        column of points when n_dimensions == 1.

        Parameters
        ----------
        x
            The input points.

        Returns
        -------
        np.ndarray
            The input points with shape (n_points, n_dimensions).
        """
        x = np.asarray(x)
        n_dim = self.config.n_dimensions
        if x.ndim != 2 or x.shape[1] != n_dim:
            x = x.reshape(-1, n_dim)
        return x

    @staticmethod
    def _reshape_results(results: np.ndarray, n_points: int) -> np.ndarray:
        """Reshape prediction results to ensure 2D output.

        Parameters
        ----------
        results
            The prediction results (scalar, 1D or 2D).
        n_points
            The number of points that were predicted.

        Returns
        -------
        np.ndarray
            The results with shape (n_points, n_outputs).
        """
        if results.ndim == 0:
            results = results.reshape(1, 1)
        elif results.ndim == 1:
            if results.shape[0] == n_points:
                results = results.reshape(n_points, 1)
            else:
                results = results.reshape(1, -1)
        else:
            results = results.reshape(results.shape[0], -1)

        return results


class SurrogateBaseWithVariance(SurrogateBase):
    """Base class for surrogate models that provide variance estimates."""

    def predict(
        self, x: np.ndarray, *args, return_variance: bool = True, **kwargs
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict the output and variance for given input points.

        Parameters
        ----------
        x
            Input points for prediction.
        return_variance
            Whether to return the variance of predictions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | np.ndarray
            Predicted output values and their variance if requested.
        """
        x = self._correct_x_shape(x)

        if not self._trained:
            self.train()

        results_val = super().predict(x, *args, **kwargs)

        if not return_variance:
            return results_val

        results_var = self._reshape_results(
            np.asarray(self.predict_variance(x, *args, **kwargs)), x.shape[0]
        )
        return (results_val, results_var)

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict only the output values for given input points.

        Parameters
        ----------
        x
            Input points for prediction.

        Returns
        -------
        np.ndarray
            Predicted values at the input points, with shape (n_points, 1).

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        self.config.logger.error(
            "The predict_values method must be implemented by subclasses of "
            "SurrogateBaseWithVariance."
        )
        raise NotImplementedError(
            "The predict_values method must be implemented by subclasses."
        )

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict only the variance for given input points.

        Parameters
        ----------
        x
            Input points for prediction.

        Returns
        -------
        np.ndarray
            Predicted variances at the input points, with shape (n_points, 1).

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        self.config.logger.error(
            "The predict_variance method must be implemented by subclasses of "
            "SurrogateBaseWithVariance."
        )
        raise NotImplementedError(
            "The predict_variance method must be implemented by subclasses."
        )


class SurrogateBaseWithVariancePerFidelity(SurrogateBaseWithVariance):
    """Base class for surrogate models, providing var estimates per fidelity level."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the SurrogateBaseWithVariancePerFidelity class."""
        super().__init__(*args, **kwargs)

    def predict_variance(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Predict the variance at given input points.

        Parameters
        ----------
        x
            Input points for prediction.

        Returns
        -------
        np.ndarray
            Predicted variance values.
        """
        x = self._correct_x_shape(x)

        if "fidelity" in kwargs:
            fidelity = kwargs.pop("fidelity")
            return self.predict_variance_fidelity(x, fidelity=fidelity)

        return self.predict_variance_fidelity(x, fidelity=0)

    def predict_variance_fidelity(
        self, x: np.ndarray, *args, fidelity: int = 0, **kwargs
    ) -> np.ndarray:
        """Predict the variance at given input points for a specific fidelity level.

        Parameters
        ----------
        x
            Input points for prediction.
        fidelity
            Fidelity level for which to predict variance.

        Returns
        -------
        np.ndarray
            Predicted variances at the input points, with shape (n_points, 1) at the
            given fidelity.

        Raises
        ------
        NotImplementedError
            Always: this method must be implemented by a subclass.
        """
        self.config.logger.error(
            "The predict_variance_fidelity method must be implemented."
        )
        raise NotImplementedError(
            "The predict_variance_fidelity method must be implemented."
        )
