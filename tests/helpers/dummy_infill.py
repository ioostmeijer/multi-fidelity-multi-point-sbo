import numpy as np

from sbo.infill import InfillBase
from sbo.surrogate import SurrogateBase


class DummyInfill(InfillBase):
    """A dummy infill class for testing."""

    def __init__(
        self,
        *args,
        new_points: list[list[dict]] | None = None,
        infill_value: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialise the DummyInfill class.

        Parameters
        ----------
        new_points
            New points to be returned by get_new_points method.
        infill_value
            Infill value to be returned by calc_infill_value method.
        """
        super().__init__(*args, **kwargs)

        self._new_points = new_points if new_points else [[{"coord": [0.0], "fid": 0}]]
        self._infill_value = infill_value

    def get_new_points(
        self, surrogate: SurrogateBase, *args, **kwargs
    ) -> tuple[list[list[dict]], float]:
        """Get new points and their infill value for testing.

        Parameters
        ----------
        surrogate
            Surrogate model used for infill calculation.

        Returns
        -------
        A tuple containing the new points and their infill value.
        """
        x = kwargs.get("x", np.array([[0.0]]))
        infill_value = self.calc_infill_value(*args, x=x, surrogate=surrogate, **kwargs)
        if not isinstance(infill_value, np.ndarray):
            infill_value = np.array([infill_value])
        return (self._new_points, min(infill_value))

    def calc_infill_value(
        self, x: np.ndarray, surrogate: SurrogateBase, *args, **kwargs
    ) -> float | np.ndarray:
        """Calculate the infill value for given points.

        Parameters
        ----------
        x
            Points at which to calculate the infill value.
        surrogate
            Surrogate model used for infill calculation.

        Returns
        -------
        The calculated infill value(s).
        """
        if "function" in kwargs:
            function = kwargs["function"]
            return function(x)
        else:
            return self._infill_value
