import numpy as np

from .Base import InfillBase


class Random(InfillBase):
    """A random multi-fidelity infill strategy.

    This infill strategy generates a specified number of new points, divided randomly
    between the lower and upper bounds defined in the configuration. It supports
    multiple fidelity levels.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the Random strategy."""
        super().__init__(*args, **kwargs)

        # Warn if constraints in config
        if any(self.config.equality_constraints().values()) or any(
            self.config.inequality_constraints().values()
        ):
            self.config.logger.warning(
                "Constraints are passed to the optimisation, however the selected "
                "infill strategy does not support constraints, and will thus be "
                "ignored."
            )

    def get_new_points(self, *args, **kwargs) -> tuple[list[list[dict]], float]:
        """Generate new random points within the defined bounds.

        Returns
        -------
        tuple[list[list[dict]], float]
            The new points and the infill value. The new points are a list of
            tasks; each task is a list of points to evaluate together
            ({'coord': np.ndarray, 'fid': int}, fid 0 is HF). The infill value is
            always inf for Random.
        """
        number_new_points = self.config["number_new_points"]

        new_points = self.config.rng.uniform(
            low=self.config.lb,
            high=self.config.ub,
            size=(number_new_points, self.config.dimension),
        )
        if self.config.num_fidelities > 1:
            fidelities = self.config.rng.integers(
                low=0,
                high=self.config.num_fidelities,
                size=(number_new_points,),
                dtype=int,
            )
        else:
            fidelities = np.zeros((number_new_points,), dtype=int)

        # Create list of new points with fidelity levels
        new_points = [
            self._return_nested_samples(
                [{"coord": new_points[i], "fid": fidelities[i]}]
            )
            for i in range(len(new_points))
        ]

        return new_points, float("inf")
