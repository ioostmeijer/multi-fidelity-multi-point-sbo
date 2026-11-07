from .Base import SurrogateBaseWithVariancePerFidelity


class CoKRG(SurrogateBaseWithVariancePerFidelity):
    """CoKriging surrogate model."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the CoKriging surrogate model.

        Raises
        ------
        NotImplementedError
            Always: CoKriging is not implemented yet.
        """
        super().__init__(*args, **kwargs)
        raise NotImplementedError("CoKriging is not yet implemented.")

    def _initialise_model(self, *args, **kwargs) -> None:
        """Initialise the model.

        TODO implement
        """
        pass

    def train(self) -> None:
        """Train the surrogate model.

        TODO implement
        """
        pass
