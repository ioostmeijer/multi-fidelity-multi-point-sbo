from .Config import Config


class Base:
    """Base class for the other classes in the framework.

    This class provides common functionality and attributes that can be
    inherited by other classes.

    Attributes
    ----------
    config
        Configuration object containing the problem definition and settings.
    """

    def __init__(self, config: Config, *args, **kwargs) -> None:
        """Initialise the Base class.

        Parameters
        ----------
        config
            The configuration of the optimisation problem.
        """
        self.config = config

        if args:
            self.config.logger.debug(
                f"Additional arguments passed to {self.__class__.__name__}.__init__(): "
                f"{args}"
            )
        if kwargs:
            self.config.logger.debug(
                f"Additional keyword arguments passed to "
                f"{self.__class__.__name__}.__init__(): {kwargs}"
            )

    def update_number_of_points(self, number_of_points: int) -> None:
        """Update the number of points in the configuration.

        Parameters
        ----------
        number_of_points
            The new number of points to set in the configuration.
        """
        if number_of_points < 1:
            self.config.logger.warning(
                "Number of new points must be at least 1. Not changing."
            )
            return

        self.config["number_new_points"] = number_of_points
