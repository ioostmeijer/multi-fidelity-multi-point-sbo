import logging
from typing import Generator

import pytest
from logger_helper import _make_logger

from sbo.utils import Base, Config


@pytest.fixture(autouse=True)
def basic_config() -> Config:
    """Fixture to provide a basic Config instance for testing."""
    return Config(
        used_algorithms={},
        lb=[0.0],
        ub=[5.0],
        functions=[lambda x: x**1, lambda x: x**2, lambda x: x**3, lambda x: x**4],
        settings=None,
        logger=_make_logger(),
    )


@pytest.mark.unit
def test_init(basic_config: Config) -> None:
    """Test that Base initialises correctly."""
    base = Base(config=basic_config)
    assert base.config == basic_config


@pytest.mark.unit
def test_update_number_of_points(basic_config: Config, caplog: Generator) -> None:
    """Test the update_number_of_points method of Base."""
    base = Base(config=basic_config)
    basic_config.logger.propagate = True

    # Test updating to a valid number of points
    base.update_number_of_points(10)
    assert base.config["number_new_points"] == 10

    # Test updating to an invalid number of points
    caplog.set_level(logging.WARNING)
    base.update_number_of_points(0)

    # Ensure a warning was logged
    assert any(
        "Number of new points must be at least 1. Not changing." in rec.getMessage()
        for rec in caplog.records
    )

    # Ensure the number of points was not changed
    assert base.config["number_new_points"] == 10
