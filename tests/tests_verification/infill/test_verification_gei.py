import math
from typing import Generator

import numpy as np
import pytest
from logger_helper import _make_logger
from mf2 import forrester
from scipy.integrate import quad
from scipy.stats import norm

from sbo import Config
from sbo.infill import EI, GEI
from sbo.surrogate import KRG

G_VALUES = [1, 2, 3, 5]
X_TRAIN = np.array([0.0, 0.375, 0.4, 0.721, 0.832, 0.969]).reshape(-1, 1)
X_TEST = np.linspace(0.0, 1.0, 100).reshape(-1, 1)


@pytest.fixture(scope="module")
def krg_fixture() -> Generator:
    """Fixture to provide a trained KRG model and GEI."""
    # Ensure new Config
    Config.destroy_instance(destroy_logger=True)

    # Get the training samples
    y_train = forrester.functions[0](X_TRAIN)

    # Get custom GEI and KRG
    config = Config(
        used_algorithms={"surrogate": "KRG", "infill": "GEI"},
        lb=forrester.l_bound,
        ub=forrester.u_bound,
        functions=[forrester.functions[0]],
        # Module fixture, set up before the test runs in its temporary directory:
        # use the test logger, so no log file is written to the current directory
        logger=_make_logger(),
    )
    krg = KRG(config)
    krg.set_training_data(
        [{"coord": x, "value": y, "fid": 0} for x, y in zip(X_TRAIN, y_train)]
    )
    krg.train()
    f_min = float(np.min(y_train))

    yield config, krg, f_min

    # Cleanup
    Config.destroy_instance(destroy_logger=True)


def _set_g(config: Config, g: int) -> Config:
    """Helper function to set g for GEI."""
    config.set_setting("infill.g", g)
    assert config["infill.g"] == g
    return config


def _get_GEI_infill(
    config: Config, krg: KRG, x_row: np.ndarray, g: int, f_min: float
) -> float:
    """Get the GEI value for a given row, with f_min as the best value.

    best_value is passed explicitly: calc_infill_value defaults to the surrogate
    optimum, while this test checks the GEI formula against a quadrature with f_min.
    """
    return float(GEI(_set_g(config, g)).calc_infill_value(x_row, krg, best_value=f_min))


def gei_quadrature(yhat: float, s: float, f_min: float, g: int) -> float:
    """Get the quadrature points for GEI."""
    if s <= 0.0:
        return max(f_min - yhat, 0.0) ** g
    u = (f_min - yhat) / s
    integ, _ = quad(lambda z: (u - z) ** g * norm.pdf(z), -np.inf, u, limit=200)
    return s**g * integ


@pytest.mark.parametrize("g", G_VALUES)
@pytest.mark.parametrize("xi", range(len(X_TEST)))
@pytest.mark.verification
def test_gei_matches_quadrature(krg_fixture: Generator, xi: int, g: int) -> None:
    """Test that GEI matches the quadrature points for various g values."""
    config, krg, f_min = krg_fixture
    x = X_TEST[xi]
    got = _get_GEI_infill(config, krg, x, g, f_min)
    yhat, s = krg.predict(x, return_variance=True)
    ref = gei_quadrature(yhat, math.sqrt(s), f_min, g)
    assert np.isclose(got, ref, rtol=1e-5, atol=1e-4), (
        f"GEI and quadrature are not equal: GEI: {got}, quadrature {ref}\n"
        f"absolute diff: {np.abs(got - ref)}, relative diff: {np.abs(got - ref) / ref}"
    )


@pytest.mark.parametrize("xi", range(len(X_TEST)))
def test_g1_reduces_to_expected_improvement(krg_fixture: Generator, xi: int) -> None:
    """Test that GEI with g=1 reduces to expected improvement (EI)."""
    config, krg, f_min = krg_fixture
    x = X_TEST[xi]
    got = _get_GEI_infill(config, krg, x, 1, f_min)

    ei = EI(config)
    ei_ref = ei.calc_infill_value(x, krg, best_value=f_min)

    assert np.isclose(got, ei_ref, rtol=1e-5, atol=1e-4), (
        f"EI and GEI are not equal: EI: {ei_ref}, GEI: {got},\n"
        f"absolute diff: {np.abs(got - ei_ref)}, "
        f"relative diff: {np.abs(got - ei_ref) / ei_ref}"
    )
