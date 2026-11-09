from typing import Generator

import numpy as np
import pytest
from logger_helper import _make_logger
from smt.applications import EGO
from smt.design_space import DesignSpace
from smt.surrogate_models import KRG as SMT_KRG

from sbo import Config
from sbo.benchmarks import FORRESTER
from sbo.infill import EI
from sbo.surrogate import KRG

X_TRAIN = np.array([0.0, 0.375, 0.4, 0.721, 0.832, 0.969]).reshape(-1, 1)
X_TEST = np.linspace(0.0, 1.0, 100).reshape(-1, 1)


@pytest.fixture(scope="module")
def fixture() -> Generator:
    """Fixture to provide a trained KRG model and EI and reference SMT."""
    # Ensure new Config
    Config.destroy_instance(destroy_logger=True)

    # Get the samples
    y_train = FORRESTER.functions[0](X_TRAIN)

    # Get custom EI and krg
    config = Config(
        used_algorithms={
            "surrogate": "KRG",
            "infill": "EI",
            "optimisation": "EgoSequential",
        },
        lb=FORRESTER.l_bound,
        ub=FORRESTER.u_bound,
        functions=[FORRESTER.functions[0]],
        # Module fixture, set up before the test runs in its temporary directory:
        # use the test logger, so no log file is written to the current directory
        logger=_make_logger(),
    )
    krg_custom = KRG(config)
    krg_custom.set_training_data(
        [{"coord": x, "value": y, "fid": 0} for x, y in zip(X_TRAIN, y_train)]
    )
    krg_custom.train()
    ei_custom = EI(config)

    # Get the smt EI
    design_space = DesignSpace(config.bounds, seed=config["random_seed"])
    smt_ego = EGO(
        surrogate=SMT_KRG(
            design_space=design_space,
            seed=config["random_seed"],
            print_global=False,
        ),
        criterion="EI",
        xdoe=X_TRAIN,
        ydoe=y_train,
        seed=config["random_seed"],
        fun=FORRESTER.functions[0],
        n_iter=1,
    )
    smt_x_opt, smt_y_opt, _, smt_x_data, smt_y_data = smt_ego.optimize(
        FORRESTER.functions[0]
    )
    smt_ego.gpr.set_training_values(X_TRAIN, y_train)
    smt_ego.gpr.train()

    yield krg_custom, ei_custom, smt_ego

    # Cleanup
    Config.destroy_instance(destroy_logger=True)


@pytest.mark.parametrize("xi", range(len(X_TEST)))
@pytest.mark.verification
def test_ei_matches_smt(fixture: Generator, xi: int) -> None:
    """Test that the custom EI matches the SMT EI."""
    krg_custom, ei_custom, smt_ego = fixture
    custom_ei_value = ei_custom.calc_infill_value(X_TEST[xi], krg_custom)
    smt_ei_value = smt_ego.EI(X_TEST[xi])[0][0]
    assert np.isclose(
        custom_ei_value, smt_ei_value, atol=1e-4, rtol=1e-5
    ), "EI values are not close: absolute_diff: {}, relative_diff: {}".format(
        custom_ei_value - smt_ei_value, (custom_ei_value - smt_ei_value) / smt_ei_value
    )
