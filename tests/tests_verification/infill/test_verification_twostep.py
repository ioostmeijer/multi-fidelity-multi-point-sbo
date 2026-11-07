"""The verification of the TwoStep algorithm.

Idea
    - sample location should be equal to single fidelity, for every iteration
    - correct calculation of the Jensen-Shannon distance (JSD)
    - correct fidelity level returned
"""

import numpy as np
import pytest
from mf2 import forrester

from sbo import SBO, Config
from sbo.surrogate import SurrogateBaseWithVariancePerFidelity


class MockSurrogate(SurrogateBaseWithVariancePerFidelity):
    """Mock surrogate to test the JSD calculation and fidelity selection."""

    def __init__(
        self, mu: dict[int, float], var: dict[int, float], config: Config
    ) -> None:
        """Initialise the mock surrogate with specified mean and variance."""
        super().__init__(config)
        self._mu = mu
        self._var = var
        self.samples = {
            0: [{"coord": np.ndarray([0]), "fid": 0, "value": 0}],
            1: [],
        }

    def _initialise_model(self, *args, **kwargs) -> bool:
        """Initialise the surrogate model."""
        return True

    def train(self, *args, **kwargs) -> None:
        """Mock train function."""
        self._trained = True

    def predict_values(self, x: np.ndarray, fidelity: int = 0) -> np.ndarray:
        """Predict the mean value for the given fidelity."""
        return np.array([self._mu[fidelity]])

    def predict_variance_fidelity(self, x: np.ndarray, fidelity: int = 0) -> np.ndarray:
        """Predict the variance for the given fidelity."""
        return np.array([self._var[fidelity]])

    def predict(
        self, x: np.ndarray, *args, return_variance: bool = True, **kwargs
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict the mean and variance for the given fidelity."""
        if return_variance:
            return self.predict_values(x), self.predict_variance_fidelity(x)
        else:
            return self.predict_values(x)


@pytest.mark.verification
def test_sample_location() -> None:
    """The sample locations and infill value should be equal to a single fidelity.

    Run an optimisation with TwoStep to get realistic training data. For each
    iteration, train the surrogate on the samples available at the start of that
    iteration, and compare TwoStep with single-fidelity EI on the same data:

    - the infill values on a grid must be equal;
    - the proposed sample locations must be equal, or, if two (almost) equally good
      maxima exist, the infill values at both proposals must be equal.

    The proposals are taken directly from the infill strategies, not from the
    evaluated samples: the optimiser replaces a proposal that duplicates an existing
    sample by a random point, which is not a choice of the infill strategy.
    """
    domain_size = np.max(np.asarray(forrester.u_bound) - np.asarray(forrester.l_bound))
    location_atol = 1e-3 * domain_size

    config = Config(
        used_algorithms={
            "optimisation": "EgoSequential",
            "surrogate": "HK",
            "infill": "TwoStep",
        },
        lb=forrester.l_bound,
        ub=forrester.u_bound,
        functions=forrester.functions,
    )
    sbo = SBO(config)
    _ = sbo.optimum
    twostep_optimisation_data = sbo.optimisation_data.copy()

    # Training data at the start of every iteration (the samples of the previous one)
    training_data = {}
    prev_iteration_samples = []
    for fid_points in twostep_optimisation_data["init"]["sample_points"].values():
        prev_iteration_samples.extend(fid_points)
    for iteration, iter_data in twostep_optimisation_data.items():
        if isinstance(iteration, int) or iteration.isdigit():
            training_data[iteration] = prev_iteration_samples.copy()
            prev_iteration_samples = [
                p
                for fid_points in iter_data["sample_points"].values()
                for p in fid_points
            ]

    x_test = np.linspace(forrester.l_bound, forrester.u_bound, 1000)

    def infill_results(sbo: SBO) -> dict:
        """Infill values on the grid and the proposed point for every iteration."""
        results = {}
        for iteration, samples in training_data.items():
            sbo._surrogate.set_training_data(samples)
            sbo._surrogate.train()
            values = np.array(
                [sbo._infill.calc_infill_value(x, sbo._surrogate) for x in x_test]
            )
            new_task, infill_value = sbo._infill(sbo._surrogate)
            results[iteration] = {
                "grid_values": values,
                "coord": np.asarray(new_task[0][0]["coord"]),
                "infill_value": float(infill_value),
            }
        return results

    twostep_results = infill_results(sbo)
    Config.destroy_instance(destroy_logger=True)

    config = Config(
        used_algorithms={
            "optimisation": "EgoSequential",
            "surrogate": "HK",
            "infill": "EI",
        },
        lb=forrester.l_bound,
        ub=forrester.u_bound,
        functions=forrester.functions,
    )
    ei_results = infill_results(SBO(config))

    assert twostep_results, "The TwoStep run has no iterations to compare."
    for i, ts in twostep_results.items():
        ei = ei_results[i]
        assert np.allclose(
            ei["grid_values"], ts["grid_values"], atol=1e-4, rtol=1e-5
        ), f"Infill values are not equal for iteration {i}."

        same_location = np.allclose(ei["coord"], ts["coord"], atol=location_atol)
        equally_good = np.isclose(
            ei["infill_value"], ts["infill_value"], atol=1e-8, rtol=1e-3
        )
        assert same_location or equally_good, (
            f"Sample locations are not equal for iteration {i}: "
            f"TwoStep {ts['coord']} (infill value {ts['infill_value']}), "
            f"EI {ei['coord']} (infill value {ei['infill_value']})."
        )


@pytest.mark.verification
def test_jsd_calculation() -> None:
    """Test if the correct list of accurate enough fidelities is returned.

    With the default threshold of 0.7
    mu    JSD    fidelities
    0     0      [0, 1]
    1sig  0.4    [0, 1]
    2sig  0.697  [0, 1]
    3sig  0.872  [0]
    50sig 1      [0]
    """
    tests = [
        {"name": "mu 0", "mus": {0: 1, 1: 1}, "vars": {0: 1, 1: 1}, "expected": [0, 1]},
        {
            "name": "mu 1sig",
            "mus": {0: 0, 1: 1},
            "vars": {0: 1, 1: 1},
            "expected": [0, 1],
        },
        {
            "name": "mu 2sig",
            "mus": {0: 0, 1: 2},
            "vars": {0: 1, 1: 1},
            "expected": [0, 1],
        },
        {"name": "mu 3sig", "mus": {0: 0, 1: 3}, "vars": {0: 1, 1: 1}, "expected": [0]},
        {
            "name": "mu 50sig",
            "mus": {0: 0, 1: 50},
            "vars": {0: 1, 1: 1},
            "expected": [0],
        },
    ]

    config = Config(
        used_algorithms={
            "optimisation": "EgoSequential",
            "surrogate": "HK",
            "infill": "TwoStep",
        },
        lb=forrester.l_bound,
        ub=forrester.u_bound,
        functions=forrester.functions,
    )
    sbo = SBO(config)

    for test in tests:
        mock_surrogate = MockSurrogate(mu=test["mus"], var=test["vars"], config=config)
        accurate_fidelities = sbo._infill._get_accurate_enough_fidelities(
            x=np.array([0]), surrogate=mock_surrogate
        )
        assert accurate_fidelities == test["expected"], (
            f"Test {test['name']} failed: "
            f"expected {test['expected']},"
            f" got {accurate_fidelities}"
        )


@pytest.mark.verification
def test_fidelity_calculation() -> None:
    """Test if the correct list of accurate enough fidelities is returned.

    With the default threshold of 0.7
    mu    JSD    fidelities    selected fidelity
    0     0      [0, 1]        1
    1sig  0.4    [0, 1]        1
    2sig  0.697  [0, 1]        1
    3sig  0.872  [0]           0
    50sig 1      [0]           0
    """
    tests = [
        {"name": "mu 0", "mus": {0: 1, 1: 1}, "vars": {0: 1, 1: 1}, "expected": 1},
        {"name": "mu 1sig", "mus": {0: 0, 1: 1}, "vars": {0: 1, 1: 1}, "expected": 1},
        {"name": "mu 2sig", "mus": {0: 0, 1: 2}, "vars": {0: 1, 1: 1}, "expected": 1},
        {"name": "mu 3sig", "mus": {0: 0, 1: 3}, "vars": {0: 1, 1: 1}, "expected": 0},
        {"name": "mu 50sig", "mus": {0: 0, 1: 50}, "vars": {0: 1, 1: 1}, "expected": 0},
    ]

    config = Config(
        used_algorithms={
            "optimisation": "EgoSequential",
            "surrogate": "HK",
            "infill": "TwoStep",
        },
        lb=forrester.l_bound,
        ub=forrester.u_bound,
        functions=forrester.functions,
    )
    sbo = SBO(config)

    for test in tests:
        mock_surrogate = MockSurrogate(mu=test["mus"], var=test["vars"], config=config)
        task, _ = sbo._infill.get_new_points(surrogate=mock_surrogate)
        selected_fidelity = task[0][0]["fid"]
        assert selected_fidelity == test["expected"], (
            f"Test {test['name']} failed: "
            f"expected {test['expected']}, "
            f"got {selected_fidelity}"
        )
