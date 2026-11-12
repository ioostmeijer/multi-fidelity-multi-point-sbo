import copy
import pathlib
import pickle
import time

import numpy as np
import pytest
from dummy_surrogate import DummySurrogate

from sbo import Config
from sbo.optimisation import OptimisationBase


def mock_config() -> Config:
    """Create a mock Config object for testing.

    Returns
    -------
    A mock Config object.
    """
    return Config(
        used_algorithms={"optimisation": "OptimisationBase"},
        functions=[lambda x: x**2],
        lb=[-10],
        ub=[10],
        settings={},
    )


@pytest.mark.unit
def test_init() -> None:
    """Test initialisation of OptimisationBase."""
    optimisation = OptimisationBase(config=mock_config())
    assert isinstance(optimisation, OptimisationBase)
    assert optimisation.config == mock_config()


@pytest.mark.unit
def test_solve_not_implemented() -> None:
    """Test that solve method raises NotImplementedError."""
    optimisation = OptimisationBase(config=mock_config())
    with pytest.raises(NotImplementedError):
        optimisation.solve(None, None)
    with pytest.raises(NotImplementedError):
        optimisation(None, None)


@pytest.mark.unit
def test_evaluate_point() -> None:
    """Test evaluation of a point."""
    optimisation = OptimisationBase(config=mock_config())

    point_init = {"coord": np.array([1.0]), "fid": 0}
    point_evaluated = optimisation.evaluate_point(point_init, inplace=False)
    point_expected = {"coord": np.array([1.0]), "fid": 0, "value": 1.0}
    assert "start_time" in point_evaluated
    assert "end_time" in point_evaluated
    assert point_evaluated["coord"] == point_expected["coord"]
    assert point_evaluated["fid"] == point_expected["fid"]
    assert point_evaluated["value"] == point_expected["value"]

    point_inplace_init = {"coord": np.array([2.0]), "fid": 0}
    point_inplace_evaluated = optimisation.evaluate_point(
        point_inplace_init, inplace=True
    )
    point_expected = {"coord": np.array([2.0]), "fid": 0, "value": 4.0}
    for p in [point_inplace_evaluated, point_inplace_init]:
        assert "start_time" in p
        assert "end_time" in p
        assert p["coord"] == point_expected["coord"]
        assert p["fid"] == point_expected["fid"]
        assert p["value"] == point_expected["value"]


@pytest.mark.unit
def test_calc_stopping_criteria() -> None:
    """Test calculation of stopping criteria."""
    optimisation = OptimisationBase(config=mock_config())

    # The iteration counter starts at 0, so iteration index max_iterations - 2 means
    # max_iterations - 1 completed iterations
    iteration_continue = mock_config().get("optimisation.max_iterations") - 2
    iteration_stop = mock_config().get("optimisation.max_iterations") + 1

    # One HF call is reserved for the final evaluation of the surrogate optimum
    function_calls_continue = {
        "total": mock_config().get("optimisation.max_function_calls") - 2,
        0: mock_config().get("optimisation.max_function_calls_HF") - 2,
    }
    function_calls_HF_stop = {
        "total": mock_config().get("optimisation.max_function_calls") - 1,
        0: mock_config().get("optimisation.max_function_calls_HF") + 1,
    }

    best_val = 0.0
    opt_data_continue = {
        "total": {"some total data": 123},
        "final": {"some final data": 456},
    }
    opt_data_continue.update(
        {
            index: {
                "best_value": best_val
                + (mock_config().get("optimisation.convergence_threshold") * 2),
                "surrogate_optimum": {"value": 0.0 + index, "coord": [0, 0], "fid": 0},
            }
            for index in range(iteration_stop + 1)
        }
    )
    opt_data_stop = copy.deepcopy(opt_data_continue)
    opt_data_stop[0] = {"best_value": best_val}
    opt_data_stop[len(opt_data_continue) + 1] = opt_data_stop[0].copy()
    for k in opt_data_stop:
        opt_data_stop[k]["surrogate_optimum"] = {
            "value": 0.0,
            "coord": [0, 0],
            "fid": 0,
        }

    infill_value_continue = mock_config().get("optimisation.min_infill_value") + 1e-3
    infill_value_stop = mock_config().get("optimisation.min_infill_value") - 1e-3

    total_function_calls_continue = {"function_calls": {0: 0}}
    total_function_calls_stop = {
        "function_calls": {
            fid: mock_config().get("optimisation.max_function_calls") + 1
            for fid in range(mock_config().num_fidelities)
        }
    }

    cases = [
        (
            iteration_continue,
            function_calls_continue,
            best_val,
            opt_data_continue,
            infill_value_continue,
            total_function_calls_continue,
            False,
        ),
        (
            iteration_stop,
            function_calls_continue,
            best_val,
            opt_data_continue,
            infill_value_continue,
            total_function_calls_continue,
            True,
        ),
        (
            iteration_continue,
            function_calls_continue,
            best_val,
            opt_data_stop,
            infill_value_continue,
            total_function_calls_continue,
            True,
        ),
        (
            iteration_continue,
            function_calls_continue,
            best_val,
            opt_data_continue,
            infill_value_continue,
            total_function_calls_stop,
            True,
        ),
        (
            iteration_continue,
            function_calls_HF_stop,
            best_val,
            opt_data_continue,
            infill_value_continue,
            total_function_calls_continue,
            True,
        ),
        (
            iteration_continue,
            function_calls_continue,
            best_val,
            opt_data_continue,
            infill_value_stop,
            total_function_calls_continue,
            True,
        ),
    ]

    for idx, (it, fc, bv, opt_data, inf, tot, expected) in enumerate(cases):
        result = optimisation._calc_stopping_criteria(
            iteration=it,
            function_call=fc,
            best_val=bv,
            infill_value=inf,
            total_data=tot,
            optimisation_data=opt_data,
        )
        assert result is expected, (
            f"case {idx} failed: got {result}, expected {expected}, {cases[idx]}"
        )


@pytest.mark.unit
def test_run_end_iteration() -> None:
    """Test running end of iteration updates."""
    optimisation = OptimisationBase(config=mock_config())

    # prepare surrogate with one sample
    initial_samples = {0: [{"coord": np.array([0.0]), "value": 0.0, "fid": 0}]}
    surrogate = DummySurrogate(config=mock_config(), samples=initial_samples)

    iteration = 1
    function_calls = {"total": 1, 0: 1}
    best_val = 0.0
    prev_best_val = 1.0
    infill_value = 0.5
    best_coord = np.array([0.0])
    start_time_iteration = time.time()
    curr_surr_opt = {"coord": [0], "value": 0.0, "fid": 0}
    prev_surr_opt = curr_surr_opt

    result = optimisation._run_end_iteration(
        iteration=iteration,
        function_calls=function_calls,
        best_val=best_val,
        prev_best_val=prev_best_val,
        infill_value=infill_value,
        best_coord=best_coord,
        surrogate=surrogate,
        start_time_iteration=start_time_iteration,
        current_surrogate_optimum=curr_surr_opt,
        prev_surrogate_optimum=prev_surr_opt,
    )

    # result must be boolean, optimisation data must be updated
    assert isinstance(result, bool)
    assert iteration in optimisation.optimisation_data
    saved = optimisation.optimisation_data[iteration]
    assert np.array_equal(saved["best_coord"], best_coord)
    assert saved["best_value"] == best_val
    assert saved["infill_value"] == infill_value
    assert saved["function_calls"] == function_calls
    # total aggregate updated
    assert optimisation.optimisation_data["total"]["best_value"] == best_val
    assert (
        optimisation.optimisation_data["total"]["function_calls"][0]
        >= function_calls[0]
    )


@pytest.mark.unit
def test_save_optimisation_data(tmp_path: pathlib.Path) -> None:
    """Test saving optimisation data to a pickle file."""
    config = mock_config()

    save_dir = tmp_path / "save_dir"
    config["save_folder"] = str(save_dir)
    optimisation = OptimisationBase(config=config)

    # fill some optimisation data
    optimisation.optimisation_data["total"]["best_coord"] = np.array([1.0])
    optimisation.optimisation_data["total"]["best_value"] = 2.0

    file_name = "test_run_data"
    optimisation.save_optimisation_data(file_name)

    saved_path = save_dir / f"{file_name}.pkl"
    assert saved_path.exists()

    with open(saved_path, "rb") as f:
        loaded = pickle.load(f)  # noqa: S301 (file written by the test)
    assert loaded == optimisation.optimisation_data


@pytest.mark.unit
def test_kriging_believer() -> None:
    """Test the kriging believer infill method."""
    optimisation = OptimisationBase(config=mock_config())

    # initial samples for surrogate
    initial_samples = {0: [{"coord": np.array([0.0]), "value": 0.0, "fid": 0}]}
    surrogate = DummySurrogate(config=mock_config(), samples=initial_samples)

    sample_locations = [{"coord": np.array([1.0]), "fid": 0}]

    # infill returns a new point
    def infill_fn(surr: DummySurrogate) -> tuple[list[dict], float]:
        """Mock infill function that returns a new point."""
        return ([{"coord": np.array([2.0]), "fid": 0}], 123.0)

    original_data = copy.deepcopy(surrogate.samples)
    new_points, infill_value = optimisation._kriging_believer(
        surrogate=surrogate, infill=infill_fn, sample_locations=sample_locations
    )

    # returned infill value must be infinity (KB cannot stop)
    assert infill_value == float("inf")
    # new_points should be whatever the infill function returned
    assert isinstance(new_points, list)
    assert new_points[0]["fid"] == 0
    # surrogate training data must have been restored
    assert surrogate.samples == original_data


@pytest.mark.unit
def test_constant_liar() -> None:
    """Test the constant liar infill method."""
    config = mock_config()
    config.settings["optimisation.multipoint_algorithm"] = "CL"
    optimisation = OptimisationBase(config=config)

    # make surrogate samples that contain values for fid 0
    initial_samples = {
        0: [
            {"coord": np.array([0.0]), "value": 1.0, "fid": 0},
            {"coord": np.array([1.0]), "value": 3.0, "fid": 0},
        ]
    }
    surrogate = DummySurrogate(config=config, samples=initial_samples)

    sample_locations = [{"coord": np.array([2.0]), "fid": 0}]

    # infill returns a new point
    def infill_fn(surr: DummySurrogate) -> tuple[list[dict], float]:
        """Mock infill function that returns a new point."""
        return ([{"coord": np.array([4.0]), "fid": 0}], 456.0)

    original_data = copy.deepcopy(surrogate.samples)
    new_points, infill_value = optimisation._constant_liar(
        surrogate=surrogate, infill=infill_fn, sample_locations=sample_locations
    )

    # returned infill value must be infinity (CL cannot stop)
    assert infill_value == float("inf")
    # new_points should be whatever the infill function returned
    assert isinstance(new_points, list)
    # surrogate training data must have been restored
    assert surrogate.samples == original_data


@pytest.mark.unit
def test_remove_duplicates_within_one_batch() -> None:
    """Two identical points in one batch: the second is replaced."""
    config = mock_config()
    optimisation = OptimisationBase(config=config)
    surrogate = DummySurrogate(
        config=config,
        mu=0.0,
        optimum_value={"coord": np.array([0.0]), "value": [0.0]},
    )
    surrogate.samples = {0: []}

    point = {"coord": np.array([0.3]), "fid": 0}
    tasks = optimisation._remove_duplicates(
        [[dict(point)], [dict(point)]], surrogate, []
    )

    assert np.allclose(tasks[0][0]["coord"], [0.3])
    assert not np.allclose(tasks[1][0]["coord"], [0.3])


@pytest.mark.unit
def test_evaluate_point_short_function_delay_uses_last_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A function_delay shorter than the fidelities uses its last value."""
    Config.destroy_instance()
    config = Config(
        used_algorithms={"optimisation": "OptimisationBase"},
        functions=[lambda x: x**2, lambda x: x**2 + 1],
        lb=[-10],
        ub=[10],
        settings={"function_delay": [0.0]},
    )
    optimisation = OptimisationBase(config=config)
    sleeps = []
    monkeypatch.setattr("sbo.optimisation.Base.time.sleep", sleeps.append)

    optimisation.evaluate_point({"coord": np.array([1.0]), "fid": 1}, inplace=True)

    assert sleeps == [0.0]


class _RecordingSurrogate(DummySurrogate):
    """Dummy surrogate that predicts its fidelity argument and records it."""

    supports_fidelity_prediction = True

    def __init__(self, *args, **kwargs) -> None:
        self.requested = []
        super().__init__(*args, **kwargs)

    def predict_values(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        fidelity = kwargs.get("fidelity")
        self.requested.append(fidelity)
        # value 10 + fidelity, or -1 if no fidelity was given (HF prediction)
        value = -1.0 if fidelity is None else 10.0 + fidelity
        return np.full((len(np.atleast_2d(x)), 1), value)


class _HFOnlySurrogate(_RecordingSurrogate):
    """Multi-fidelity dummy surrogate that can not predict per fidelity."""

    supports_fidelity_prediction = False


def _two_fidelity_config() -> Config:
    """Two-fidelity config for the Kriging Believer tests."""
    Config.destroy_instance()
    return Config(
        used_algorithms={"optimisation": "OptimisationBase"},
        functions=[lambda x: x**2, lambda x: x**2 + 1],
        lb=[-10],
        ub=[10],
        settings={},
    )


def _lies_seen_by_infill(surrogate_cls: type) -> tuple[dict, list]:
    """Run KB with an HF and an LF pending point; return the lies and requests."""
    config = _two_fidelity_config()
    optimisation = OptimisationBase(config=config)
    surrogate = surrogate_cls(
        config=config,
        samples={0: [{"coord": np.array([0.0]), "value": 0.0, "fid": 0}], 1: []},
    )
    sample_locations = [
        {"coord": np.array([1.0]), "fid": 0},
        {"coord": np.array([2.0]), "fid": 1},
    ]
    seen = {}

    def infill_fn(surr: DummySurrogate) -> tuple[list[dict], float]:
        for fid, points in surr.samples.items():
            for p in points:
                seen[(fid, float(np.asarray(p["coord"]).ravel()[0]))] = p["value"]
        return ([{"coord": np.array([3.0]), "fid": 0}], 1.0)

    optimisation._kriging_believer(
        surrogate=surrogate, infill=infill_fn, sample_locations=sample_locations
    )
    return seen, surrogate.requested


@pytest.mark.unit
def test_kriging_believer_lies_at_point_fidelity() -> None:
    """A surrogate that predicts per fidelity gets the lie at the point's fidelity."""
    seen, requested = _lies_seen_by_infill(_RecordingSurrogate)

    assert requested == [0, 1]
    assert seen[(0, 1.0)] == 10.0
    assert seen[(1, 2.0)] == 11.0


@pytest.mark.unit
def test_kriging_believer_hf_lie_without_fidelity_support() -> None:
    """A surrogate without per-fidelity prediction gets the HF prediction as lie."""
    seen, requested = _lies_seen_by_infill(_HFOnlySurrogate)

    assert requested == [None, None]
    assert seen[(0, 1.0)] == -1.0
    assert seen[(1, 2.0)] == -1.0


@pytest.mark.unit
def test_evaluate_point_with_float_returning_function() -> None:
    """Test that evaluate_point handles a function that returns a plain float."""
    Config.destroy_instance()
    config = Config(
        used_algorithms={"optimisation": "OptimisationBase"},
        functions=[lambda x: float(x[0] ** 2)],
        lb=[-10],
        ub=[10],
        settings={},
    )
    optimisation = OptimisationBase(config=config)

    point = optimisation.evaluate_point({"coord": np.array([3.0]), "fid": 0})

    assert point["value"] == 9.0
