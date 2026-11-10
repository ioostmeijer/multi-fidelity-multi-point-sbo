import dataclasses

import numpy as np
import pytest

from sbo.benchmarks import (
    ALL_BENCHMARKS,
    FORRESTER,
    FunctionSet,
    MultiFidelityFunction,
    as_2d,
    get_benchmark,
)


def _square(x: np.ndarray) -> np.ndarray:
    """Return the sum of squares per point (a simple fidelity for the tests)."""
    return np.sum(as_2d(x, 2) ** 2, axis=1)


def _make(**changes: object) -> MultiFidelityFunction:
    """Create a valid 2D benchmark, with some fields changed."""
    fields = {
        "name": "test_problem",
        "functions": (_square,),
        "l_bound": np.array([-1.0, -1.0]),
        "u_bound": np.array([1.0, 1.0]),
        "x_opt": np.array([0.0, 0.0]),
        "f_opt": 0.0,
    }
    fields.update(changes)
    return MultiFidelityFunction(**fields)


@pytest.mark.unit
def test_valid_benchmark_converts_fields() -> None:
    """Test that lists are converted to read-only float arrays and a tuple."""
    benchmark = _make(functions=[_square, _square], l_bound=[-1, -1], f_opt=0)

    assert benchmark.functions == (_square, _square)
    assert benchmark.l_bound.dtype == float
    assert benchmark.n_dim == 2
    assert benchmark.n_fidelities == 2
    assert isinstance(benchmark.f_opt, float)


@pytest.mark.unit
def test_optimum_is_optional() -> None:
    """Test that x_opt and f_opt can be left out."""
    benchmark = _make(x_opt=None, f_opt=None)

    assert benchmark.x_opt is None
    assert benchmark.f_opt is None


@pytest.mark.unit
def test_benchmark_is_read_only() -> None:
    """Test that neither the fields nor the arrays in them can be changed."""
    benchmark = _make()

    with pytest.raises(dataclasses.FrozenInstanceError):
        benchmark.l_bound = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="read-only"):
        benchmark.l_bound[0] = 0.5
    with pytest.raises(ValueError, match="read-only"):
        benchmark.x_opt[0] = 0.5


@pytest.mark.unit
def test_input_arrays_are_copied() -> None:
    """Test that changing the original array doesn't change the benchmark."""
    l_bound = np.array([-1.0, -1.0])
    benchmark = _make(l_bound=l_bound)

    l_bound[0] = -5.0

    assert benchmark.l_bound[0] == -1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"name": ""}, ValueError, "name"),
        ({"name": "Test Problem"}, ValueError, "name"),
        ({"functions": ()}, ValueError, "at least one"),
        ({"functions": (_square, 1.0)}, TypeError, "callable"),
        ({"l_bound": np.array([-1.0])}, ValueError, "same shape"),
        ({"l_bound": np.array([[-1.0, -1.0]])}, ValueError, "1D"),
        ({"u_bound": np.array([1.0, np.inf])}, ValueError, "finite"),
        ({"l_bound": np.array([1.0, -1.0])}, ValueError, "smaller"),
        ({"x_opt": np.array([0.0])}, ValueError, "shape"),
        ({"x_opt": np.array([0.0, 2.0])}, ValueError, "outside"),
    ],
)
def test_invalid_benchmark_raises(
    changes: dict, error: type[Exception], message: str
) -> None:
    """Test the checks in __post_init__."""
    with pytest.raises(error, match=message):
        _make(**changes)


@pytest.mark.unit
def test_as_2d_shapes() -> None:
    """Test that one point and several points both become a 2D array."""
    assert as_2d(np.array([0.1, 0.2]), 2).shape == (1, 2)
    assert as_2d(np.array([[0.1, 0.2], [0.3, 0.4]]), 2).shape == (2, 2)
    assert as_2d([0.5], 1).shape == (1, 1)


@pytest.mark.unit
def test_as_2d_wrong_dimension_raises() -> None:
    """Test that points with the wrong number of dimensions are rejected."""
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        as_2d(np.array([0.1, 0.2, 0.3]), 2)


@pytest.mark.unit
def test_benchmark_satisfies_function_set_protocol() -> None:
    """Test that a benchmark has the interface Config.from_json needs."""
    assert isinstance(_make(), FunctionSet)
    assert not isinstance(object(), FunctionSet)


@pytest.mark.unit
def test_get_benchmark() -> None:
    """Test looking up a benchmark by name, ignoring case and spaces."""
    assert get_benchmark("forrester") is FORRESTER
    assert get_benchmark(" Forrester ") is FORRESTER


@pytest.mark.unit
def test_get_unknown_benchmark_raises() -> None:
    """Test that an unknown name lists the available benchmarks."""
    with pytest.raises(ValueError, match="Available benchmarks: forrester"):
        get_benchmark("does_not_exist")


@pytest.mark.unit
@pytest.mark.parametrize("benchmark", ALL_BENCHMARKS, ids=lambda b: b.name)
def test_all_benchmarks_are_consistent(benchmark: MultiFidelityFunction) -> None:
    """Test every registered benchmark: unique name, output shape and optimum."""
    assert [b.name for b in ALL_BENCHMARKS].count(benchmark.name) == 1

    rng = np.random.default_rng(0)
    points = rng.uniform(benchmark.l_bound, benchmark.u_bound, (5, benchmark.n_dim))
    for function in benchmark.functions:
        assert function(points).shape == (5,)
        assert function(points[0]).shape == (1,)

    if benchmark.x_opt is not None and benchmark.f_opt is not None:
        assert benchmark.functions[0](benchmark.x_opt)[0] == pytest.approx(
            benchmark.f_opt, abs=1e-8
        )
