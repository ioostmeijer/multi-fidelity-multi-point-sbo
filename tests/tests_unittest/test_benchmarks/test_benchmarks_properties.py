import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sbo.benchmarks import ALL_BENCHMARKS, MultiFidelityFunction, as_2d

# Fractions of the design space per variable: 0 is the lower bound and 1 the upper bound
_UNIT_FLOATS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@st.composite
def _points_in_bounds(
    draw: st.DrawFn, benchmark: MultiFidelityFunction, max_points: int = 10
) -> np.ndarray:
    """Draw points with shape (n, n_dim) inside the bounds of a benchmark.

    Parameters
    ----------
    draw
        The Hypothesis draw function.
    benchmark
        The benchmark that gives the bounds.
    max_points
        The maximum number of points n.

    Returns
    -------
    np.ndarray
        The points, with shape (n, n_dim).
    """
    n_points = draw(st.integers(min_value=1, max_value=max_points))
    fractions = draw(arrays(float, (n_points, benchmark.n_dim), elements=_UNIT_FLOATS))
    points = benchmark.l_bound + fractions * (benchmark.u_bound - benchmark.l_bound)
    return np.clip(points, benchmark.l_bound, benchmark.u_bound)


@pytest.mark.unit
@pytest.mark.parametrize("benchmark", ALL_BENCHMARKS, ids=lambda b: b.name)
@given(data=st.data())
def test_benchmark_output_is_finite_vector(
    benchmark: MultiFidelityFunction, data: st.DataObject
) -> None:
    """Test that every fidelity maps (n, n_dim) points to n finite values."""
    points = data.draw(_points_in_bounds(benchmark), label="points")
    for function in benchmark.functions:
        values = function(points)
        assert values.shape == (points.shape[0],)
        assert np.all(np.isfinite(values))


@pytest.mark.unit
@pytest.mark.parametrize("benchmark", ALL_BENCHMARKS, ids=lambda b: b.name)
@given(data=st.data())
def test_benchmark_points_are_independent(
    benchmark: MultiFidelityFunction, data: st.DataObject
) -> None:
    """Test that a point gets the same value alone as in a batch of points."""
    points = data.draw(_points_in_bounds(benchmark, max_points=5), label="points")
    for function in benchmark.functions:
        batch = function(points)
        for point, value in zip(points, batch, strict=True):
            np.testing.assert_allclose(function(point), [value], rtol=1e-12)


@pytest.mark.unit
@given(
    n_points=st.integers(min_value=1, max_value=10),
    n_dim=st.integers(min_value=1, max_value=10),
)
def test_as_2d_keeps_2d_points(n_points: int, n_dim: int) -> None:
    """Test that points with shape (n, n_dim) keep their shape and values."""
    points = np.arange(n_points * n_dim).reshape(n_points, n_dim)
    result = as_2d(points, n_dim)
    assert result.shape == (n_points, n_dim)
    assert result.dtype == float
    np.testing.assert_array_equal(result, points)


@pytest.mark.unit
@given(n_dim=st.integers(min_value=1, max_value=10))
def test_as_2d_reads_1d_input_as_one_point(n_dim: int) -> None:
    """Test that one point with shape (n_dim,) becomes shape (1, n_dim)."""
    point = np.arange(n_dim)
    result = as_2d(point, n_dim)
    assert result.shape == (1, n_dim)
    np.testing.assert_array_equal(result[0], point)


@pytest.mark.unit
@given(
    shape=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
    n_dim=st.integers(min_value=1, max_value=5),
)
def test_as_2d_rejects_wrong_dimension(shape: list[int], n_dim: int) -> None:
    """Test that every shape other than (n_dim,) or (n, n_dim) raises a ValueError."""
    valid = (len(shape) == 1 and shape[0] == n_dim) or (
        len(shape) == 2 and shape[1] == n_dim
    )
    points = np.zeros(shape)
    if valid:
        assert as_2d(points, n_dim).shape[1] == n_dim
    else:
        with pytest.raises(ValueError, match="Expected points with shape"):
            as_2d(points, n_dim)
