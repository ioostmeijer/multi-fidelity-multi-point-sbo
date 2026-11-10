import pytest

import sbo.constants
from sbo.benchmarks import ALL_BENCHMARKS
from sbo.constants import BENCHMARK_FUNCTION_DELAY, function_delays, known_optima


@pytest.mark.unit
def test_mf2_known_optima_is_deprecated_alias() -> None:
    """Test that MF2_KNOWN_OPTIMA warns and gives f_opt of every benchmark problem."""
    with pytest.warns(DeprecationWarning, match="f_opt"):
        known = sbo.constants.MF2_KNOWN_OPTIMA

    assert known == {benchmark.name: benchmark.f_opt for benchmark in ALL_BENCHMARKS}


@pytest.mark.unit
def test_mf2_function_delay_is_deprecated_alias() -> None:
    """Test that MF2_FUNCTION_DELAY warns and gives BENCHMARK_FUNCTION_DELAY."""
    with pytest.warns(DeprecationWarning, match="BENCHMARK_FUNCTION_DELAY"):
        delays = sbo.constants.MF2_FUNCTION_DELAY

    assert delays is BENCHMARK_FUNCTION_DELAY


@pytest.mark.unit
def test_deprecated_constants_from_submodules() -> None:
    """Test that the deprecated names also warn when taken from the submodules."""
    with pytest.warns(DeprecationWarning):
        assert known_optima.MF2_KNOWN_OPTIMA
    with pytest.warns(DeprecationWarning):
        assert function_delays.MF2_FUNCTION_DELAY is BENCHMARK_FUNCTION_DELAY


@pytest.mark.unit
def test_deprecation_warning_points_at_caller() -> None:
    """Test that the warning is attributed to the code that used the name."""
    with pytest.warns(DeprecationWarning) as record:
        sbo.constants.MF2_KNOWN_OPTIMA  # noqa: B018

    assert record[0].filename == __file__


@pytest.mark.unit
def test_function_delay_for_every_benchmark() -> None:
    """Test that every benchmark problem has a function delay, keyed by its name."""
    assert set(BENCHMARK_FUNCTION_DELAY) == {b.name for b in ALL_BENCHMARKS}


@pytest.mark.unit
def test_unknown_constant_gives_attribute_error() -> None:
    """Test that other unknown names still give an AttributeError."""
    with pytest.raises(AttributeError):
        sbo.constants.NO_SUCH_CONSTANT  # noqa: B018
