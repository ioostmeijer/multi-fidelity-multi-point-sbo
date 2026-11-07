import pytest
from dummy_infill import DummyInfill
from dummy_surrogate import DummySurrogate

from sbo.infill import InfillBase
from sbo.optimisation import EgoAsync, EgoSequential
from sbo.surrogate import SurrogateBase
from sbo.utils import Config


def dummy_config() -> Config:
    """Create a dummy Config object for testing.

    Returns
    -------
    A dummy Config object with predefined settings for testing purposes.
    """
    return Config(
        used_algorithms={"optimisation": "EgoAsync"},
        functions=[lambda x: x**2, lambda x: (x - 2) ** 2],
        lb=[-10],
        ub=[10],
    )


def _import_mpi() -> None:
    """Helper function to import MPI for testing purposes."""
    global MPI
    global mock_mpiexec
    try:
        from mockmpi import mock_mpiexec
        from mpi4py import MPI
    except (ImportError, RuntimeError):
        pytest.skip("MPI not available", allow_module_level=True)
        MPI = None


@pytest.mark.unit
def test_no_mpi_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that EgoAsync falls back to EgoSequential when MPI is not available."""
    monkeypatch.setitem(EgoAsync.__new__.__globals__, "MPI", None)

    ego_async_instance = EgoAsync(config=dummy_config(), mpi_comm=None)

    assert isinstance(ego_async_instance, EgoSequential)


def _test_single_mpi_process(comm: "MPI.Comm") -> None:
    """Helper function to test single MPI process scenario."""
    instance = EgoAsync(config=dummy_config(), mpi_comm=comm)
    assert isinstance(instance, EgoSequential)


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_single_mpi_process() -> None:
    """Test that EgoAsync falls back to EgoSequential when MPI size < 2."""
    _import_mpi()
    mock_mpiexec(1, _test_single_mpi_process)


def _test_init(comm: "MPI.Comm") -> EgoAsync:
    """Helper function to test EgoAsync initialisation with multiple MPI processes."""
    instance = EgoAsync(config=dummy_config(), mpi_comm=comm)
    assert isinstance(instance, EgoAsync)
    assert instance.mpi_size == 4
    assert instance.config == dummy_config()
    assert instance.mpi_rank in range(4)


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_init_multiple_mpi_processes() -> None:
    """Test that EgoAsync initialises correctly with multiple MPI processes."""
    _import_mpi()
    mock_mpiexec(4, _test_init)


def _test_coordinator(comm: "MPI.Comm") -> None:
    """Check that the coordinator process calls the coordinator method."""
    config = dummy_config()
    instance = EgoAsync(config=config, mpi_comm=comm)
    if instance.mpi_rank == 0:
        # Check if instance._solve_coordinator() is called from instance.solve()
        def _solve_coordinator_mock(
            surrogate: SurrogateBase, infill: InfillBase, *args, **kwargs
        ) -> dict:
            return {"status": "coordinator_called"}

        instance._solve_coordinator = _solve_coordinator_mock  # type: ignore
        result = instance.solve(
            DummySurrogate(config=config), DummyInfill(config=config)
        )
        assert result == {"status": "coordinator_called"}


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_coordinator_is_called() -> None:
    """Test that the coordinator process runs the coordinator method."""
    _import_mpi()
    mock_mpiexec(4, _test_coordinator)


def _test_worker(comm: "MPI.Comm") -> None:
    config = dummy_config()
    instance = EgoAsync(config=config, mpi_comm=comm)
    if instance.mpi_rank != 0:
        # Check if instance._solve_worker() is called from instance.solve()
        def _solve_worker_mock() -> dict:
            return {"status": "worker_called"}

        instance._solve_worker = _solve_worker_mock
        result = instance.solve(
            DummySurrogate(config=config), DummyInfill(config=config)
        )
        assert result == {"status": "worker_called"}


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_worker_is_called() -> None:
    """Test that the worker processes run the worker method."""
    _import_mpi()
    mock_mpiexec(4, _test_worker)
