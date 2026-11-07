import pytest
from dummy_infill import DummyInfill
from dummy_surrogate import DummySurrogate

from sbo.optimisation import EgoMPI, EgoSequential
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
    """Test that EgoMPI falls back to EgoSequential when MPI is not available."""
    monkeypatch.setitem(EgoMPI.__new__.__globals__, "MPI", None)

    ego_async_instance = EgoMPI(config=dummy_config(), mpi_comm=None)

    assert isinstance(ego_async_instance, EgoSequential)


def _test_single_mpi_process(comm: "MPI.Comm") -> None:
    instance = EgoMPI(config=dummy_config(), mpi_comm=comm)
    assert isinstance(instance, EgoSequential)


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_single_mpi_process() -> None:
    """Test that EgoMPI falls back to EgoSequential when MPI size < 2."""
    _import_mpi()
    mock_mpiexec(1, _test_single_mpi_process)


def _test_init(comm: "MPI.Comm") -> None:
    instance = EgoMPI(config=dummy_config(), mpi_comm=comm)
    assert isinstance(instance, EgoMPI)
    assert instance.mpi_size == 4
    assert instance.config == dummy_config()
    assert instance.mpi_rank in range(4)


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_init_multiple_mpi_processes() -> None:
    """Test that EgoMPI initialises correctly with multiple MPI processes."""
    _import_mpi()
    mock_mpiexec(4, _test_init)


def _test_coordinator(comm: "MPI.Comm") -> None:
    config = dummy_config()
    instance = EgoMPI(config=config, mpi_comm=comm)
    if instance.mpi_rank == 0:

        def _solve_coordinator_mock(
            surrogate: DummySurrogate, infill: DummyInfill
        ) -> dict:
            return {"status": "coordinator_called"}

        instance._solve_coordinator = _solve_coordinator_mock
        result = instance.solve(
            DummySurrogate(config=config), DummyInfill(config=config)
        )
        assert result == {"status": "coordinator_called"}


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_coordinator_process() -> None:
    """Test the coordinator process functionality of EgoMPI."""
    _import_mpi()
    mock_mpiexec(4, _test_coordinator)


def _test_worker(comm: "MPI.Comm") -> None:
    config = dummy_config()
    instance = EgoMPI(config=config, mpi_comm=comm)
    if instance.mpi_rank != 0:

        def _solve_worker_mock() -> dict:
            return {"status": "worker_called"}

        instance._solve_worker = _solve_worker_mock
        result = instance.solve(
            DummySurrogate(config=config), DummyInfill(config=config)
        )
        assert result == {"status": "worker_called"}


@pytest.mark.mock_mpi
@pytest.mark.unit
def test_worker_process() -> None:
    """Test the worker process functionality of EgoMPI."""
    _import_mpi()
    mock_mpiexec(4, _test_worker)
