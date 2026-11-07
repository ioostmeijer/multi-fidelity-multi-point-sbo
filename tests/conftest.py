import logging
import pathlib
import shutil

import pytest

from sbo.utils.Config import Config
from sbo.utils.Logger import setup_logger

# Maps a path to the corresponding marker
_MARKER_BY_DIR = {
    "tests_unittest": "unit",
    "tests_verification": "verification",
    "tests_e2e": "e2e",
}


def pytest_collection_modifyitems(items: "list[pytest.Item]") -> None:
    """Tag each test with a marker based on dir."""
    for item in items:
        parts = pathlib.Path(str(item.fspath)).parts
        for directory, marker in _MARKER_BY_DIR.items():
            if directory in parts:
                item.add_marker(marker)
                break


@pytest.fixture(scope="session", autouse=True)
def configure_test_logger(tmp_path_factory: "pytest.TempPathFactory") -> None:
    """Fixture to configure a test logger for the entire test session.

    Parameters
    ----------
    tmp_path_factory
        A pytest fixture that provides a factory for temporary directories.
    """
    log_dir = tmp_path_factory.mktemp("test_logs")

    setup_logger(
        name="test_logger",
        level=logging.DEBUG,
        save_dir=str(log_dir),
        to_file=True,
        overwrite=True,
        force=True,
    )

    yield log_dir

    shutil.rmtree(log_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def run_in_temporary_directory(
    tmp_path_factory: "pytest.TempPathFactory", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Run each test in its own temporary working directory.

    Files a test writes to a relative path (e.g. the default ``logs/`` folder, plots
    or saved optimisation data) end up there instead of in the directory pytest was
    started from, so the tests leave no files behind.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture(autouse=True)
def reset_environment() -> None:
    """Reset the global state (Config and loggers) before and after each test."""
    Config.destroy_instance()

    logging.shutdown()
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).handlers.clear()

    yield

    Config.destroy_instance()
    logging.shutdown()
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).handlers.clear()
