import logging
from pathlib import Path


def _make_logger() -> logging.Logger:
    """Helper to create a test logger.

    Returns
    -------
    The test logger.
    """
    # Use the test session logger configured in conftest.py
    return logging.getLogger("test_logger")


def _find_log_file(save_dir: Path, prefix: str) -> list[Path]:
    """Helper function to find log files in a directory with a given prefix.

    Parameters
    ----------
    save_dir
        Directory to find log files.
    prefix
        Prefix for log files.

    Returns
    -------
    The list of paths to the log files found.
    """
    files = [p for p in save_dir.iterdir() if p.is_file() and p.name.startswith(prefix)]
    return files
