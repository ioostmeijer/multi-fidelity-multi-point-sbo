import logging
from pathlib import Path
from typing import Generator

import pytest
from logger_helper import _find_log_file

from sbo.utils.Logger import get_logger, setup_logger


@pytest.mark.unit
def test_log_file_created(configure_test_logger: None) -> None:
    """Test that a log file is created when to_file=True."""
    save_dir = Path(configure_test_logger)
    name = "my_test_logger"

    logger = setup_logger(
        name=name,
        level=logging.INFO,
        save_dir=str(save_dir),
        to_file=True,
        overwrite=True,
        force=True,
    )
    logger.info("file creation test")

    # flush handler(s) to ensure file writes are persisted
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    files = _find_log_file(save_dir, f"{name}_")
    assert len(files) == 1
    assert files[0].suffix == ".log"


@pytest.mark.unit
def test_log_message_format_includes_level_and_name(
    configure_test_logger: None,
) -> None:
    """Test that log messages include the logger name and level tag."""
    save_dir = Path(configure_test_logger)
    name = "format_logger"

    logger = setup_logger(
        name=name,
        level=logging.INFO,
        save_dir=str(save_dir),
        to_file=True,
        overwrite=True,
        force=True,
    )
    logger.info("hello world")

    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    files = _find_log_file(save_dir, f"{name}_")
    assert files, "No log file found"
    content = files[0].read_text(encoding="utf-8")
    # Expect the logger name and level tag in the logged line
    assert f" - {name} - [INFO] - hello world" in content


@pytest.mark.unit
def test_overwrite_truncates_previous_content(
    monkeypatch: Generator, configure_test_logger: None
) -> None:
    """Test that setting overwrite=True truncates previous log file content."""
    save_dir = Path(configure_test_logger)
    name = "ovr_logger"

    # make filename deterministic by setting SLURM_JOB_ID
    monkeypatch.setenv("SLURM_JOB_ID", "42")

    # first create with append mode (overwrite=False)
    logger1 = setup_logger(
        name=name,
        level=logging.INFO,
        save_dir=str(save_dir),
        to_file=True,
        overwrite=False,
        force=True,
    )
    logger1.info("first message")
    for h in logger1.handlers:
        try:
            h.flush()
        except Exception:
            pass

    # now re-create with overwrite=True which should truncate the same file
    logger2 = setup_logger(
        name=name,
        level=logging.INFO,
        save_dir=str(save_dir),
        to_file=True,
        overwrite=True,
        force=True,
    )
    logger2.info("second message")
    for h in logger2.handlers:
        try:
            h.flush()
        except Exception:
            pass

    files = _find_log_file(save_dir, f"{name}_job_42")
    assert files
    content = files[0].read_text(encoding="utf-8")
    assert "first message" not in content
    assert "second message" in content


@pytest.mark.unit
def test_stream_handler_writes_to_stdout_and_no_file_created(
    capfd: Generator, configure_test_logger: None
) -> None:
    """Test that a stream handler writes to stdout and no file is created."""
    save_dir = Path(configure_test_logger)
    name = "stream_logger"

    logger = setup_logger(
        name=name,
        level=logging.INFO,
        save_dir=str(save_dir),
        to_file=False,
        overwrite=True,
        force=True,
    )
    logger.info("streamed message")

    # flush handlers to ensure stdout content is emitted
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    captured = capfd.readouterr()
    assert "streamed message" in captured.out

    # ensure no file with this logger prefix was created
    files = _find_log_file(save_dir, f"{name}_")
    assert not files


@pytest.mark.unit
def test_get_logger_returns_existing_logger(configure_test_logger: None) -> None:
    """Test that get_logger returns the same logger instance if already set up."""
    save_dir = Path(configure_test_logger)
    name = "existing_logger"

    logger1 = setup_logger(
        name=name,
        level=logging.INFO,
        save_dir=str(save_dir),
        to_file=True,
        overwrite=True,
        force=True,
    )

    logger2 = get_logger(name=name)

    assert logger1 is logger2


@pytest.mark.unit
def test_get_logger_creates_new_logger_if_not_exists(
    configure_test_logger: None,
) -> None:
    """Test that get_logger creates a new logger if it does not exist."""
    name = "new_logger"

    logger = get_logger(name=name)

    assert logger.name == name
    assert isinstance(logger, logging.Logger)
