import logging
import os
import sys
import traceback
from datetime import datetime

try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    MPI = None


def setup_logger(
    name: str = "default_logger",
    level: int = logging.INFO,
    save_dir: str = "logs",
    to_file: bool = True,
    overwrite: bool = False,
    force: bool = True,
) -> logging.Logger:
    """Set up a logger with the specified name and level.

    Parameters
    ----------
    name
        Name of the logger.
    level
        Logging level.
    save_dir
        directory to save log files if to_file is True.
    to_file
        Whether to log to a file or to stdout.
    overwrite
        Whether to overwrite existing log files.
    force
        Whether to force reconfiguration of the logger.

    Returns
    -------
    logging.Logger
        The logger, with one handler writing to a file in save_dir or to stdout.
    """
    logger = logging.getLogger(name)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if logger.hasHandlers() and not force:
        return logger  # Logger already set up

    rank = 0
    size = 1
    if MPI is not None and MPI.Is_initialized():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    logger.setLevel(level)

    if to_file:
        os.makedirs(save_dir, exist_ok=True)
        slurm_job_id = os.getenv("SLURM_JOB_ID", None)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if size > 1 and slurm_job_id is None and "rank" not in name:
            log_name = f"{name}_rank_{rank}.log"
        elif slurm_job_id is not None:
            if "rank" not in name and size > 1:
                log_name = f"{name}_job_{slurm_job_id}_rank_{rank}.log"
            else:
                log_name = f"{name}_job_{slurm_job_id}.log"
        else:
            log_name = f"{name}_{timestamp}.log"

        log_file = os.path.join(save_dir, log_name)
        handler = logging.FileHandler(log_file, mode="w" if overwrite else "a")
    else:
        handler = logging.StreamHandler(sys.stdout)

    if size > 1:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(relativepath)-35s - %(name)s - [Rank %(rank)d] - "
            "[%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(relativepath)-35s - %(name)s - [%(levelname)s] - "
            "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _attach_record_attrs(record: logging.LogRecord) -> bool:
        """Attach additional attributes to the log record.

        Parameters
        ----------
        record
            Log record.

        Returns
        -------
        bool
            True if attributes were attached successfully.
        """
        record.rank = rank
        try:
            record.relativepath = os.path.relpath(record.pathname)
        except Exception:
            record.relativepath = record.pathname
        return True

    handler.addFilter(_attach_record_attrs)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    if rank == 0:
        logger.info(f"Logger initialised (MPI size={size}, to_file={to_file})")

    return logger


def get_logger(name: str = "default_logger", **kwargs) -> logging.Logger:
    """Retrieve an existing logger by name.

    Parameters
    ----------
    name
        Name of the logger.
    **kwargs
        Keyword arguments to pass to the logger, if not logger exists with name.

    Returns
    -------
    logging.Logger
        The existing logger with this name, or a new one set up with setup_logger (the
        default logger also prints warnings to stderr).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name=name, **kwargs)

        # The default logger writes to a file. Also show warnings and errors on
        # screen, so that e.g. invalid settings do not go unnoticed.
        if name == "default_logger" and kwargs.get("to_file", True):
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(
                logging.Formatter("%(name)s - [%(levelname)s] - %(message)s")
            )
            logger.addHandler(console_handler)
    return logger


def delete_logger(name: str = "default_logger") -> None:
    """Delete a logger by name.

    Parameters
    ----------
    name
        Name of the logger.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logging.Logger.manager.loggerDict.pop(name, None)


def log_stack_trace(logger: logging.Logger, level: int = logging.ERROR) -> None:
    """Log the current stack trace or the active exception stack trace.

    Parameters
    ----------
    logger
        Logger instance to use for logging.
    level
        Logging level.
    """
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_type is not None:
        # Log exception stack trace
        stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    else:
        # Log current stack trace
        stack_trace = "".join(traceback.format_stack())
    logger.log(level, "Stack Trace:\n%s", stack_trace)
