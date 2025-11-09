import sys
from pathlib import Path

from loguru import logger


def configure_logger(log_dir="logs"):
    """
    Configures the loguru logger to write to a file and to the console.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "experiment_{time}.log"

    logger.remove()  # Remove default handler
    # Log to console
    logger.add(sys.stderr, level="INFO")
    # Log to file
    logger.add(log_file_path, level="DEBUG", rotation="10 MB", compression="zip")

    logger.info("Logger configured.")