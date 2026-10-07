"""Structured logging framework for DocuMind AI."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class DocuMindLogFormatter(logging.Formatter):
    """Custom formatter with clean alignment and contextual metadata."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
        super().__init__(fmt=fmt or DEFAULT_LOG_FORMAT, datefmt=datefmt or DATE_FORMAT)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_to_console: bool = True,
) -> None:
    """Configure DocuMind logging handlers and formatters.

    Args:
        level: Root logging level for the documind package.
        log_file: Optional path to write persistent log output.
        log_to_console: Whether to emit logs to sys.stdout.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    documind_logger = logging.getLogger("documind")
    documind_logger.setLevel(level)

    # Clear existing handlers to prevent duplicates
    documind_logger.handlers.clear()

    formatter = DocuMindLogFormatter()

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        documind_logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        documind_logger.addHandler(file_handler)


def get_logger(module_name: str) -> logging.Logger:
    """Retrieve a child logger under the documind namespace.

    Args:
        module_name: Component or module name (e.g. 'ocr', 'validation').
    """
    if module_name.startswith("documind."):
        return logging.getLogger(module_name)
    return logging.getLogger(f"documind.{module_name}")
