"""Structured export contracts for JSON, CSV, and validation reports."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.core.config import StorageConfig
from src.core.models import ProcessingResult


class BaseExporter(ABC):
    """Abstract interface for exporting structured results to persistent storage formats."""

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        self.config = config or StorageConfig()

    @abstractmethod
    def export_json(
        self,
        result: ProcessingResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export full processing result to a formatted JSON document.

        Args:
            result: ProcessingResult instance.
            output_path: Destination path override.

        Returns:
            Path to the written JSON file.
        """
        pass

    @abstractmethod
    def export_csv(
        self,
        result: ProcessingResult,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """Export all extracted tables to individual structured CSV files.

        Args:
            result: ProcessingResult instance.
            output_dir: Target output directory for CSV files.

        Returns:
            List of Paths to all written CSV files.
        """
        pass

    @abstractmethod
    def export_validation_report(
        self,
        result: ProcessingResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export a comprehensive human-readable validation report (Markdown / Text).

        Args:
            result: ProcessingResult instance.
            output_path: Destination path override.

        Returns:
            Path to the written report file.
        """
        pass
