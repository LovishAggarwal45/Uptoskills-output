"""Ingestion contract and abstract base class for document intake."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from src.core.config import IngestionConfig
from src.core.models import Document, DocumentMetadata, DocumentPage


class BaseIngestionEngine(ABC):
    """Abstract interface for ingesting raw files (PDF, images) into Document aggregates."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        self.config = config or IngestionConfig()

    @abstractmethod
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Verify that the target file exists, is readable, and matches supported formats.

        Args:
            file_path: Path to the target document.

        Returns:
            True if valid, raises IngestionError/UnsupportedFileTypeError otherwise.
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract filesystem and format-level metadata without full OCR processing.

        Args:
            file_path: Path to the target document.

        Returns:
            DocumentMetadata instance.
        """
        pass

    @abstractmethod
    def ingest(self, file_path: Union[str, Path]) -> Document:
        """Ingest the target file, decompose into pages, and instantiate a Document aggregate.

        Args:
            file_path: Path to the target document.

        Returns:
            Document instance with initialized pages (raw images or PDF buffers).
        """
        pass
