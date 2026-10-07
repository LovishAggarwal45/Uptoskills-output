"""Abstract base interfaces for document, field, and entity extraction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.classification.models import ClassificationResult
from src.core.config import ExtractionConfig
from src.core.models import Document, ExtractedField
from src.core.types import DocumentType
from src.extraction.models import ExtractedEntity, ExtractionResult


class BaseDocumentExtractor(ABC):
    """Abstract interface for high-level document intelligence extraction."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    @abstractmethod
    def extract(
        self,
        document: Document,
        classification_result: Optional[ClassificationResult] = None,
    ) -> ExtractionResult:
        """Extract structured domain fields and generic entities from document.

        Args:
            document: Ingested document aggregate populated with pages and OCR results.
            classification_result: Optional document classification output to drive specialized extractors.

        Returns:
            ExtractionResult containing extracted fields, entities, statistics, and provenance.
        """
        pass


class BaseFieldExtractor(ABC):
    """Abstract interface for extracting key-value pairs, dates, amounts, and structured fields."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    @abstractmethod
    def extract_fields(
        self,
        document: Document,
        doc_type: Optional[DocumentType] = None,
    ) -> Dict[str, ExtractedField]:
        """Extract structured fields from document pages with source provenance."""
        pass


class BaseEntityExtractor(ABC):
    """Abstract interface for extracting generic typed entities (dates, money, emails, phones)."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        self.config = config or ExtractionConfig()

    @abstractmethod
    def extract_entities(self, document: Document) -> List[ExtractedEntity]:
        """Extract generic typed entities across all document pages."""
        pass


__all__ = [
    "BaseDocumentExtractor",
    "BaseFieldExtractor",
    "BaseEntityExtractor",
    "ExtractionResult",
]
