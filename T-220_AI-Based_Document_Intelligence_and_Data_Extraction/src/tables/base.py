"""Abstract contracts and interfaces for DocuMind AI Table Extraction Subsystem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.config import DocuMindConfig
from src.core.models import Document, DocumentPage
from src.ocr.models import OCRLine, OCRWord, PageOCRResult
from src.tables.models import (
    LineItem,
    Table,
    TableExtractionResult,
    TableRegion,
    TableValidationResult,
)


class BaseTableDetector(ABC):
    """Abstract interface for detecting 2D tabular regions on document pages."""

    @abstractmethod
    def detect_regions(
        self,
        words: List[OCRWord],
        lines: Optional[List[OCRLine]] = None,
        page_number: int = 1,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> List[TableRegion]:
        """Detect candidate tabular bounding regions on a single page."""
        pass


class BaseTableReconstructor(ABC):
    """Abstract interface for reconstructing rows, columns, and cells within a table region."""

    @abstractmethod
    def reconstruct_table(
        self,
        region: TableRegion,
        words: List[OCRWord],
        table_id: str,
        page_number: int = 1,
    ) -> Table:
        """Reconstruct 2D grid structure from detected region and word tokens."""
        pass


class BaseLineItemExtractor(ABC):
    """Abstract interface for extracting canonical semantic line items from reconstructed tables."""

    @abstractmethod
    def extract_line_items(
        self,
        table: Table,
        document_id: str = "doc",
    ) -> List[LineItem]:
        """Extract typed line items from table grid rows."""
        pass


class BaseTableValidator(ABC):
    """Abstract interface for mathematical, horizontal, and vertical integrity validation."""

    @abstractmethod
    def validate_table(
        self,
        table: Table,
        document_totals: Optional[Dict[str, float]] = None,
    ) -> TableValidationResult:
        """Perform arithmetic validation across row calculations and table totals."""
        pass


class BaseTableContinuationDetector(ABC):
    """Abstract interface for identifying and linking multi-page table continuations."""

    @abstractmethod
    def link_continuations(
        self,
        tables: List[Table],
    ) -> List[Table]:
        """Identify multi-page continuous tables and link/merge them."""
        pass


class BaseTableConfidenceScorer(ABC):
    """Abstract interface for calculating explainable composite table confidence scores."""

    @abstractmethod
    def calculate_confidence(
        self,
        table: Table,
    ) -> float:
        """Compute composite confidence score and breakdown for a reconstructed table."""
        pass


class BaseTableExtractor(ABC):
    """Master abstract coordinator for end-to-end table extraction across documents."""

    @abstractmethod
    def extract(self, document: Document) -> TableExtractionResult:
        """Extract all tables and line items from an ingested multi-page document."""
        pass


__all__ = [
    "BaseTableDetector",
    "BaseTableReconstructor",
    "BaseLineItemExtractor",
    "BaseTableValidator",
    "BaseTableContinuationDetector",
    "BaseTableConfidenceScorer",
    "BaseTableExtractor",
]
