"""Exceptions for DocuMind AI Table & Line-Item Extraction Subsystem."""

from typing import Any, Dict, Optional

from src.core.exceptions import (
    DocuMindError,
    ExtractionConfigurationError,
    ExtractionError,
    InvalidExtractionInput,
    ValidationError,
)


class TableExtractionError(ExtractionError):
    """Base exception for all table intelligence and line-item extraction failures."""
    pass


class InvalidTableInput(InvalidExtractionInput, TableExtractionError):
    """Raised when the document, page, or OCR payload provided for table extraction is invalid or missing."""
    pass


class TableStructureError(TableExtractionError):
    """Raised when table geometry, column boundaries, or 2D grid construction fails."""
    pass


class TableConfigurationError(ExtractionConfigurationError, TableExtractionError):
    """Raised when table extraction configuration or column mapping parameters are malformed."""
    pass


class TableValidationError(ValidationError, TableExtractionError):
    """Raised when table arithmetic or structural validation encounters fatal inconsistencies."""
    pass


__all__ = [
    "TableExtractionError",
    "InvalidTableInput",
    "TableStructureError",
    "TableConfigurationError",
    "TableValidationError",
]
