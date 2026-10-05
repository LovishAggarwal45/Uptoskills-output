"""Core module: data models, configuration, logging, exceptions, and base types."""
from src.core.types import (
    DocumentType,
    FieldType,
    SeverityLevel,
    ValidationStatus,
    ConfidenceSource,
    ExtractionMethod,
    ReviewTriggerType,
)
from src.core.models import (
    BoundingBox,
    Provenance,
    OCRTextRegion,
    DocumentMetadata,
    DocumentPage,
    Document,
    ExtractedField,
    TableCell,
    ExtractedTable,
    ValidationResult,
    ReviewFlag,
    ProcessingSummary,
    ProcessingResult,
)
from src.core.exceptions import DocuMindError
from src.core.config import DocuMindConfig, load_config
from src.core.logging import get_logger, configure_logging

__all__ = [
    "DocumentType",
    "FieldType",
    "SeverityLevel",
    "ValidationStatus",
    "ConfidenceSource",
    "ExtractionMethod",
    "ReviewTriggerType",
    "BoundingBox",
    "Provenance",
    "OCRTextRegion",
    "DocumentMetadata",
    "DocumentPage",
    "Document",
    "ExtractedField",
    "TableCell",
    "ExtractedTable",
    "ValidationResult",
    "ReviewFlag",
    "ProcessingSummary",
    "ProcessingResult",
    "DocuMindError",
    "DocuMindConfig",
    "load_config",
    "get_logger",
    "configure_logging",
]
