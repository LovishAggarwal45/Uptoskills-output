"""Exception hierarchy for DocuMind AI."""

from typing import Any, Dict, Optional


class DocuMindError(Exception):
    """Base exception class for all DocuMind AI domain errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception details to a serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# Ingestion Errors
class IngestionError(DocuMindError):
    """Base error for document ingestion failures."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when an ingested file has an unsupported extension or MIME type."""
    pass


class EmptyFileError(IngestionError):
    """Raised when an ingested file contains zero bytes."""
    pass


class CorruptedDocumentError(IngestionError):
    """Raised when an ingested file cannot be parsed, read, or rendered."""
    pass


class ImageDecodingError(IngestionError):
    """Raised when an image file cannot be decoded by image processing backends."""
    pass


class PDFRenderingError(IngestionError):
    """Raised when a PDF page fails to rasterize into an image buffer."""
    pass


class PopplerNotFoundError(PDFRenderingError):
    """Raised when PDF rasterization requires Poppler utilities, but Poppler is not found in PATH or configured."""
    pass


class FileSizeExceededError(IngestionError):
    """Raised when a document exceeds the configured maximum allowable file size."""
    pass


# Preprocessing Errors
class PreprocessingError(DocuMindError):
    """Base error for image preprocessing failures."""
    pass


# OCR Errors
class OCRError(DocuMindError):
    """Base error for OCR engine processing failures."""
    pass


class OCREngineNotFoundError(OCRError):
    """Raised when the configured OCR backend executable or library is missing."""
    pass


# Classification Errors
class ClassificationError(DocuMindError):
    """Base error for document classification failures."""
    pass


class InvalidClassificationInput(ClassificationError):
    """Raised when the document provided for classification is invalid or empty."""
    pass


class ClassificationConfigurationError(ClassificationError):
    """Raised when classification rules or configurations are malformed or missing."""
    pass


# Extraction Errors
class ExtractionError(DocuMindError):
    """Base error for entity, key-value, or table extraction failures."""
    pass


class InvalidExtractionInput(ExtractionError):
    """Raised when the document or OCR payload provided for extraction is invalid."""
    pass


class ExtractionConfigurationError(ExtractionError):
    """Raised when extraction rules, patterns, or configurations are malformed."""
    pass


class AmbiguousExtractionError(ExtractionError):
    """Raised when conflicting candidate values cannot be deterministically resolved."""
    pass


class SchemaMismatchError(ExtractionError):
    """Raised when extracted data cannot conform to the target document schema."""
    pass


# Validation Errors
class ValidationError(DocuMindError):
    """Base error for validation engine execution failures."""
    pass


# Confidence & Scoring Errors
class ConfidenceCalculationError(DocuMindError):
    """Base error for confidence score synthesis failures."""
    pass


# Export Errors
class ExportError(DocuMindError):
    """Base error for JSON, CSV, or validation report export failures."""
    pass


# Pipeline Errors
class PipelineExecutionError(DocuMindError):
    """Base error for pipeline orchestration failures."""
    pass


class PipelineStepError(PipelineExecutionError):
    """Raised when an individual pipeline step fails during execution."""

    def __init__(
        self,
        step_name: str,
        message: str,
        original_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = details or {}
        merged_details["step_name"] = step_name
        if original_exception:
            merged_details["original_error"] = str(original_exception)
            merged_details["original_error_type"] = original_exception.__class__.__name__

        super().__init__(
            message=f"Pipeline step '{step_name}' failed: {message}",
            error_code="PIPELINE_STEP_ERROR",
            details=merged_details,
        )
        self.step_name = step_name
        self.original_exception = original_exception
