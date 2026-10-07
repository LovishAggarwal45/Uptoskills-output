"""OCR subsystem domain exceptions."""

from typing import Any, Dict, Optional

from src.core.exceptions import OCRError


class OCRException(OCRError):
    """Base exception for all OCR subsystem errors."""
    pass


class OCREngineUnavailableError(OCRException):
    """Raised when the configured OCR backend executable or library is missing."""

    def __init__(
        self,
        engine_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = message or (
            f"OCR engine '{engine_name}' is unavailable. "
            "Ensure the OCR executable is installed and configured in system PATH or config."
        )
        merged_details = details or {}
        merged_details["engine_name"] = engine_name
        super().__init__(message=msg, error_code="OCR_ENGINE_UNAVAILABLE", details=merged_details)
        self.engine_name = engine_name


class OCRConfigurationError(OCRException):
    """Raised when OCR parameters (PSM, OEM, language) are invalid or misconfigured."""
    pass


class OCRProcessingError(OCRException):
    """Raised when optical character recognition fails during image processing."""

    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        page_number: Optional[int] = None,
        engine_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = details or {}
        if document_id:
            merged_details["document_id"] = document_id
        if page_number is not None:
            merged_details["page_number"] = page_number
        if engine_name:
            merged_details["engine_name"] = engine_name

        super().__init__(message=message, error_code="OCR_PROCESSING_ERROR", details=merged_details)
        self.document_id = document_id
        self.page_number = page_number
        self.engine_name = engine_name


class OCRLanguageUnavailableError(OCRException):
    """Raised when a requested OCR language traineddata pack is not installed in the engine."""

    def __init__(
        self,
        language: str,
        available_languages: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = (
            f"Requested OCR language '{language}' is not available in the OCR engine. "
            f"Available languages: {available_languages or []}"
        )
        merged_details = details or {}
        merged_details["requested_language"] = language
        merged_details["available_languages"] = available_languages or []
        super().__init__(message=msg, error_code="OCR_LANGUAGE_UNAVAILABLE", details=merged_details)
        self.language = language
        self.available_languages = available_languages or []


class OCRInvalidImageError(OCRException):
    """Raised when an image provided for OCR is empty, corrupted, or has invalid dimensions."""
    pass
