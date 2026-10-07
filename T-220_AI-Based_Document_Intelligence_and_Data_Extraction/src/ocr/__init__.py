"""Optical Character Recognition (OCR) subsystem for DocuMind AI."""

from src.ocr.availability import OCRAvailabilityResult, check_ocr_availability, find_tesseract_binary
from src.ocr.base import BaseOCREngine
from src.ocr.engine_factory import OCREngineFactory
from src.ocr.exceptions import (
    OCRConfigurationError,
    OCREngineUnavailableError,
    OCRException,
    OCRInvalidImageError,
    OCRLanguageUnavailableError,
    OCRProcessingError,
)
from src.ocr.models import (
    DocumentOCRResult,
    OCRBlock,
    OCRLevel,
    OCRLine,
    OCRWord,
    PageOCRResult,
)
from src.ocr.postprocessing import (
    compute_bounding_box_union,
    normalize_tesseract_confidence,
    reconstruct_page_text,
    sort_reading_order,
)
from src.ocr.processor import DocumentOCRProcessor
from src.ocr.tesseract_engine import TesseractOCREngine

__all__ = [
    "BaseOCREngine",
    "TesseractOCREngine",
    "OCREngineFactory",
    "DocumentOCRProcessor",
    "OCRAvailabilityResult",
    "check_ocr_availability",
    "find_tesseract_binary",
    "OCRLevel",
    "OCRWord",
    "OCRLine",
    "OCRBlock",
    "PageOCRResult",
    "DocumentOCRResult",
    "normalize_tesseract_confidence",
    "compute_bounding_box_union",
    "sort_reading_order",
    "reconstruct_page_text",
    "OCRException",
    "OCREngineUnavailableError",
    "OCRConfigurationError",
    "OCRProcessingError",
    "OCRLanguageUnavailableError",
    "OCRInvalidImageError",
]
