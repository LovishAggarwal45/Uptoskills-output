"""Document ingestion subsystem for DocuMind AI."""

from src.ingestion.base import BaseIngestionEngine
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.image_ingestion import ImageIngestionHandler
from src.ingestion.pdf_ingestion import PDFIngestionHandler
from src.ingestion.validators import (
    compute_sha256,
    detect_poppler_path,
    sniff_file_format,
    validate_file_exists_and_readable,
    validate_file_not_empty,
    validate_file_size_limit,
    validate_image_decodable,
    validate_pdf_structure,
)

__all__ = [
    "BaseIngestionEngine",
    "DocumentLoader",
    "ImageIngestionHandler",
    "PDFIngestionHandler",
    "compute_sha256",
    "detect_poppler_path",
    "sniff_file_format",
    "validate_file_exists_and_readable",
    "validate_file_not_empty",
    "validate_file_size_limit",
    "validate_image_decodable",
    "validate_pdf_structure",
]
