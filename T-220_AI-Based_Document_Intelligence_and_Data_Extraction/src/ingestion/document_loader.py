"""Unified DocumentLoader implementing BaseIngestionEngine."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional, Union

from src.core.config import IngestionConfig
from src.core.exceptions import (
    IngestionError,
    UnsupportedFileTypeError,
)
from src.core.logging import get_logger
from src.core.models import Document, DocumentMetadata
from src.ingestion.base import BaseIngestionEngine
from src.ingestion.image_ingestion import ImageIngestionHandler
from src.ingestion.pdf_ingestion import PDFIngestionHandler
from src.ingestion.validators import (
    compute_sha256,
    sniff_file_format,
    validate_file_exists_and_readable,
    validate_file_not_empty,
    validate_file_size_limit,
    validate_image_decodable,
    validate_pdf_structure,
)

logger = get_logger("ingestion.loader")


class DocumentLoader(BaseIngestionEngine):
    """Production-grade document loader supporting PDF and raster image formats."""

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        processed_dir: Union[str, Path] = "data/processed",
    ) -> None:
        super().__init__(config=config)
        self.processed_dir = Path(processed_dir)
        self.image_handler = ImageIngestionHandler(config=self.config)
        self.pdf_handler = PDFIngestionHandler(config=self.config)

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate existence, readability, size, and header signature.

        Args:
            file_path: Path to target file.

        Returns:
            True if valid.

        Raises:
            FileNotFoundError, EmptyFileError, FileSizeExceededError, UnsupportedFileTypeError
        """
        path = validate_file_exists_and_readable(file_path)
        validate_file_not_empty(path)
        validate_file_size_limit(path, self.config.max_file_size_bytes)
        sniff_file_format(path)
        return True

    def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract high-level metadata without fully rendering all raster pages.

        Args:
            file_path: Path to target file.

        Returns:
            DocumentMetadata instance.
        """
        path = validate_file_exists_and_readable(file_path)
        validate_file_not_empty(path)
        validate_file_size_limit(path, self.config.max_file_size_bytes)
        mime_type, _ = sniff_file_format(path)
        sha256_hash = compute_sha256(path)
        file_size = path.stat().st_size
        doc_id = self._generate_document_id(path, sha256_hash)

        if mime_type == "application/pdf":
            page_count, _, _ = validate_pdf_structure(path)
            is_scanned = False
        else:
            _, _, _, _ = validate_image_decodable(path)
            page_count = 1
            is_scanned = True

        return DocumentMetadata(
            document_id=doc_id,
            filename=path.name,
            file_path=path,
            file_type=mime_type,
            file_size_bytes=file_size,
            checksum_sha256=sha256_hash,
            page_count=page_count,
            is_scanned=is_scanned,
        )

    def ingest(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Document:
        """Ingest a document file, safely decomposing and rendering all pages.

        Args:
            file_path: Path to target file (PDF, PNG, JPG, TIFF, BMP).
            document_id: Optional explicit document identifier.
            output_dir: Optional destination directory for staged page images.

        Returns:
            Fully populated Document aggregate with DocumentPage entries.
        """
        path = validate_file_exists_and_readable(file_path)
        validate_file_not_empty(path)
        validate_file_size_limit(path, self.config.max_file_size_bytes)
        mime_type, canonical_ext = sniff_file_format(path)

        sha256_hash = compute_sha256(path)
        doc_id = document_id or self._generate_document_id(path, sha256_hash)
        target_output_dir = Path(output_dir) if output_dir else self.processed_dir

        logger.info(
            f"Initiating ingestion for '{path.name}' [MIME: {mime_type}] (Doc ID: {doc_id})"
        )

        if mime_type == "application/pdf":
            return self.pdf_handler.ingest_pdf(
                file_path=path,
                document_id=doc_id,
                output_dir=target_output_dir,
            )
        elif mime_type.startswith("image/"):
            return self.image_handler.ingest_image(
                file_path=path,
                document_id=doc_id,
                output_dir=target_output_dir,
            )
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported document format: '{mime_type}' for file '{path.name}'"
            )

    @staticmethod
    def _generate_document_id(file_path: Path, sha256_hash: str) -> str:
        """Generate a deterministic, collision-resistant document identifier."""
        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)[:20].strip("_")
        short_hash = sha256_hash[:8]
        timestamp = int(time.time())
        return f"doc_{clean_stem}_{short_hash}_{timestamp}"
