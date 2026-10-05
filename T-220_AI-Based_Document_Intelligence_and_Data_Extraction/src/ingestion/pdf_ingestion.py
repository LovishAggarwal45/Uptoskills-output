"""PDF ingestion handler for single and multi-page documents using pypdf and pdf2image."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)

from src.core.config import IngestionConfig
from src.core.exceptions import (
    CorruptedDocumentError,
    PDFRenderingError,
    PopplerNotFoundError,
)
from src.core.logging import get_logger
from src.core.models import Document, DocumentMetadata, DocumentPage
from src.core.types import DocumentType
from src.ingestion.validators import (
    compute_sha256,
    detect_poppler_path,
    sniff_file_format,
    validate_pdf_structure,
)

logger = get_logger("ingestion.pdf")


class PDFIngestionHandler:
    """Handles PDF parsing, page decomposition, and high-fidelity raster rendering."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        self.config = config or IngestionConfig()

    def check_poppler_availability(self) -> Tuple[bool, Optional[str]]:
        """Verify whether Poppler utilities are accessible for PDF rendering.

        Returns:
            Tuple of (is_available, resolved_poppler_path).
        """
        poppler_path = detect_poppler_path(self.config.poppler_path)
        return poppler_path is not None, poppler_path

    def ingest_pdf(
        self,
        file_path: Path,
        document_id: str,
        output_dir: Path,
    ) -> Document:
        """Parse PDF structure, rasterize pages at configured DPI, and construct Document aggregate.

        Args:
            file_path: Validated path to the input PDF file.
            document_id: Unique identifier for the processing session.
            output_dir: Destination directory where rendered page images will be saved.

        Returns:
            Document aggregate with all pages rendered and preserved in order.

        Raises:
            PopplerNotFoundError: If Poppler is required but unavailable on the host system.
            PDFRenderingError: If page rasterization encounters rendering errors.
            CorruptedDocumentError: If PDF structural inspection fails.
        """
        mime_type, _ = sniff_file_format(file_path)
        page_count, is_encrypted, pdf_meta = validate_pdf_structure(file_path)
        sha256_hash = compute_sha256(file_path)
        file_size = file_path.stat().st_size

        if is_encrypted:
            raise CorruptedDocumentError(
                f"PDF file '{file_path.name}' is password-protected or encrypted. Decrypt before ingestion.",
                details={"file_path": str(file_path), "is_encrypted": True},
            )

        # Check Poppler availability for rasterization
        is_poppler_ok, poppler_dir = self.check_poppler_availability()
        if not is_poppler_ok:
            raise PopplerNotFoundError(
                f"Cannot rasterize PDF '{file_path.name}': Poppler binaries (pdftoppm) were not found. "
                "On Windows, download Poppler binaries and set 'poppler_path' in configs/default.yaml "
                "or add Poppler's 'bin' directory to system PATH.",
                details={
                    "file_path": str(file_path),
                    "configured_poppler_path": self.config.poppler_path,
                },
            )

        doc_output_dir = output_dir / document_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Rasterizing {page_count}-page PDF '{file_path.name}' at {self.config.render_dpi} DPI..."
        )

        try:
            rendered_images = convert_from_path(
                pdf_path=str(file_path),
                dpi=self.config.render_dpi,
                poppler_path=poppler_dir,
                timeout=self.config.pdf_rendering_timeout_seconds,
            )
        except (PDFInfoNotInstalledError, FileNotFoundError) as e:
            raise PopplerNotFoundError(
                f"Poppler execution failed: {e}. Ensure Poppler binaries are installed and accessible.",
                details={"file_path": str(file_path), "poppler_path": poppler_dir},
            ) from e
        except (PDFPageCountError, PDFSyntaxError, Exception) as e:
            raise PDFRenderingError(
                f"Failed to render pages for PDF '{file_path.name}': {e}",
                details={"file_path": str(file_path), "original_error": str(e)},
            ) from e

        if len(rendered_images) != page_count:
            logger.warning(
                f"Page count mismatch for '{file_path.name}': expected {page_count}, rendered {len(rendered_images)}"
            )

        # Extract native digital text layers per page if available
        digital_texts: List[str] = []
        try:
            reader = PdfReader(str(file_path))
            for page_obj in reader.pages:
                try:
                    txt = page_obj.extract_text() or ""
                except Exception:
                    txt = ""
                digital_texts.append(txt)
        except Exception as e:
            logger.debug(f"Could not extract digital PDF text for '{file_path.name}': {e}")

        pages: List[DocumentPage] = []
        for idx, img in enumerate(rendered_images, start=1):
            page_img_path = doc_output_dir / f"page_{idx:03d}_original.png"
            img.save(page_img_path, format="PNG")

            dig_txt = digital_texts[idx - 1] if idx - 1 < len(digital_texts) else ""
            width, height = img.size
            pages.append(
                DocumentPage(
                    page_number=idx,
                    raw_text=dig_txt if dig_txt.strip() else None,
                    width=float(width),
                    height=float(height),
                    dpi=self.config.render_dpi,
                    image_path=page_img_path,
                    metadata={
                        "source_format": ".pdf",
                        "color_mode": img.mode,
                        "has_digital_text": bool(dig_txt.strip()),
                        "digital_text_length": len(dig_txt),
                        "ingestion_status": "success",
                    },
                )
            )

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            file_path=file_path,
            file_type=mime_type,
            file_size_bytes=file_size,
            checksum_sha256=sha256_hash,
            page_count=len(pages),
            is_scanned=False,  # Can contain native vector and text layers
        )

        logger.info(
            f"Successfully ingested {len(pages)} pages from PDF '{file_path.name}' (ID: {document_id})"
        )

        return Document(
            metadata=metadata,
            pages=pages,
            classified_type=DocumentType.UNKNOWN,
        )
