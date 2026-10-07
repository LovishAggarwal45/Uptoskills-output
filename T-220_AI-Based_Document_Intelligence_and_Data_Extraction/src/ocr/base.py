"""Optical Character Recognition (OCR) abstraction interface and base engine contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src.core.config import OCRConfig
from src.core.models import Document, DocumentPage, OCRTextRegion
from src.ocr.availability import OCRAvailabilityResult, check_ocr_availability
from src.ocr.models import DocumentOCRResult, PageOCRResult


class BaseOCREngine(ABC):
    """Pluggable OCR engine contract supporting interchangeable OCR backends (Tesseract, etc.)."""

    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        self.config = config or OCRConfig()

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Unique identifier of the OCR backend (e.g. 'tesseract', 'easyocr')."""
        pass

    def is_available(self) -> bool:
        """Check whether the OCR engine and its runtime dependencies are available."""
        return self.get_availability().is_ready

    def get_availability(self) -> OCRAvailabilityResult:
        """Perform comprehensive environment check for this OCR engine."""
        return check_ocr_availability(self.config)

    @abstractmethod
    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        page_number: int = 1,
    ) -> PageOCRResult:
        """Execute OCR on a raw image buffer, array, or file path.

        Args:
            image_input: Path, numpy array, or PIL Image.
            page_number: 1-indexed page number for coordinate and provenance mapping.

        Returns:
            PageOCRResult with structured word, line, and block regions.
        """
        pass

    def process_page(
        self,
        page: DocumentPage,
        image_path: Optional[Union[str, Path]] = None,
    ) -> PageOCRResult:
        """Execute OCR on a DocumentPage instance.

        Args:
            page: DocumentPage instance.
            image_path: Optional path override (e.g. to preprocessed image).

        Returns:
            PageOCRResult populated with text and regions.
        """
        target_path = image_path or page.image_path
        if not target_path:
            raise ValueError(f"DocumentPage {page.page_number} has no image_path to perform OCR.")

        return self.process_image(
            image_input=target_path,
            page_number=page.page_number,
        )

    def process_document(self, document: Document) -> DocumentOCRResult:
        """Execute OCR across all pages of a Document aggregate and update page contents.

        Args:
            document: Document aggregate with initialized pages.

        Returns:
            DocumentOCRResult aggregating all page-level OCR results.
        """
        page_results: List[PageOCRResult] = []
        total_duration = 0.0

        for page in sorted(document.pages, key=lambda p: p.page_number):
            res = self.process_page(page)
            page_results.append(res)
            total_duration += res.execution_time_seconds

            # Populate document page attributes directly
            page.raw_text = res.raw_text
            page.ocr_text_regions = res.text_regions

        full_doc_text = "\n\n--- Page Break ---\n\n".join(
            p.raw_text for p in page_results if p.raw_text
        )

        all_words = [w for p in page_results for w in p.words]
        all_lines = [l for p in page_results for l in p.lines]
        all_blocks = [b for p in page_results for b in p.blocks]

        mean_conf = (
            sum(w.confidence for w in all_words) / len(all_words)
            if all_words
            else None
        )

        return DocumentOCRResult(
            document_id=document.id,
            pages=page_results,
            full_text=full_doc_text,
            total_words=len(all_words),
            total_lines=len(all_lines),
            total_blocks=len(all_blocks),
            mean_confidence=mean_conf,
            engine_name=self.engine_name,
            total_duration_seconds=round(total_duration, 4),
        )
