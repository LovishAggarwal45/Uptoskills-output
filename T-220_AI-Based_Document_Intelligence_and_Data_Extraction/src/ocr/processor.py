"""High-level document OCR processing coordinator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.core.config import DocuMindConfig, OCRConfig
from src.core.logging import get_logger
from src.core.models import Document, DocumentPage
from src.ocr.base import BaseOCREngine
from src.ocr.engine_factory import OCREngineFactory
from src.ocr.exceptions import OCRProcessingError
from src.ocr.models import DocumentOCRResult, PageOCRResult

logger = get_logger("ocr.processor")


class DocumentOCRProcessor:
    """Coordinates page-level OCR recognition and document-level text aggregation."""

    def __init__(
        self,
        engine: Optional[BaseOCREngine] = None,
        config: Optional[OCRConfig] = None,
        use_preprocessed_images: bool = True,
    ) -> None:
        self.config = config or OCRConfig()
        self.engine = engine or OCREngineFactory.create_engine(config=self.config)
        self.use_preprocessed_images = use_preprocessed_images

    def process_document(
        self,
        document: Document,
        processed_dir: Optional[Union[str, Path]] = None,
    ) -> DocumentOCRResult:
        """Run optical character recognition across all pages of a Document.

        Args:
            document: Ingested Document aggregate.
            processed_dir: Optional directory containing preprocessed page images.

        Returns:
            DocumentOCRResult containing all page-level recognitions and aggregated text.
        """
        logger.info(
            f"Starting document-level OCR for '{document.metadata.filename}' "
            f"({len(document.pages)} pages) using engine '{self.engine.engine_name}'"
        )

        start_time = time.perf_counter()
        page_results: List[PageOCRResult] = []
        page_errors: Dict[int, str] = {}

        for page in sorted(document.pages, key=lambda p: p.page_number):
            target_image = self._resolve_page_image(page, document.id, processed_dir)

            try:
                page_res = self.engine.process_image(
                    image_input=target_image,
                    page_number=page.page_number,
                )
                page_results.append(page_res)

                # Mutate DocumentPage fields in-place
                page.raw_text = page_res.raw_text
                page.ocr_text_regions = page_res.text_regions
                page.metadata["ocr_engine"] = self.engine.engine_name
                page.metadata["ocr_mean_confidence"] = page_res.mean_confidence
                page.metadata["ocr_words"] = page_res.words
                page.metadata["page_ocr_result"] = page_res

            except Exception as e:
                err_msg = f"OCR failed on page {page.page_number}: {e}"
                logger.error(err_msg, exc_info=True)
                page_errors[page.page_number] = str(e)
                page.metadata["ocr_error"] = str(e)

                # If processing error occurs, create empty result to preserve page mapping
                empty_res = PageOCRResult(
                    page_number=page.page_number,
                    raw_text="",
                    engine_name=self.engine.engine_name,
                    mean_confidence=0.0,
                    execution_time_seconds=0.0,
                    provenance_metadata={"error": str(e)},
                )
                page_results.append(empty_res)

        total_elapsed = time.perf_counter() - start_time

        all_words = [w for p in page_results for w in p.words]
        all_lines = [l for p in page_results for l in p.lines]
        all_blocks = [b for p in page_results for b in p.blocks]

        full_doc_text = "\n\n--- Page Break ---\n\n".join(
            p.raw_text for p in page_results if p.raw_text
        )

        mean_conf = (
            sum(w.confidence for w in all_words) / len(all_words)
            if all_words
            else None
        )

        logger.info(
            f"Finished document OCR for '{document.id}': {len(all_words)} words recognized "
            f"across {len(page_results)} pages in {total_elapsed:.3f}s"
        )

        return DocumentOCRResult(
            document_id=document.id,
            pages=page_results,
            full_text=full_doc_text,
            total_words=len(all_words),
            total_lines=len(all_lines),
            total_blocks=len(all_blocks),
            mean_confidence=mean_conf,
            engine_name=self.engine.engine_name,
            total_duration_seconds=round(total_elapsed, 4),
            metadata={
                "page_errors": page_errors,
                "has_errors": len(page_errors) > 0,
            },
        )

    def _resolve_page_image(
        self,
        page: DocumentPage,
        document_id: str,
        processed_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Resolve whether to use the preprocessed image or fallback to original page image."""
        if self.use_preprocessed_images:
            # Check for standard preprocessed location
            if processed_dir:
                candidate = Path(processed_dir) / document_id / f"page_{page.page_number:03d}_processed.png"
                if candidate.exists():
                    return candidate

            # Check if sibling page_XXX_processed.png exists near page.image_path
            if page.image_path:
                orig_path = Path(page.image_path)
                sibling_processed = orig_path.parent / f"page_{page.page_number:03d}_processed.png"
                if sibling_processed.exists():
                    return sibling_processed

        if page.image_path and Path(page.image_path).exists():
            return Path(page.image_path)

        raise OCRProcessingError(
            message=f"No readable image found for page {page.page_number} (Doc ID: {document_id})",
            document_id=document_id,
            page_number=page.page_number,
            engine_name=self.engine.engine_name,
        )
