"""Master Table Intelligence and Line-Item Extraction Coordinator."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import DocuMindConfig, TableConfig
from src.core.logging import get_logger
from src.core.models import Document, DocumentPage
from src.ocr.models import OCRLine, OCRWord
from src.tables.base import BaseTableExtractor
from src.tables.cell_parser import CellParser
from src.tables.column_parser import ColumnParser
from src.tables.confidence import TableConfidenceScorer
from src.tables.continuation import TableContinuationDetector
from src.tables.detector import RuleBasedTableDetector
from src.tables.exceptions import InvalidTableInput
from src.tables.header_detector import HeaderDetector
from src.tables.line_item_extractor import LineItemExtractor
from src.tables.models import (
    LineItem,
    Table,
    TableExtractionResult,
    TableRegion,
)
from src.tables.row_parser import RowParser
from src.tables.structure import TableStructureReconstructor
from src.tables.validators import TableValidator

logger = get_logger("tables.extractor")


class TableExtractor(BaseTableExtractor):
    """Production-grade coordinator for table detection, 2D grid reconstruction, line-item extraction, and validation."""

    def __init__(
        self,
        config: Optional[TableConfig] = None,
    ) -> None:
        self.config = config or TableConfig()

        self.header_detector = HeaderDetector(
            min_header_columns=self.config.detection.min_columns,
            line_height_tolerance_px=self.config.reconstruction.row_tolerance_px,
        )
        self.column_parser = ColumnParser(
            min_column_gap_px=self.config.reconstruction.column_gap_threshold_px,
        )
        self.row_parser = RowParser(
            line_height_tolerance_px=self.config.reconstruction.row_tolerance_px,
        )
        self.structure_reconstructor = TableStructureReconstructor(
            column_parser=self.column_parser,
            row_parser=self.row_parser,
            header_detector=self.header_detector,
        )
        self.detector = RuleBasedTableDetector(
            min_rows=self.config.detection.min_rows,
            min_columns=self.config.detection.min_columns,
            header_detector=self.header_detector,
            max_row_gap_px=self.config.detection.max_row_gap_px,
            enable_headerless_detection=self.config.detection.enable_headerless_detection,
        )
        self.line_item_extractor = LineItemExtractor(
            amount_tolerance=self.config.validation.amount_tolerance,
        )
        self.validator = TableValidator(
            amount_tolerance=self.config.validation.amount_tolerance,
            total_tolerance=self.config.validation.total_tolerance,
        )
        self.continuation_detector = TableContinuationDetector(
            min_column_overlap_ratio=self.config.continuation.min_column_overlap_ratio,
        )
        self.confidence_scorer = TableConfidenceScorer()

    def extract(
        self,
        document: Document,
        document_totals: Optional[Dict[str, float]] = None,
    ) -> TableExtractionResult:
        """Extract all tables and canonical line items from an ingested multi-page document.

        Args:
            document: Document instance populated with pages and OCR text regions.
            document_totals: Optional dictionary of extracted document-level totals for cross-validation.

        Returns:
            TableExtractionResult containing reconstructed tables, line items, and statistics.

        Raises:
            InvalidTableInput: If document is None or invalid.
        """
        if document is None or not isinstance(document, Document):
            raise InvalidTableInput("Invalid input: document must be an instance of Document.")

        start_time = time.perf_counter()
        doc_id = document.id

        logger.info(f"Initiating table extraction for document '{doc_id}' ({len(document.pages)} pages).")

        extracted_tables: List[Table] = []
        all_line_items: List[LineItem] = []
        warnings: List[str] = []

        # 1. Process each page individually
        for page in document.pages:
            page_tables = self.extract_page(
                page=page,
                document_id=doc_id,
                document_totals=document_totals,
            )
            extracted_tables.extend(page_tables)

        # 2. Multi-Page Continuation Linking
        if len(extracted_tables) > 1 and self.config.continuation.enabled:
            extracted_tables = self.continuation_detector.link_continuations(extracted_tables)

        # 3. Collect Line Items and Continuation Pairs
        continuation_pairs: List[Tuple[str, str]] = []
        for t in extracted_tables:
            all_line_items.extend(t.line_items)
            if t.is_continuation and t.continuation_of_table_id:
                continuation_pairs.append((t.continuation_of_table_id, t.table_id))

            # Add table validation warnings if any
            if t.validation_result and not t.validation_result.is_valid:
                for msg in t.validation_result.messages:
                    warnings.append(f"Table '{t.table_id}' [Page {t.page_number}]: {msg}")

        elapsed = time.perf_counter() - start_time

        stats = {
            "total_pages": len(document.pages),
            "tables_extracted": len(extracted_tables),
            "line_items_extracted": len(all_line_items),
            "multi_page_continuations": len(continuation_pairs),
            "execution_time_seconds": round(elapsed, 4),
        }

        logger.info(
            f"Table extraction completed for '{doc_id}': {len(extracted_tables)} tables, "
            f"{len(all_line_items)} line items ({elapsed*1000:.1f}ms)."
        )

        return TableExtractionResult(
            document_id=doc_id,
            tables=extracted_tables,
            line_items=all_line_items,
            unresolved_tables=[],
            continuation_pairs=continuation_pairs,
            statistics=stats,
            execution_time_seconds=elapsed,
            warnings=warnings,
            metadata={
                "table_count": len(extracted_tables),
                "line_item_count": len(all_line_items),
            },
        )

    def extract_page(
        self,
        page: DocumentPage,
        document_id: str = "doc",
        document_totals: Optional[Dict[str, float]] = None,
    ) -> List[Table]:
        """Extract tables from a single document page.

        Args:
            page: Single DocumentPage instance.
            document_id: Document ID for provenance tracking.
            document_totals: Optional document-level totals for cross-validation.

        Returns:
            List of reconstructed and validated Table instances on the page.
        """
        words = self._extract_ocr_words_from_page(page)
        if not words:
            return []

        # Detect candidate regions
        regions = self.detector.detect_regions(
            words=words,
            page_number=page.page_number,
            page_width=page.width,
            page_height=page.height,
        )

        page_tables: List[Table] = []

        for idx, region in enumerate(regions):
            table_id = f"table_{page.page_number}_{idx + 1}"

            # Reconstruct 2D Table Grid
            table = self.structure_reconstructor.reconstruct_table(
                region=region,
                words=words,
                table_id=table_id,
                page_number=page.page_number,
            )

            # Extract Line Items
            line_items = self.line_item_extractor.extract_line_items(
                table=table,
                document_id=document_id,
            )
            table.line_items = line_items

            # Validate Table
            val_res = self.validator.validate_table(
                table=table,
                document_totals=document_totals,
            )
            table.validation_result = val_res
            table.validation_status = (
                table.validation_status
                if not val_res
                else (
                    table.validation_status
                    if not val_res.is_valid
                    else table.validation_status
                )
            )

            # Calculate Composite Confidence Score
            self.confidence_scorer.calculate_confidence(table)

            page_tables.append(table)

        return page_tables

    def _extract_ocr_words_from_page(self, page: DocumentPage) -> List[OCRWord]:
        """Convert page OCRTextRegions or metadata words into OCRWord instances."""
        words: List[OCRWord] = []

        # Check if page has structured OCR words stored in metadata
        if "ocr_words" in page.metadata:
            for w_data in page.metadata["ocr_words"]:
                if isinstance(w_data, OCRWord):
                    words.append(w_data)
                elif isinstance(w_data, dict):
                    words.append(OCRWord(
                        text=w_data["text"],
                        confidence=float(w_data.get("confidence", 0.9)),
                        raw_confidence=float(w_data.get("raw_confidence", 90.0)),
                        bounding_box=w_data["bounding_box"],
                        page_number=page.page_number,
                    ))
            if words:
                return words

        # Extract from page.ocr_text_regions
        for idx, tr in enumerate(page.ocr_text_regions):
            if tr.sub_regions:
                # Line or Block with word sub-regions
                for sub in tr.sub_regions:
                    if sub.bounding_box:
                        words.append(
                            OCRWord(
                                text=sub.text,
                                confidence=sub.confidence if sub.confidence is not None else 0.90,
                                raw_confidence=(sub.confidence * 100.0) if sub.confidence is not None else 90.0,
                                bounding_box=sub.bounding_box,
                                page_number=page.page_number,
                                word_num=idx,
                            )
                        )
            elif tr.bounding_box:
                words.append(
                    OCRWord(
                        text=tr.text,
                        confidence=tr.confidence if tr.confidence is not None else 0.90,
                        raw_confidence=(tr.confidence * 100.0) if tr.confidence is not None else 90.0,
                        bounding_box=tr.bounding_box,
                        page_number=page.page_number,
                        word_num=idx,
                    )
                )

        return words

    def extract_document(
        self,
        document: Document,
        document_totals: Optional[Dict[str, float]] = None,
    ) -> TableExtractionResult:
        """Alias for extract()."""
        return self.extract(document, document_totals=document_totals)


__all__ = [
    "TableExtractor",
]
