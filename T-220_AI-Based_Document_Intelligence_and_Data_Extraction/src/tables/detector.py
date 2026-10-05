"""Table region detection with invoice-aware boundary and false-positive suppression."""

from __future__ import annotations

import re
from typing import List, Optional, Set

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.ocr.models import OCRLine, OCRWord
from src.tables.base import BaseTableDetector
from src.tables.header_detector import HeaderDetector
from src.tables.models import TableHeader, TableRegion

logger = get_logger("tables.detector")

TERMINATION_KEYWORDS: Set[str] = {
    "terms and conditions",
    "terms & conditions",
    "authorized signature",
    "signature",
    "notes:",
    "special instructions",
    "thank you for your business",
    "bank details",
    "wire transfer instructions",
    "remit to:",
    "remit payment to:",
    "payment terms",
    "payment due",
    "subtotal",
    "grand total",
    "total due",
    "amount due",
    "bill to",
    "ship to",
    "invoice no",
    "invoice number",
    "due date",
    "item details",
    "gst",
    "vat",
}


def is_termination_line(text: str) -> bool:
    """Check whether a line marks invoice metadata, summary, or footer text."""
    cleaned = re.sub(r"[^\w\s-]", "", text.strip().lower())
    normalized = re.sub(r"\s+", " ", cleaned).strip()

    for keyword in TERMINATION_KEYWORDS:
        key = re.sub(r"[^\w\s-]", "", keyword.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if re.search(r"\b" + re.escape(key) + r"\b", normalized):
            return True

    return False


def _group_words_into_lines(words: List[OCRWord]) -> List[List[OCRWord]]:
    """Group OCR words by vertical proximity."""
    if not words:
        return []

    ordered = sorted(
        words,
        key=lambda word: (
            (word.bounding_box.ymin + word.bounding_box.ymax) / 2.0,
            word.bounding_box.xmin,
        ),
    )

    result: List[List[OCRWord]] = []
    current: List[OCRWord] = [ordered[0]]

    for word in ordered[1:]:
        word_mid = (
            word.bounding_box.ymin + word.bounding_box.ymax
        ) / 2.0
        current_mid = sum(
            (item.bounding_box.ymin + item.bounding_box.ymax) / 2.0
            for item in current
        ) / len(current)

        if abs(word_mid - current_mid) <= 12.0:
            current.append(word)
        else:
            result.append(sorted(current, key=lambda item: item.bounding_box.xmin))
            current = [word]

    if current:
        result.append(sorted(current, key=lambda item: item.bounding_box.xmin))

    return result


def _line_bbox(row_words: List[OCRWord]) -> BoundingBox:
    """Build the bounding box for a line of OCR words."""
    return BoundingBox(
        xmin=min(word.bounding_box.xmin for word in row_words),
        ymin=min(word.bounding_box.ymin for word in row_words),
        xmax=max(word.bounding_box.xmax for word in row_words),
        ymax=max(word.bounding_box.ymax for word in row_words),
    )


def _looks_like_product_row(row_words: List[OCRWord]) -> bool:
    """Check for a product-like row with description and numeric evidence."""
    text = " ".join(word.text for word in row_words).strip()
    if not text or is_termination_line(text):
        return False

    # Ignore common invoice metadata and address lines.
    if re.search(
        r"\b(invoice\s*(?:no|number|date)|due\s*date|bill\s*to|ship\s*to|"
        r"payment\s*terms?|subtotal|grand\s*total|total\s*due|"
        r"gst|vat|tax|street|road|avenue|hyderabad|telangana|india)\b",
        text,
        re.IGNORECASE,
    ):
        return False

    # A product row should have at least two separated numeric tokens,
    # usually an item/HSN, quantity, price, or amount.
    numeric_tokens = re.findall(
        r"(?<![A-Za-z])(?:₹|\$|€|£)?\s*-?\d[\d,]*(?:\.\d{1,2})?(?![A-Za-z])",
        text,
    )
    if len(numeric_tokens) < 2:
        return False

    # Require some alphabetic description content.
    alpha_count = sum(char.isalpha() for char in text)
    if alpha_count < 3:
        return False

    return True


class RuleBasedTableDetector(BaseTableDetector):
    """Detect table regions using headers and repeated product-row patterns."""

    def __init__(
        self,
        min_rows: int = 1,
        min_columns: int = 2,
        header_detector: Optional[HeaderDetector] = None,
        max_row_gap_px: float = 60.0,
        enable_headerless_detection: bool = True,
    ) -> None:
        self.min_rows = min_rows
        self.min_columns = min_columns
        self.header_detector = header_detector or HeaderDetector()
        self.max_row_gap_px = max_row_gap_px
        self.enable_headerless_detection = enable_headerless_detection

    def detect_regions(
        self,
        words: List[OCRWord],
        lines: Optional[List[OCRLine]] = None,
        page_number: int = 1,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> List[TableRegion]:
        """Detect candidate table regions on a single page."""
        if not words:
            return []

        headers = self.header_detector.detect_header_candidates(
            words=words,
            lines=lines,
            page_number=page_number,
        )

        regions: List[TableRegion] = []
        for header in headers:
            region = self._grow_region_from_header(
                header=header,
                words=words,
                lines=lines,
                page_number=page_number,
                page_height=page_height,
            )
            if region and self._validate_table_region(region):
                regions.append(region)

        # Only use the headerless fallback when no credible header-based
        # region was detected.
        if not regions and self.enable_headerless_detection:
            region = self._detect_headerless_table(
                words=words,
                lines=lines,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
            )
            if region and self._validate_table_region(region):
                regions.append(region)

        logger.info(
            "Detected %s table region(s) on Page %s.",
            len(regions),
            page_number,
        )
        return regions

    def _grow_region_from_header(
        self,
        header: TableHeader,
        words: List[OCRWord],
        lines: Optional[List[OCRLine]],
        page_number: int,
        page_height: Optional[float],
    ) -> Optional[TableRegion]:
        """Grow a detected header only through nearby table-like rows."""
        if not header.bounding_box:
            return None

        header_ymax = header.bounding_box.ymax
        header_cells = [
            cell for cell in header.cells if cell.bounding_box is not None
        ]
        centers = [
            (cell.bounding_box.xmin + cell.bounding_box.xmax) / 2.0
            for cell in header_cells
        ]

        below = [
            word for word in words
            if word.bounding_box.ymin >= header_ymax - 2.0
        ]
        row_lines = _group_words_into_lines(below)

        included: List[List[OCRWord]] = []
        last_ymax = header_ymax

        for row in row_lines:
            bbox = _line_bbox(row)
            text = " ".join(word.text for word in row)

            if bbox.ymin - last_ymax > self.max_row_gap_px:
                break
            if is_termination_line(text):
                break

            # A single aligned text field is not enough to establish a row.
            aligned = set()
            for word in row:
                center = (
                    word.bounding_box.xmin + word.bounding_box.xmax
                ) / 2.0
                for index, header_center in enumerate(centers):
                    cell_width = header_cells[index].bounding_box.width
                    tolerance = max(35.0, cell_width * 1.5)
                    if abs(center - header_center) <= tolerance:
                        aligned.add(index)

            product_like = _looks_like_product_row(row)
            enough_alignment = len(aligned) >= min(2, len(centers))

            if not product_like and not enough_alignment:
                # Ignore page text instead of expanding the table boundary.
                continue

            if len(centers) >= 2 and not enough_alignment and not product_like:
                continue

            included.append(row)
            last_ymax = bbox.ymax

        if not included:
            return None

        header_words = [
            word for cell in header.cells for word in cell.words
        ]
        table_words = header_words + [
            word for row in included for word in row
        ]
        if not table_words:
            return None

        region_bbox = BoundingBox(
            xmin=min(word.bounding_box.xmin for word in table_words),
            ymin=min(word.bounding_box.ymin for word in table_words),
            xmax=max(word.bounding_box.xmax for word in table_words),
            ymax=max(word.bounding_box.ymax for word in table_words),
        )

        row_boxes = [_line_bbox(row) for row in included]
        confidence = min(
            1.0,
            header.confidence * 0.6
            + min(1.0, len(included) / 3.0) * 0.4,
        )

        return TableRegion(
            page_number=page_number,
            bounding_box=region_bbox,
            header=header,
            row_boxes=row_boxes,
            confidence_score=confidence,
            detection_signals={
                "header_score": header.confidence,
                "data_rows_count": len(included),
                "has_header": True,
            },
        )

    def _detect_headerless_table(
        self,
        words: List[OCRWord],
        lines: Optional[List[OCRLine]],
        page_number: int,
        page_width: Optional[float],
        page_height: Optional[float],
    ) -> Optional[TableRegion]:
        """Find a compact sequence of repeated product-like rows."""
        if len(words) < 6:
            return None

        all_lines = _group_words_into_lines(words)
        candidates = []

        for row in all_lines:
            if len(row) < self.min_columns:
                continue
            if _looks_like_product_row(row):
                candidates.append(row)

        if len(candidates) < max(2, self.min_rows):
            logger.info(
                "No reliable headerless table found on Page %s.",
                page_number,
            )
            return None

        # Split candidates into nearby runs. Do not combine unrelated numeric
        # rows from the header, address block, totals, and footer.
        runs: List[List[List[OCRWord]]] = []
        current_run: List[List[OCRWord]] = []

        for row in candidates:
            bbox = _line_bbox(row)
            if current_run:
                previous_bbox = _line_bbox(current_run[-1])
                gap = bbox.ymin - previous_bbox.ymax
                if gap > self.max_row_gap_px:
                    if len(current_run) >= max(2, self.min_rows):
                        runs.append(current_run)
                    current_run = []
            current_run.append(row)

        if len(current_run) >= max(2, self.min_rows):
            runs.append(current_run)

        if not runs:
            return None

        # Prefer the run with the most product-like rows, then the smallest
        # vertical span. This avoids using the whole page as a table.
        selected = max(
            runs,
            key=lambda run: (
                len(run),
                -(_line_bbox(run[-1]).ymax - _line_bbox(run[0]).ymin),
            ),
        )

        selected_words = [word for row in selected for word in row]
        bbox = BoundingBox(
            xmin=min(word.bounding_box.xmin for word in selected_words),
            ymin=min(word.bounding_box.ymin for word in selected_words),
            xmax=max(word.bounding_box.xmax for word in selected_words),
            ymax=max(word.bounding_box.ymax for word in selected_words),
        )

        return TableRegion(
            page_number=page_number,
            bounding_box=bbox,
            header=None,
            row_boxes=[_line_bbox(row) for row in selected],
            confidence_score=0.65,
            detection_signals={
                "has_header": False,
                "data_rows_count": len(selected),
            },
        )

    def _validate_table_region(self, region: TableRegion) -> bool:
        """Reject regions that are too small to contain a useful table."""
        return (
            region.bounding_box.width > 50.0
            and region.bounding_box.height > 15.0
        )


__all__ = [
    "RuleBasedTableDetector",
    "is_termination_line",
]
