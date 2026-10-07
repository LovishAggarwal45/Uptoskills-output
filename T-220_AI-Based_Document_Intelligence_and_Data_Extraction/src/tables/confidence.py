"""Explainable table confidence scoring from structural, spatial, arithmetic, and OCR signals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.tables.base import BaseTableConfidenceScorer
from src.tables.models import Table

logger = get_logger("tables.confidence")


class TableConfidenceScorer(BaseTableConfidenceScorer):
    """Calculates explainable composite confidence score for reconstructed tables."""

    def __init__(
        self,
        header_weight: float = 0.25,
        column_alignment_weight: float = 0.20,
        row_regularity_weight: float = 0.20,
        arithmetic_weight: float = 0.20,
        ocr_confidence_weight: float = 0.15,
    ) -> None:
        self.header_weight = header_weight
        self.column_alignment_weight = column_alignment_weight
        self.row_regularity_weight = row_regularity_weight
        self.arithmetic_weight = arithmetic_weight
        self.ocr_confidence_weight = ocr_confidence_weight

    def calculate_confidence(self, table: Table) -> float:
        """Compute composite confidence score and breakdown for a reconstructed table.

        Args:
            table: Reconstructed Table instance with rows, columns, and line items.

        Returns:
            Composite confidence score in range [0.0, 1.0].
        """
        score, breakdown = self.evaluate_breakdown(table)
        table.confidence = score
        table.confidence_breakdown = breakdown
        return score

    def evaluate_breakdown(self, table: Table) -> Tuple[float, Dict[str, float]]:
        """Compute detailed signal scores contributing to table confidence."""
        # 1. Header Recognition Signal
        header_score = self._score_header(table)

        # 2. Column Alignment Signal
        col_score = self._score_columns(table)

        # 3. Row Regularity Signal
        row_score = self._score_rows(table)

        # 4. Arithmetic Consistency Signal
        arithmetic_score = self._score_arithmetic(table)

        # 5. Cell OCR Recognition Signal
        ocr_score = self._score_cell_ocr(table)

        # Weighted composite calculation
        total = (
            (header_score * self.header_weight)
            + (col_score * self.column_alignment_weight)
            + (row_score * self.row_regularity_weight)
            + (arithmetic_score * self.arithmetic_weight)
            + (ocr_score * self.ocr_confidence_weight)
        )

        composite = max(0.0, min(1.0, total))

        breakdown = {
            "header_recognition": round(header_score, 4),
            "column_alignment": round(col_score, 4),
            "row_regularity": round(row_score, 4),
            "arithmetic_consistency": round(arithmetic_score, 4),
            "ocr_confidence": round(ocr_score, 4),
            "composite_confidence": round(composite, 4),
        }

        return composite, breakdown

    def _score_header(self, table: Table) -> float:
        """Score header quality based on explicit detection and canonical column mappings."""
        if not table.columns:
            return 0.0

        canonical_count = sum(1 for c in table.columns if c.canonical_field is not None)
        has_explicit = table.metadata.get("has_explicit_header", len(table.headers) > 0)

        if has_explicit:
            if canonical_count >= 3:
                return 1.00
            elif canonical_count >= 2:
                return 0.85
            elif canonical_count >= 1:
                return 0.70
            return 0.60
        else:
            return 0.35  # Headerless table

    def _score_columns(self, table: Table) -> float:
        """Score column sharpness and non-overlapping boundary geometry."""
        if not table.columns or len(table.columns) < 2:
            return 0.50

        # Check column width validity
        all_valid_widths = all(col.width > 5.0 for col in table.columns)
        if not all_valid_widths:
            return 0.40

        # Check non-overlapping order
        is_ordered = all(
            table.columns[i].x_start < table.columns[i + 1].x_start
            for i in range(len(table.columns) - 1)
        )

        return 0.95 if is_ordered else 0.60

    def _score_rows(self, table: Table) -> float:
        """Score row regularity and vertical spacing consistency."""
        if not table.rows:
            return 0.0
        if len(table.rows) == 1:
            return 0.75

        # Check row heights
        heights = [
            r.bounding_box.height
            for r in table.rows
            if r.bounding_box
        ]
        if not heights:
            return 0.50

        mean_h = sum(heights) / len(heights)
        if mean_h <= 0:
            return 0.50

        # Height variance
        max_dev = max(abs(h - mean_h) for h in heights)
        dev_ratio = max_dev / mean_h

        if dev_ratio <= 0.5:
            return 0.95
        elif dev_ratio <= 1.0:
            return 0.80
        return 0.65

    def _score_arithmetic(self, table: Table) -> float:
        """Score based on arithmetic validity of itemized rows."""
        if not table.line_items:
            return 0.80

        validated_items = [
            item for item in table.line_items
            if item.is_valid_arithmetic is not None
        ]
        if not validated_items:
            return 0.80

        valid_count = sum(1 for item in validated_items if item.is_valid_arithmetic)
        return valid_count / len(validated_items)

    def _score_cell_ocr(self, table: Table) -> float:
        """Score mean OCR confidence across all non-empty table cells."""
        confidences: List[float] = []
        for row in table.rows:
            for cell in row.cells:
                if cell.confidence is not None and cell.text.strip():
                    confidences.append(cell.confidence)

        if not confidences:
            return 0.75
        return sum(confidences) / len(confidences)


__all__ = [
    "TableConfidenceScorer",
]
