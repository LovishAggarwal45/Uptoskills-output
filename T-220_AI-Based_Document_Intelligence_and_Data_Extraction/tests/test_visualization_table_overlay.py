"""Unit tests for TableOverlayBuilder creating table, cell, and line-item annotations."""

import unittest

from src.core.models import BoundingBox
from src.core.types import SeverityLevel, ValidationStatus
from src.visualization.models import AnnotationType, EvidenceRegion
from src.visualization.table_overlay import TableOverlayBuilder


class TestTableOverlay(unittest.TestCase):
    """Test suite for table visual overlay builder."""

    def test_build_valid_table_cell_annotation(self) -> None:
        """Verify building table cell visual annotation."""
        region = EvidenceRegion(
            evidence_id="ev_cell_1",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.TABLE_CELL,
            bbox=BoundingBox(50, 100, 200, 130),
            table_id="table_1",
            row_index=0,
            column_name="description",
            source_text="Cloud Storage",
            extraction_confidence=0.95,
            is_spatial=True,
        )

        ann = TableOverlayBuilder.build_table_annotation(region)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.table_id, "table_1")
        self.assertEqual(ann.row_index, 0)
        self.assertEqual(ann.badge_text, "[CELL]")
        self.assertEqual(ann.severity, SeverityLevel.INFO)

    def test_build_line_item_math_mismatch_highlights_error(self) -> None:
        """Verify line item with arithmetic mismatch is highlighted in red with MATH-ERR."""
        region = EvidenceRegion(
            evidence_id="ev_item_1",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.TABLE_LINE_ITEM,
            bbox=BoundingBox(50, 100, 600, 150),
            table_id="table_1",
            row_index=1,
            source_text="Hardware Unit",
            validation_status=ValidationStatus.INVALID,
            review_required=True,
            review_reasons=["Line item arithmetic mismatch: 3 * 100 != 250"],
            is_spatial=True,
        )

        ann = TableOverlayBuilder.build_table_annotation(region)
        self.assertIsNotNone(ann)
        self.assertIn("MATH-ERR", ann.label)
        self.assertEqual(ann.severity, SeverityLevel.CRITICAL)


if __name__ == "__main__":
    unittest.main()
