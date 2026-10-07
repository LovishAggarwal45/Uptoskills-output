"""Unit tests for ReviewOverlayBuilder creating review targets and validation issue visual highlights."""

import unittest

from src.core.models import BoundingBox
from src.core.types import SeverityLevel, ValidationStatus
from src.confidence.models import ConfidenceBand
from src.visualization.models import AnnotationType, EvidenceRegion
from src.visualization.review_overlay import ReviewOverlayBuilder


class TestReviewOverlay(unittest.TestCase):
    """Test suite for human review visual overlay builder."""

    def test_build_urgent_review_target(self) -> None:
        """Verify building high-visibility URGENT review target annotation."""
        region = EvidenceRegion(
            evidence_id="ev_rev_01",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.REVIEW_CRITICAL,
            bbox=BoundingBox(100, 300, 450, 350),
            field_name="total",
            review_required=True,
            review_reasons=["Critical grand-total calculation mismatch"],
            validation_status=ValidationStatus.INVALID,
            is_spatial=True,
        )

        ann = ReviewOverlayBuilder.build_review_annotation(region, show_reasons=True, line_width=3)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.severity, SeverityLevel.CRITICAL)
        self.assertEqual(ann.line_width, 3)
        self.assertIn("Critical", ann.label)

    def test_build_warning_review_target(self) -> None:
        """Verify building WARNING review annotation."""
        region = EvidenceRegion(
            evidence_id="ev_rev_02",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.VALIDATION_WARNING,
            bbox=BoundingBox(50, 150, 200, 180),
            field_name="due_date",
            review_required=False,
            review_reasons=["Due date exceeds standard 90-day window"],
            validation_status=ValidationStatus.WARNING,
            is_spatial=True,
        )

        ann = ReviewOverlayBuilder.build_review_annotation(region, show_reasons=True)
        self.assertIsNotNone(ann)
        self.assertEqual(ann.severity, SeverityLevel.WARNING)


if __name__ == "__main__":
    unittest.main()
