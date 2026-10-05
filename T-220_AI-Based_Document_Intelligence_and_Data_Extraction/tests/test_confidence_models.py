"""Unit tests for confidence and review routing domain models."""

import unittest
from datetime import datetime, timezone

from src.core.models import BoundingBox, Provenance, ReviewFlag
from src.core.types import DocumentType, ExtractionMethod, SeverityLevel, ValidationStatus
from src.confidence.models import (
    ConfidenceBand,
    ConfidenceSignals,
    DocumentConfidence,
    FieldConfidence,
    LineItemConfidence,
    ReviewIssue,
    ReviewPriority,
    ReviewQueueItem,
    ReviewReason,
    ReviewRoutingResult,
    ReviewStatus,
    ReviewTarget,
    ReviewTargetType,
    TableConfidence,
)


class TestConfidenceModels(unittest.TestCase):
    """Test suite for Phase 8 data model creation, boundaries, and validation."""

    def test_confidence_signals_bounded(self) -> None:
        """Verify ConfidenceSignals enforces [0.0, 1.0] bounds on all signals."""
        signals = ConfidenceSignals(
            ocr_confidence=0.92,
            extraction_confidence=0.88,
            classification_confidence=0.95,
            validation_score=1.0,
            conflict_count=0,
            required_field_missing=False,
        )
        self.assertEqual(signals.ocr_confidence, 0.92)
        self.assertEqual(signals.extraction_confidence, 0.88)
        self.assertEqual(signals.conflict_count, 0)

        # Out-of-bounds raises ValueError
        with self.assertRaises(ValueError):
            ConfidenceSignals(ocr_confidence=1.5)

        with self.assertRaises(ValueError):
            ConfidenceSignals(extraction_confidence=-0.1)

    def test_confidence_signals_missing_values_allowed(self) -> None:
        """Verify ConfidenceSignals allows None for missing signals without fabrication."""
        signals = ConfidenceSignals(
            ocr_confidence=None,
            extraction_confidence=0.75,
            table_confidence=None,
        )
        self.assertIsNone(signals.ocr_confidence)
        self.assertIsNone(signals.table_confidence)
        self.assertEqual(signals.extraction_confidence, 0.75)

    def test_field_confidence_creation(self) -> None:
        """Verify FieldConfidence creation, bounds, and provenance preservation."""
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance("doc_1", 1, "$450.00", bbox, 0.98, ExtractionMethod.REGEX_PATTERN)

        fc = FieldConfidence(
            field_name="total",
            value="$450.00",
            normalized_value=450.00,
            original_extraction_confidence=0.95,
            ocr_confidence=0.98,
            validation_status=ValidationStatus.VALID,
            aggregated_confidence=0.96,
            confidence_band=ConfidenceBand.HIGH,
            review_required=False,
            provenance=prov,
        )

        self.assertEqual(fc.field_name, "total")
        self.assertEqual(fc.original_extraction_confidence, 0.95)
        self.assertEqual(fc.ocr_confidence, 0.98)
        self.assertEqual(fc.aggregated_confidence, 0.96)
        self.assertEqual(fc.confidence_band, ConfidenceBand.HIGH)
        self.assertFalse(fc.review_required)
        self.assertIsNotNone(fc.provenance)

    def test_line_item_and_table_confidence(self) -> None:
        """Verify LineItemConfidence and TableConfidence models."""
        lic = LineItemConfidence(
            row_index=0,
            description="Consulting Services",
            aggregated_confidence=0.94,
            confidence_band=ConfidenceBand.HIGH,
            original_confidence=0.95,
            ocr_confidence=0.98,
            is_valid_arithmetic=True,
            review_required=False,
        )

        tc = TableConfidence(
            table_id="tbl_main",
            page_number=1,
            structure_confidence=0.90,
            row_confidence=0.92,
            cell_confidence=0.95,
            validation_status=ValidationStatus.VALID,
            aggregated_confidence=0.93,
            confidence_band=ConfidenceBand.HIGH,
            review_required=False,
            line_item_confidences=[lic],
        )

        self.assertEqual(tc.table_id, "tbl_main")
        self.assertEqual(len(tc.line_item_confidences), 1)
        self.assertEqual(tc.line_item_confidences[0].description, "Consulting Services")
        self.assertTrue(tc.line_item_confidences[0].is_valid_arithmetic)

    def test_review_target_and_issue(self) -> None:
        """Verify ReviewTarget spatial bounding and ReviewIssue."""
        bbox = BoundingBox(xmin=50.0, ymin=50.0, xmax=200.0, ymax=80.0)
        target = ReviewTarget(
            target_type=ReviewTargetType.FIELD,
            page_number=1,
            field_name="invoice_date",
            bounding_box=bbox,
            reason="Invalid date format",
            severity=SeverityLevel.ERROR,
        )

        issue = ReviewIssue(
            issue_id="ISSUE_DATE_01",
            rule_name="Date Format Validation",
            severity=SeverityLevel.ERROR,
            message="The extracted invoice date '99/99/2026' is invalid",
            affected_targets=[target],
            source_signal="validation_report",
        )

        self.assertEqual(issue.issue_id, "ISSUE_DATE_01")
        self.assertEqual(len(issue.affected_targets), 1)
        self.assertEqual(issue.affected_targets[0].field_name, "invoice_date")
        self.assertEqual(issue.affected_targets[0].page_number, 1)
        self.assertIsNotNone(issue.affected_targets[0].bounding_box)

    def test_review_queue_item_properties(self) -> None:
        """Verify ReviewQueueItem construction and helper properties."""
        target = ReviewTarget(
            target_type=ReviewTargetType.DOCUMENT,
            page_number=1,
            reason="Document classification ambiguous",
            severity=SeverityLevel.ERROR,
        )
        issue = ReviewIssue(
            issue_id="ISSUE_CLF_01",
            rule_name="Classification Check",
            severity=SeverityLevel.ERROR,
            message="Ambiguous document type",
            affected_targets=[target],
        )

        item = ReviewQueueItem(
            document_id="doc_queue_001",
            document_type=DocumentType.UNKNOWN,
            priority=ReviewPriority.HIGH,
            status=ReviewStatus.PENDING,
            document_confidence=0.55,
            confidence_band=ConfidenceBand.LOW,
            validation_status=ValidationStatus.INVALID,
            review_required=True,
            reasons=["Ambiguous document type", "Confidence below threshold"],
            issues=[issue],
            targets=[target],
        )

        self.assertEqual(item.document_id, "doc_queue_001")
        self.assertEqual(item.priority, ReviewPriority.HIGH)
        self.assertEqual(item.status, ReviewStatus.PENDING)
        self.assertEqual(item.target_count, 1)
        self.assertEqual(item.critical_issue_count, 0)
        self.assertTrue(item.review_required)


if __name__ == "__main__":
    unittest.main()
