"""Unit tests for Phase 8 model dictionary and JSON serialization roundtrips."""

import json
import unittest

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


class TestConfidenceSerialization(unittest.TestCase):
    """Test suite ensuring all Phase 8 models support complete JSON roundtrip serialization."""

    def test_field_confidence_json_roundtrip(self) -> None:
        """Verify FieldConfidence serialization to dict and JSON."""
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance("doc_1", 1, "INV-2026-001", bbox, 0.98, ExtractionMethod.REGEX_PATTERN)

        fc = FieldConfidence(
            field_name="invoice_number",
            value="INV-2026-001",
            normalized_value="INV-2026-001",
            original_extraction_confidence=0.96,
            ocr_confidence=0.98,
            validation_status=ValidationStatus.VALID,
            aggregated_confidence=0.97,
            confidence_band=ConfidenceBand.HIGH,
            review_required=False,
            review_reasons=[],
            signals=ConfidenceSignals(ocr_confidence=0.98, extraction_confidence=0.96),
            provenance=prov,
            explanation="Strong recognition across all signals",
        )

        d = fc.to_dict()
        self.assertEqual(d["field_name"], "invoice_number")
        self.assertEqual(d["aggregated_confidence"], 0.97)

        reconstructed = FieldConfidence.from_dict(d)
        self.assertEqual(reconstructed.field_name, fc.field_name)
        self.assertEqual(reconstructed.aggregated_confidence, fc.aggregated_confidence)
        self.assertEqual(reconstructed.ocr_confidence, fc.ocr_confidence)
        self.assertEqual(reconstructed.provenance.bounding_box.xmin, 10.0)

    def test_review_queue_item_json_roundtrip(self) -> None:
        """Verify ReviewQueueItem serialization to dict and JSON."""
        bbox = BoundingBox(xmin=20.0, ymin=30.0, xmax=150.0, ymax=60.0)
        target = ReviewTarget(
            target_type=ReviewTargetType.FIELD,
            page_number=1,
            field_name="total",
            bounding_box=bbox,
            reason="Low confidence score",
            severity=SeverityLevel.ERROR,
        )

        issue = ReviewIssue(
            issue_id="ISSUE_TOTAL",
            rule_name="Total Confidence Check",
            severity=SeverityLevel.ERROR,
            message="Field 'total' has confidence 0.45",
            affected_targets=[target],
        )

        item = ReviewQueueItem(
            document_id="doc_json_01",
            document_type=DocumentType.INVOICE,
            priority=ReviewPriority.HIGH,
            status=ReviewStatus.PENDING,
            document_confidence=0.62,
            confidence_band=ConfidenceBand.LOW,
            validation_status=ValidationStatus.INVALID,
            review_required=True,
            reasons=["Field 'total' has low confidence"],
            issues=[issue],
            targets=[target],
        )

        json_str = item.to_json()
        self.assertIn("doc_json_01", json_str)
        self.assertIn("high", json_str)

        d = json.loads(json_str)
        reconstructed = ReviewQueueItem.from_dict(d)

        self.assertEqual(reconstructed.document_id, item.document_id)
        self.assertEqual(reconstructed.priority, ReviewPriority.HIGH)
        self.assertEqual(len(reconstructed.issues), 1)
        self.assertEqual(reconstructed.issues[0].affected_targets[0].field_name, "total")

    def test_review_routing_result_roundtrip(self) -> None:
        """Verify ReviewRoutingResult complete JSON serialization."""
        doc_conf = DocumentConfidence(
            document_id="doc_routing_01",
            document_type=DocumentType.INVOICE,
            overall_confidence=0.92,
            confidence_band=ConfidenceBand.HIGH,
            review_required=False,
        )

        res = ReviewRoutingResult(
            document_id="doc_routing_01",
            is_straight_through=True,
            document_confidence=doc_conf,
            queue_item=None,
        )

        json_str = res.to_json()
        d = json.loads(json_str)
        reconstructed = ReviewRoutingResult.from_dict(d)

        self.assertEqual(reconstructed.document_id, res.document_id)
        self.assertTrue(reconstructed.is_straight_through)
        self.assertEqual(reconstructed.document_confidence.overall_confidence, 0.92)


if __name__ == "__main__":
    unittest.main()
