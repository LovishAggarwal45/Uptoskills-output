"""Unit tests for ReviewRouter and end-to-end routing decisions."""

import unittest
from pathlib import Path

from src.core.config import ConfidenceConfig
from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    ExtractedField,
    Provenance,
)
from src.core.types import (
    DocumentType,
    ExtractionMethod,
    FieldType,
    SeverityLevel,
    ValidationStatus,
)
from src.confidence.models import (
    ConfidenceBand,
    ReviewPriority,
    ReviewStatus,
)
from src.confidence.review_router import ReviewRouter
from src.tables.models import LineItem, Table, TableValidationResult
from src.validation.models import ValidationIssue, ValidationReport


class TestReviewRouter(unittest.TestCase):
    """Test suite for ReviewRouter coordinating confidence calculations, review rules, and queue dispatch."""

    def setUp(self) -> None:
        self.router = ReviewRouter(config=ConfidenceConfig())

    def _create_invoice_doc(self) -> Document:
        meta = DocumentMetadata("doc_inv_route", "inv.pdf", Path("data/inv.pdf"), "application/pdf", 1024, "sha", 1)
        page = DocumentPage(1, raw_text="Invoice # INV-001\nTotal: $500.00")
        return Document(
            metadata=meta,
            pages=[page],
            classified_type=DocumentType.INVOICE,
            classification_confidence=0.95,
        )

    def test_clean_invoice_straight_through_processing(self) -> None:
        """Verify that a clean, highly confident document is marked STP / AUTO_APPROVED."""
        doc = self._create_invoice_doc()
        bbox = BoundingBox(xmin=10, ymin=10, xmax=100, ymax=30)
        prov = Provenance(doc.id, 1, "INV-001", bbox, 0.98, ExtractionMethod.REGEX_PATTERN)

        fields = {
            "invoice_number": ExtractedField(
                name="invoice_number",
                value="INV-001",
                normalized_value="INV-001",
                field_type=FieldType.IDENTIFIER,
                provenance=prov,
                extraction_confidence=0.96,
                validation_status=ValidationStatus.VALID,
                is_required=True,
            ),
            "total": ExtractedField(
                name="total",
                value="$500.00",
                normalized_value=500.00,
                field_type=FieldType.CURRENCY,
                provenance=prov,
                extraction_confidence=0.95,
                validation_status=ValidationStatus.VALID,
                is_required=True,
            ),
        }

        report = ValidationReport(
            document_id=doc.id,
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.VALID,
            validation_score=1.0,
        )

        res = self.router.route_document(
            document=doc,
            fields=fields,
            tables=[],
            validation_report=report,
        )

        self.assertTrue(res.is_straight_through)
        self.assertFalse(res.review_required)
        self.assertIsNotNone(res.queue_item)
        self.assertEqual(res.queue_item.status, ReviewStatus.AUTO_APPROVED)
        self.assertEqual(res.queue_item.priority, ReviewPriority.LOW)
        self.assertEqual(len(res.queue_item.issues), 0)

    def test_missing_required_field_routes_to_review(self) -> None:
        """Verify missing required field prevents STP and flags document for review with HIGH priority."""
        doc = self._create_invoice_doc()

        fields = {
            "invoice_number": ExtractedField(
                name="invoice_number",
                value="INV-001",
                normalized_value="INV-001",
                extraction_confidence=0.96,
                validation_status=ValidationStatus.VALID,
                is_required=True,
            ),
            "total": ExtractedField(
                name="total",
                value="",  # Missing value!
                normalized_value=None,
                extraction_confidence=0.0,
                validation_status=ValidationStatus.INVALID,
                is_required=True,
            ),
        }

        report = ValidationReport(
            document_id=doc.id,
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.INVALID,
            validation_score=0.50,
        )

        res = self.router.route_document(
            document=doc,
            fields=fields,
            tables=[],
            validation_report=report,
        )

        self.assertFalse(res.is_straight_through)
        self.assertTrue(res.review_required)
        self.assertEqual(res.queue_item.status, ReviewStatus.PENDING)
        self.assertEqual(res.queue_item.priority, ReviewPriority.HIGH)
        self.assertGreater(len(res.queue_item.targets), 0)

    def test_table_arithmetic_mismatch_routes_to_urgent_review(self) -> None:
        """Verify table calculation error triggers URGENT human review."""
        doc = self._create_invoice_doc()

        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-001", extraction_confidence=0.95),
            "total": ExtractedField(name="total", value="$500.00", extraction_confidence=0.95),
        }

        table = Table(
            table_id="tbl_1",
            page_number=1,
            headers=["Item", "Qty", "Price", "Amount"],
            confidence=0.90,
            line_items=[
                LineItem(row_index=0, description="Server", quantity=2.0, unit_price=100.0, amount=300.0, is_valid_arithmetic=False)
            ],
            validation_status=ValidationStatus.INVALID,
            validation_result=TableValidationResult(is_valid=False, row_checks_failed=1),
        )

        res = self.router.route_document(
            document=doc,
            fields=fields,
            tables=[table],
            validation_report=None,
        )

        self.assertFalse(res.is_straight_through)
        self.assertTrue(res.review_required)
        self.assertEqual(res.queue_item.priority, ReviewPriority.URGENT)
        self.assertTrue(any("arithmetic" in r.lower() for r in res.queue_item.reasons))


if __name__ == "__main__":
    unittest.main()
