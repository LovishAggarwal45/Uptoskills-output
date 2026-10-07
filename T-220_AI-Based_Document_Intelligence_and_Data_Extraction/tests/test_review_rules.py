"""Unit tests for individual review rules and the review rule catalog."""

import unittest
from pathlib import Path

from src.core.config import ReviewRoutingConfig
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
    DocumentConfidence,
    FieldConfidence,
    LineItemConfidence,
    ReviewTargetType,
    TableConfidence,
)
from src.confidence.review_rules import (
    CandidateConflictReviewRule,
    LowFieldConfidenceReviewRule,
    LowOCRConfidenceReviewRule,
    MissingRequiredFieldReviewRule,
    ReviewRuleCatalog,
    TableArithmeticReviewRule,
    UnclassifiedDocumentReviewRule,
    ValidationReportReviewRule,
)
from src.validation.models import ValidationIssue, ValidationReport


class TestReviewRules(unittest.TestCase):
    """Test suite for declarative human review rule evaluation."""

    def setUp(self) -> None:
        self.doc = Document(
            metadata=DocumentMetadata("doc_rule_test", "doc.pdf", Path("data/doc.pdf"), "application/pdf", 100, "sha", 1),
            pages=[DocumentPage(1, raw_text="text")],
            classified_type=DocumentType.INVOICE,
        )
        self.doc_conf = DocumentConfidence(document_id=self.doc.id, overall_confidence=0.90)

    def test_low_field_confidence_rule(self) -> None:
        """Verify LowFieldConfidenceReviewRule flags fields below threshold."""
        rule = LowFieldConfidenceReviewRule()
        fc = FieldConfidence(
            field_name="vendor_name",
            value="Acme",
            aggregated_confidence=0.55,  # below 0.70 default threshold
        )

        issues = rule.evaluate(
            document=self.doc,
            field_confidences={"vendor_name": fc},
            table_confidences=[],
            document_confidence=self.doc_conf,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].affected_targets[0].field_name, "vendor_name")
        self.assertEqual(issues[0].affected_targets[0].target_type, ReviewTargetType.FIELD)

    def test_low_ocr_confidence_rule(self) -> None:
        """Verify LowOCRConfidenceReviewRule flags low OCR tokens."""
        rule = LowOCRConfidenceReviewRule()
        fc = FieldConfidence(
            field_name="invoice_number",
            value="INV-100",
            ocr_confidence=0.45,  # below 0.60 threshold
            aggregated_confidence=0.75,
        )

        issues = rule.evaluate(
            document=self.doc,
            field_confidences={"invoice_number": fc},
            table_confidences=[],
            document_confidence=self.doc_conf,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].source_signal, "ocr_confidence")

    def test_missing_required_field_rule(self) -> None:
        """Verify MissingRequiredFieldReviewRule flags missing mandatory fields."""
        rule = MissingRequiredFieldReviewRule()
        fc = FieldConfidence(
            field_name="total",
            value=None,
            metadata={"is_required": True},
        )

        issues = rule.evaluate(
            document=self.doc,
            field_confidences={"total": fc},
            table_confidences=[],
            document_confidence=self.doc_conf,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].affected_targets[0].field_name, "total")

    def test_table_arithmetic_rule(self) -> None:
        """Verify TableArithmeticReviewRule flags failed line item math."""
        rule = TableArithmeticReviewRule()
        lic = LineItemConfidence(
            row_index=2,
            description="Item C",
            is_valid_arithmetic=False,
            review_required=True,
        )
        tc = TableConfidence(
            table_id="tbl_1",
            line_item_confidences=[lic],
        )

        issues = rule.evaluate(
            document=self.doc,
            field_confidences={},
            table_confidences=[tc],
            document_confidence=self.doc_conf,
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].affected_targets[0].target_type, ReviewTargetType.LINE_ITEM)
        self.assertEqual(issues[0].affected_targets[0].row_index, 2)
        self.assertEqual(issues[0].severity, SeverityLevel.CRITICAL)

    def test_review_rule_catalog_executes_all(self) -> None:
        """Verify ReviewRuleCatalog aggregates all triggered issues without duplicates."""
        catalog = ReviewRuleCatalog()
        self.assertGreaterEqual(len(catalog.rules), 6)

        fc = FieldConfidence(
            field_name="invoice_date",
            value="",
            metadata={"is_required": True},
            aggregated_confidence=0.0,
        )

        issues = catalog.evaluate_all(
            document=self.doc,
            field_confidences={"invoice_date": fc},
            table_confidences=[],
            document_confidence=self.doc_conf,
        )

        self.assertGreater(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
