"""Unit tests for document-level composite confidence synthesis and risk aggregation."""

import unittest
from pathlib import Path

from src.core.config import DocumentConfidenceConfig
from src.core.models import Document, DocumentMetadata, DocumentPage
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.confidence.document_confidence import DocumentConfidenceCalculator
from src.confidence.models import (
    ConfidenceBand,
    DocumentConfidence,
    FieldConfidence,
    TableConfidence,
)
from src.validation.models import ValidationIssue, ValidationReport


class TestDocumentConfidence(unittest.TestCase):
    """Test suite for document-level composite confidence synthesis."""

    def setUp(self) -> None:
        self.config = DocumentConfidenceConfig(
            classification_weight=0.20,
            field_weight=0.40,
            table_weight=0.20,
            validation_weight=0.20,
            high_threshold=0.85,
            medium_threshold=0.65,
            low_threshold=0.40,
            review_confidence_threshold=0.70,
        )
        self.calc = DocumentConfidenceCalculator(config=self.config)

    def _create_sample_doc(self, doc_type: DocumentType = DocumentType.INVOICE, clf_conf: float = 0.95) -> Document:
        meta = DocumentMetadata(
            document_id="doc_test_01",
            filename="invoice.pdf",
            file_path=Path("data/invoice.pdf"),
            file_type="application/pdf",
            file_size_bytes=2048,
            checksum_sha256="sha256_mock",
            page_count=1,
        )
        page = DocumentPage(page_number=1, raw_text="Sample text")
        return Document(
            metadata=meta,
            pages=[page],
            classified_type=doc_type,
            classification_confidence=clf_conf,
        )

    def test_strong_document_high_composite_score(self) -> None:
        """Verify high composite score and STP eligibility when all subsystems succeed."""
        doc = self._create_sample_doc(DocumentType.INVOICE, 0.95)

        fields = {
            "invoice_number": FieldConfidence(
                field_name="invoice_number",
                value="INV-001",
                original_extraction_confidence=0.96,
                aggregated_confidence=0.96,
                confidence_band=ConfidenceBand.HIGH,
            ),
            "total": FieldConfidence(
                field_name="total",
                value="$500.00",
                original_extraction_confidence=0.94,
                aggregated_confidence=0.94,
                confidence_band=ConfidenceBand.HIGH,
            ),
        }

        tables = [
            TableConfidence(
                table_id="tbl_1",
                structure_confidence=0.92,
                row_confidence=0.95,
                cell_confidence=0.92,
                aggregated_confidence=0.93,
                confidence_band=ConfidenceBand.HIGH,
            )
        ]

        report = ValidationReport(
            document_id=doc.id,
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.VALID,
            validation_score=1.0,
        )

        doc_conf = self.calc.calculate(
            document=doc,
            field_confidences=fields,
            table_confidences=tables,
            validation_report=report,
        )

        # Expected: 0.95*0.2 + 0.95*0.4 + 0.93*0.2 + 1.0*0.2 = 0.19 + 0.38 + 0.186 + 0.20 = 0.956
        self.assertGreater(doc_conf.overall_confidence, 0.90)
        self.assertEqual(doc_conf.confidence_band, ConfidenceBand.HIGH)
        self.assertFalse(doc_conf.review_required)

    def test_critical_validation_failure_overrides_high_confidence(self) -> None:
        """CRITICAL TEST: Verify high confidence document with validation error still triggers review."""
        doc = self._create_sample_doc(DocumentType.INVOICE, 0.98)

        fields = {
            "total": FieldConfidence(
                field_name="total",
                value="$1,000.00",
                original_extraction_confidence=0.98,
                aggregated_confidence=0.98,
                confidence_band=ConfidenceBand.HIGH,
            )
        }

        # Validation report has blocking error (e.g. subtotal + tax != total)
        issue = ValidationIssue(
            rule_id="RULE_SUBTOTAL_TAX_TOTAL",
            rule_name="Financial Reconciliation Check",
            status=ValidationStatus.INVALID,
            severity=SeverityLevel.CRITICAL,
            message="Subtotal ($800) + Tax ($100) != Total ($1000)",
            affected_fields=["subtotal", "tax", "total"],
        )
        report = ValidationReport(
            document_id=doc.id,
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.INVALID,
            validation_score=0.75,
            issues=[issue],
        )

        doc_conf = self.calc.calculate(
            document=doc,
            field_confidences=fields,
            table_confidences=[],
            validation_report=report,
        )

        # Overall confidence is still moderately high
        self.assertGreater(doc_conf.overall_confidence, 0.80)
        # BUT review MUST be required due to validation error!
        self.assertTrue(doc_conf.review_required)
        self.assertTrue(any("validation" in r.lower() for r in doc_conf.review_reasons))

    def test_unclassified_document_triggers_review(self) -> None:
        """Verify UNKNOWN document type triggers review."""
        doc = self._create_sample_doc(DocumentType.UNKNOWN, 0.30)
        doc_conf = self.calc.calculate(document=doc, field_confidences={}, table_confidences=[])

        self.assertTrue(doc_conf.review_required)
        self.assertTrue(any("unknown" in r.lower() for r in doc_conf.review_reasons))


if __name__ == "__main__":
    unittest.main()
