"""Tests for Phase 10 Document Exporter subsystem."""

import csv
import io
import json
import unittest
import zipfile

from src.core.models import (
    BoundingBox,
    ExtractedField,
    Provenance,
)
from src.core.types import (
    DocumentType,
    SeverityLevel,
    ValidationStatus,
)
from src.classification.models import ClassificationResult as DocumentClassificationResult
from src.extraction.models import ExtractionResult
from src.tables.models import (
    LineItem as TableLineItem,
    Table,
    TableCell,
    TableColumn,
    TableExtractionResult,
    TableRow,
    TableValueType,
)
from src.validation.models import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from src.confidence.models import (
    ConfidenceBand,
    DocumentConfidence,
    FieldConfidence,
    ReviewPriority,
    ReviewQueueItem,
    ReviewStatus,
)
from src.export.exporter import DocumentExporter
from src.persistence.models import CorrectionRecord, DecisionRecord, ReviewAction


class TestDocumentExporter(unittest.TestCase):
    """Test suite for DocumentExporter formats."""

    def setUp(self) -> None:
        self.exporter = DocumentExporter()

        self.classification = DocumentClassificationResult(
            document_type=DocumentType.INVOICE,
            confidence=0.95,
            evidence=[],
        )

        bbox = BoundingBox(xmin=10, ymin=20, xmax=100, ymax=40)
        prov = Provenance(document_id="doc-exp-1", page_number=1, raw_text="$1,500.00", bounding_box=bbox, ocr_confidence=0.98)

        self.extraction = ExtractionResult(
            document_id="doc-exp-1",
            document_type=DocumentType.INVOICE,
            fields={
                "total_amount": ExtractedField(
                    name="total_amount",
                    value=1500.0,
                    normalized_value=1500.0,
                    provenance=prov,
                    extraction_confidence=0.96,
                    validation_status=ValidationStatus.VALID,
                ),
                "invoice_number": ExtractedField(
                    name="invoice_number",
                    value="INV-2026-99",
                    normalized_value="INV-2026-99",
                    provenance=prov,
                    extraction_confidence=0.99,
                    validation_status=ValidationStatus.VALID,
                ),
            },
            entities=[],
            metadata={"currency": "USD"},
        )

        col1 = TableColumn(index=0, name="Description", x_start=10, x_end=150, alignment="left", inferred_type=TableValueType.TEXT)
        col2 = TableColumn(index=1, name="Quantity", x_start=160, x_end=200, alignment="right", inferred_type=TableValueType.INTEGER)
        col3 = TableColumn(index=2, name="Amount", x_start=210, x_end=280, alignment="right", inferred_type=TableValueType.CURRENCY)

        cell1 = TableCell(row_index=0, col_index=0, text="Cloud Hosting", confidence=0.95, page_number=1)
        cell2 = TableCell(row_index=0, col_index=1, text="2", confidence=0.95, page_number=1)
        cell3 = TableCell(row_index=0, col_index=2, text="1500.00", confidence=0.95, page_number=1)

        row = TableRow(cells=[cell1, cell2, cell3], row_index=0, is_header=False, confidence=0.95)
        line_item = TableLineItem(
            row_index=0,
            page_number=1,
            description="Cloud Hosting",
            quantity=2.0,
            unit_price=750.0,
            amount=1500.0,
            confidence=0.95,
            is_valid_arithmetic=True,
        )

        table = Table(
            table_id="table-1",
            page_number=1,
            headers=["Description", "Quantity", "Amount"],
            columns=[col1, col2, col3],
            rows=[row],
            line_items=[line_item],
            confidence=0.95,
        )

        self.tables = TableExtractionResult(document_id="doc-exp-1", tables=[table], metadata={"total_tables": 1})

        self.validation = ValidationReport(
            document_id="doc-exp-1",
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.VALID,
            validation_score=1.0,
            rules_evaluated=5,
            rules_passed=5,
            issues=[],
        )

        self.confidence = DocumentConfidence(
            document_id="doc-exp-1",
            overall_confidence=0.94,
            confidence_band=ConfidenceBand.HIGH,
            classification_confidence=0.95,
            mean_field_confidence=0.975,
            mean_table_confidence=0.95,
            validation_score=1.0,
            review_required=False,
            field_confidences={
                "total_amount": FieldConfidence(
                    field_name="total_amount",
                    value=1500.0,
                    aggregated_confidence=0.96,
                    confidence_band=ConfidenceBand.HIGH,
                    ocr_confidence=0.98,
                    original_extraction_confidence=0.96,
                    review_required=False,
                )
            },
            component_breakdown={"ocr_quality": 0.98, "extraction_pattern": 0.96},
        )

        self.review_queue = ReviewQueueItem(
            document_id="doc-exp-1",
            status=ReviewStatus.PENDING,
            priority=ReviewPriority.LOW,
            targets=[],
            issues=[],
            review_required=False,
        )

    def test_export_json_payload(self) -> None:
        """Test export_json creates complete serializable dictionary."""
        data = self.exporter.export_json(
            document_id="doc-exp-1",
            filename="invoice.pdf",
            classification=self.classification,
            extraction=self.extraction,
            tables=self.tables,
            validation=self.validation,
            confidence=self.confidence,
            review_queue=self.review_queue,
        )

        self.assertEqual(data["document_id"], "doc-exp-1")
        self.assertEqual(data["document_type"], "invoice")
        self.assertEqual(data["fields"]["total_amount"]["value"], 1500.0)
        self.assertEqual(len(data["tables"]["tables"]), 1)
        self.assertEqual(data["confidence"]["confidence_band"], "high")

        # Verify JSON round-trip
        json_str = json.dumps(data)
        self.assertIn("INV-2026-99", json_str)

    def test_export_reviewed_json_with_corrections(self) -> None:
        """Test export_reviewed_json applies human overrides and includes decision history."""
        corrections = [
            CorrectionRecord(
                correction_id="corr-1",
                document_id="doc-exp-1",
                field_name="total_amount",
                original_value="1500.0",
                corrected_value="1550.0",
                reviewer_id="reviewer_lead",
                reason="Included missing shipping surcharge",
            )
        ]
        decisions = [
            DecisionRecord(
                decision_id="dec-1",
                document_id="doc-exp-1",
                action=ReviewAction.APPROVE,
                reviewer_id="reviewer_lead",
                notes="Approved with adjusted shipping.",
            )
        ]

        data = self.exporter.export_reviewed_json(
            document_id="doc-exp-1",
            filename="invoice.pdf",
            classification=self.classification,
            extraction=self.extraction,
            tables=self.tables,
            validation=self.validation,
            confidence=self.confidence,
            corrections=corrections,
            decisions=decisions,
        )

        field = data["fields"]["total_amount"]
        self.assertEqual(field["value"], "1550.0")
        self.assertTrue(field["is_human_corrected"])
        self.assertEqual(field["corrected_by"], "reviewer_lead")
        self.assertEqual(field["original_extracted_value"], 1500.0)
        self.assertEqual(len(data["review_decisions"]), 1)
        self.assertEqual(data["review_decisions"][0]["action"], "approve")

    def test_export_csv_formats(self) -> None:
        """Test export_csv returns CSV text or ZIP archive of CSV files."""
        # Single table CSV
        csv_bytes, mime, fn = self.exporter.export_csv("doc-exp-1", self.tables, self.extraction)
        self.assertIn("text/csv", mime)
        self.assertTrue(fn.endswith(".csv"))

        content = csv_bytes.decode("utf-8")
        self.assertIn("Cloud Hosting", content)
        self.assertIn("1500.00", content)

    def test_export_validation_report_markdown(self) -> None:
        """Test export_validation_report generates comprehensive Markdown summary."""
        issue = ValidationIssue(
            rule_id="VAL-TOTAL-1",
            rule_name="Total Match Check",
            status=ValidationStatus.WARNING,
            severity=ValidationSeverity.WARNING,
            message="Line items sum does not equal subtotal.",
            affected_fields=["total_amount", "subtotal"],
        )
        val_with_issue = ValidationReport(
            document_id="doc-exp-1",
            document_type="invoice",
            overall_status=ValidationStatus.WARNING,
            validation_score=0.8,
            rules_evaluated=5,
            rules_passed=4,
            issues=[issue],
        )

        md = self.exporter.export_validation_report(
            document_id="doc-exp-1",
            filename="invoice.pdf",
            validation=val_with_issue,
            confidence=self.confidence,
            review_queue=self.review_queue,
        )

        self.assertIn("# DocuMind AI — Document Validation & Integrity Report", md)
        self.assertIn("VAL-TOTAL-1", md)
        self.assertIn("Line items sum does not equal subtotal.", md)
        self.assertIn("Overall Status | `WARNING`", md)


if __name__ == "__main__":
    unittest.main()
