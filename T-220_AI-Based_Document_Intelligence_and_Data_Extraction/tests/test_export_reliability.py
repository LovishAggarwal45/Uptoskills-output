"""Unit and integration tests for export reliability in DocuMind AI."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.core.config import StorageConfig
from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    ExtractedField,
    ProcessingResult,
    ProcessingSummary,
    Provenance,
)
from src.core.types import DocumentType, FieldType, ValidationStatus
from src.export.exporter import DocumentExporter
from src.persistence.models import CorrectionRecord, DecisionRecord


class TestExportReliability(unittest.TestCase):
    """Test full JSON, Reviewed JSON, CSV, and Markdown export functionality."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.json_dir = self.temp_dir / "json"
        self.csv_dir = self.temp_dir / "csv"
        self.reports_dir = self.temp_dir / "reports"

        self.config = StorageConfig(
            output_json_dir=str(self.json_dir),
            output_csv_dir=str(self.csv_dir),
            output_reports_dir=str(self.reports_dir),
        )
        self.exporter = DocumentExporter(config=self.config)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_result(self, doc_id: str = "doc_exp_test") -> ProcessingResult:
        page = DocumentPage(page_number=1, width=800.0, height=1000.0, dpi=300, raw_text="Sample text")
        meta = DocumentMetadata(
            document_id=doc_id,
            filename="sample_invoice.pdf",
            file_path="sample_invoice.pdf",
            file_type="application/pdf",
            file_size_bytes=12345,
            checksum_sha256="sha256sample",
            page_count=1,
        )
        doc = Document(metadata=meta, pages=[page], classified_type=DocumentType.INVOICE)

        prov = Provenance(document_id=doc_id, page_number=1, raw_text="INV-1001", ocr_confidence=0.98)
        field_inv = ExtractedField(
            name="invoice_number",
            value="INV-1001",
            normalized_value="INV-1001",
            field_type=FieldType.IDENTIFIER,
            provenance=prov,
            extraction_confidence=0.98,
            validation_status=ValidationStatus.VALID,
        )
        field_tot = ExtractedField(
            name="total",
            value="$1,200.00",
            normalized_value=1200.0,
            field_type=FieldType.CURRENCY,
            provenance=prov,
            extraction_confidence=0.95,
            validation_status=ValidationStatus.VALID,
        )

        summary = ProcessingSummary(
            document_id=doc_id,
            document_type=DocumentType.INVOICE,
            total_pages=1,
            total_fields_extracted=2,
            total_tables_extracted=0,
            validation_passed_count=5,
            validation_warning_count=0,
            validation_error_count=0,
            overall_confidence_score=0.96,
            review_required=False,
            processing_time_seconds=0.45,
        )

        return ProcessingResult(
            document=doc,
            fields={"invoice_number": field_inv, "total": field_tot},
            tables=[],
            validation_results=[],
            review_flags=[],
            summary=summary,
        )

    def test_export_json_structure(self) -> None:
        """Verify export_json writes complete, valid JSON containing metadata and fields."""
        result = self._create_mock_result("doc_json_01")
        dest_path = self.exporter.export_json(result)

        self.assertTrue(Path(dest_path).exists())
        with open(dest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("document", data)
        self.assertIn("fields", data)
        self.assertIn("invoice_number", data["fields"])
        self.assertEqual(data["fields"]["invoice_number"]["value"], "INV-1001")
        self.assertIn("summary", data)

    def test_export_reviewed_json_separates_corrections(self) -> None:
        """Verify export_reviewed_json preserves original extracted value and logs reviewer corrections."""
        result = self._create_mock_result("doc_rev_01")

        corr = CorrectionRecord(
            correction_id="corr_01",
            document_id="doc_rev_01",
            field_name="total",
            original_value="$1,200.00",
            corrected_value="$1,250.00",
            reviewer_id="reviewer_alice",
            reason="Corrected OCR digit transposition",
            created_at="2026-10-03T10:00:00Z",
            is_applied=True,
        )

        dest_path = self.exporter.export_reviewed_json(
            result=result,
            corrections=[corr],
            audit_trail=[{"event_type": "FIELD_CORRECTED", "actor": "reviewer_alice"}],
        )

        self.assertTrue(Path(dest_path).exists())
        with open(dest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertTrue(data.get("is_reviewed"))
        self.assertIn("corrections_applied", data)
        self.assertEqual(len(data["corrections_applied"]), 1)

        # Check field in export
        total_field = data["fields"]["total"]
        self.assertEqual(total_field["original_extracted_value"], "$1,200.00")
        self.assertEqual(total_field["value"], "$1,250.00")
        self.assertTrue(total_field.get("is_human_corrected"))
        self.assertEqual(total_field.get("corrected_by"), "reviewer_alice")

    def test_export_csv_fields_and_tables(self) -> None:
        """Verify export_csv creates both fields.csv and table CSVs."""
        result = self._create_mock_result("doc_csv_01")
        paths = self.exporter.export_csv(result)

        self.assertGreater(len(paths), 0)
        fields_csv = [p for p in paths if p.name.endswith("_fields.csv")]
        self.assertEqual(len(fields_csv), 1)

        content = fields_csv[0].read_text(encoding="utf-8")
        self.assertIn("Field Name,Value,Normalized Value", content)
        self.assertIn("invoice_number,INV-1001", content)
        self.assertIn('total,"$1,200.00"', content)

    def test_export_validation_report_markdown(self) -> None:
        """Verify export_validation_report generates Markdown report with Executive Summary."""
        result = self._create_mock_result("doc_rep_01")
        report_path = self.exporter.export_validation_report(result)

        self.assertTrue(Path(report_path).exists())
        content = Path(report_path).read_text(encoding="utf-8")
        self.assertIn("# DocuMind AI — Document Validation & Integrity Report", content)
        self.assertIn("## 1. Executive Summary", content)
        self.assertIn("Overall Status", content)
        self.assertIn("doc_rep_01", content)


if __name__ == "__main__":
    unittest.main()
