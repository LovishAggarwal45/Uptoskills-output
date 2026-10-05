"""Unit tests for confidence signal extraction from upstream domain models."""

import unittest
from pathlib import Path

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
from src.confidence.signals import (
    extract_document_signals,
    extract_field_signals,
    extract_table_signals,
)
from src.tables.models import LineItem, Table
from src.validation.models import ValidationIssue, ValidationReport


class TestConfidenceSignals(unittest.TestCase):
    """Test suite for extracting signals without hallucinating or fabricating missing values."""

    def test_extract_field_signals_full_provenance(self) -> None:
        """Extract signals when OCR confidence, extraction confidence, and validation are all present."""
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance("doc_1", 1, "INV-2026-001", bbox, 0.98, ExtractionMethod.REGEX_PATTERN)

        field = ExtractedField(
            name="invoice_number",
            value="INV-2026-001",
            normalized_value="INV-2026-001",
            field_type=FieldType.IDENTIFIER,
            provenance=prov,
            extraction_confidence=0.95,
            validation_status=ValidationStatus.VALID,
            is_required=True,
        )

        signals = extract_field_signals(field)

        self.assertEqual(signals.ocr_confidence, 0.98)
        self.assertEqual(signals.extraction_confidence, 0.95)
        self.assertEqual(signals.validation_status, ValidationStatus.VALID)
        self.assertEqual(signals.conflict_count, 0)
        self.assertFalse(signals.required_field_missing)

    def test_extract_field_signals_missing_ocr_is_none(self) -> None:
        """Verify that when OCR confidence is unavailable, signal is explicitly None (not 0.5/0.8)."""
        field = ExtractedField(
            name="vendor_name",
            value="Acme Corp",
            normalized_value="Acme Corp",
            field_type=FieldType.ORGANIZATION,
            provenance=None,  # No provenance
            extraction_confidence=0.85,
        )

        signals = extract_field_signals(field)

        self.assertIsNone(signals.ocr_confidence)
        self.assertEqual(signals.extraction_confidence, 0.85)

    def test_extract_field_signals_with_candidates_conflict(self) -> None:
        """Verify candidate conflicts are detected when multiple distinct values exist."""
        field = ExtractedField(
            name="total",
            value="$450.00",
            normalized_value=450.00,
            extraction_confidence=0.80,
            candidates=[
                {"value": "$450.00", "normalized_value": 450.00},
                {"value": "$500.00", "normalized_value": 500.00},
            ],
        )

        signals = extract_field_signals(field)
        self.assertEqual(signals.conflict_count, 1)

    def test_extract_field_signals_missing_required_field(self) -> None:
        """Verify missing required field sets required_field_missing=True."""
        field = ExtractedField(
            name="invoice_date",
            value="",
            normalized_value=None,
            is_required=True,
            extraction_confidence=0.0,
        )

        signals = extract_field_signals(field)
        self.assertTrue(signals.required_field_missing)

    def test_extract_table_signals(self) -> None:
        """Verify table signals extraction."""
        table = Table(
            table_id="tbl_1",
            page_number=1,
            headers=["Item", "Amount"],
            confidence=0.92,
            confidence_breakdown={"structure_confidence": 0.90, "row_confidence": 0.95, "cell_confidence": 0.92},
            validation_status=ValidationStatus.VALID,
        )

        signals = extract_table_signals(table)
        self.assertEqual(signals.table_confidence, 0.92)
        self.assertEqual(signals.validation_status, ValidationStatus.VALID)
        self.assertEqual(signals.metadata["structure_confidence"], 0.90)

    def test_extract_document_signals(self) -> None:
        """Verify document-level signals synthesis."""
        meta = DocumentMetadata(
            document_id="doc_signals_test",
            filename="doc.pdf",
            file_path=Path("data/doc.pdf"),
            file_type="application/pdf",
            file_size_bytes=1000,
            checksum_sha256="sha",
            page_count=1,
        )
        page = DocumentPage(
            page_number=1,
            raw_text="Sample text",
            metadata={"quality_metrics": {"overall_quality": 0.95}},
        )
        doc = Document(
            metadata=meta,
            pages=[page],
            classified_type=DocumentType.INVOICE,
            classification_confidence=0.96,
        )

        report = ValidationReport(
            document_id=doc.id,
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.VALID,
            validation_score=0.98,
        )

        signals = extract_document_signals(doc, fields={}, tables=[], validation_report=report)

        self.assertEqual(signals.classification_confidence, 0.96)
        self.assertEqual(signals.validation_status, ValidationStatus.VALID)
        self.assertEqual(signals.validation_score, 0.98)
        self.assertEqual(signals.source_quality_score, 0.95)


if __name__ == "__main__":
    unittest.main()
