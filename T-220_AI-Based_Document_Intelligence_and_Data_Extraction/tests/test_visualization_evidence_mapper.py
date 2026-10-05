"""Unit tests for EvidenceMapper compiling pipeline artifacts into spatial evidence and manifests."""

import unittest
from pathlib import Path

from src.core.models import BoundingBox, Document, DocumentMetadata, DocumentPage, ExtractedField, Provenance
from src.core.types import DocumentType, ExtractionMethod, FieldType, SeverityLevel, ValidationStatus
from src.confidence.models import (
    ConfidenceBand,
    DocumentConfidence,
    FieldConfidence,
    ReviewPriority,
    ReviewQueueItem,
    ReviewRoutingResult,
    ReviewStatus,
    ReviewTarget,
    ReviewTargetType,
)
from src.extraction.models import EntityType, ExtractedEntity
from src.ocr.models import OCRWord
from src.tables.models import LineItem, Table, TableCell, TableColumn, TableRow, TableValueType
from src.validation.models import ValidationCategory, ValidationIssue, ValidationReport
from src.visualization.evidence_mapper import EvidenceMapper
from src.visualization.models import AnnotationType


class TestEvidenceMapper(unittest.TestCase):
    """Test suite for EvidenceMapper consolidation."""

    def setUp(self) -> None:
        self.doc_id = "doc_test_mapper_01"
        self.bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=250.0, ymax=140.0)
        self.meta = DocumentMetadata(
            document_id=self.doc_id,
            filename="sample_invoice.pdf",
            file_path=Path("data/sample_invoice.pdf"),
            file_type="application/pdf",
            file_size_bytes=1024,
            checksum_sha256="dummy_sha",
            page_count=1,
        )
        self.page = DocumentPage(
            page_number=1,
            width=1000.0,
            height=1400.0,
            raw_text="Invoice INV-001 Total $500.00",
            metadata={
                "ocr_words": [
                    OCRWord(text="Invoice", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(50, 100, 120, 140), page_number=1),
                    OCRWord(text="INV-001", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(130, 100, 220, 140), page_number=1),
                ]
            },
        )
        self.document = Document(metadata=self.meta, pages=[self.page])

    def test_map_ocr_and_field_evidence(self) -> None:
        """Verify OCR words and spatial extracted fields are mapped correctly."""
        fields = {
            "invoice_number": ExtractedField(
                name="invoice_number",
                value="INV-001",
                normalized_value="INV-001",
                field_type=FieldType.IDENTIFIER,
                provenance=Provenance(self.doc_id, 1, "INV-001", self.bbox, 0.97, ExtractionMethod.REGEX_PATTERN),
                extraction_confidence=0.95,
                validation_status=ValidationStatus.VALID,
            ),
            "tax_id": ExtractedField(
                name="tax_id",
                value="N/A",
                normalized_value=None,
                field_type=FieldType.IDENTIFIER,
                provenance=None,  # Unlocated field
                extraction_confidence=0.30,
            ),
        }

        mapper = EvidenceMapper(document_id=self.doc_id)
        manifest, anns_by_page = mapper.map_all(
            document=self.document,
            fields=fields,
        )

        self.assertEqual(manifest.document_id, self.doc_id)
        # 2 OCR words + 1 spatial field = 3 spatial regions
        self.assertEqual(len(manifest.evidence_regions), 3)
        # 1 unlocated field (tax_id)
        self.assertEqual(len(manifest.unlocated_evidence), 1)
        self.assertEqual(manifest.unlocated_evidence[0].field_name, "tax_id")
        self.assertFalse(manifest.unlocated_evidence[0].is_spatial)

        # Verify page 1 annotations
        self.assertIn(1, anns_by_page)
        fld_anns = [a for a in anns_by_page[1] if a.annotation_type == AnnotationType.EXTRACTED_FIELD]
        self.assertEqual(len(fld_anns), 1)
        self.assertEqual(fld_anns[0].field_name, "invoice_number")

    def test_map_table_and_validation_issues(self) -> None:
        """Verify table cells, line-item math failures, and validation issues are mapped."""
        table = Table(
            table_id="tbl_1",
            page_number=1,
            headers=["Description", "Amount"],
            columns=[
                TableColumn(0, "Description", "description", 50, 300, "left", TableValueType.TEXT),
                TableColumn(1, "Amount", "amount", 300, 500, "right", TableValueType.CURRENCY),
            ],
            rows=[],
            line_items=[
                LineItem(
                    row_index=0,
                    description="Consulting",
                    quantity=2.0,
                    unit_price=100.0,
                    amount=250.0,  # Math mismatch
                    page_number=1,
                    confidence=0.70,
                    is_valid_arithmetic=False,
                )
            ],
            bounding_box=BoundingBox(50, 300, 500, 500),
            confidence=0.85,
        )

        val_report = ValidationReport(
            document_id=self.doc_id,
            overall_status=ValidationStatus.INVALID,
            issues=[
                ValidationIssue(
                    rule_id="RULE_LINE_MATH",
                    rule_name="Line Math Check",
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.CRITICAL,
                    message="2 * 100 != 250",
                    affected_fields=["amount"],
                    bounding_box=BoundingBox(50, 300, 500, 500),
                    page_number=1,
                )
            ],
        )

        mapper = EvidenceMapper(document_id=self.doc_id)
        manifest, anns_by_page = mapper.map_all(
            document=self.document,
            tables=[table],
            validation_report=val_report,
        )

        val_regions = [r for r in manifest.evidence_regions if r.region_type == AnnotationType.VALIDATION_CRITICAL]
        self.assertEqual(len(val_regions), 1)
        self.assertEqual(val_regions[0].validation_status, ValidationStatus.INVALID)

        # Verify review annotations generated
        rev_anns = [a for a in anns_by_page[1] if a.annotation_type == AnnotationType.VALIDATION_CRITICAL]
        self.assertEqual(len(rev_anns), 1)
        self.assertEqual(rev_anns[0].severity, SeverityLevel.CRITICAL)


if __name__ == "__main__":
    unittest.main()
