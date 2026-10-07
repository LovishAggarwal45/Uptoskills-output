"""Unit tests for DocuMind AI domain models, geometry, and serialization."""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    ExtractedField,
    ExtractedTable,
    OCRTextRegion,
    ProcessingResult,
    ProcessingSummary,
    Provenance,
    ReviewFlag,
    TableCell,
    ValidationResult,
)
from src.core.types import (
    ConfidenceSource,
    DocumentType,
    ExtractionMethod,
    FieldType,
    ReviewTriggerType,
    SeverityLevel,
    ValidationStatus,
)


class TestBoundingBox(unittest.TestCase):
    """Test suite for 2D BoundingBox geometry and coordinate transformations."""

    def test_valid_bounding_box_creation(self) -> None:
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=110.0, ymax=120.0)
        self.assertEqual(bbox.width, 100.0)
        self.assertEqual(bbox.height, 100.0)
        self.assertEqual(bbox.area, 10000.0)
        self.assertFalse(bbox.is_normalized)

    def test_invalid_coordinates_raise_value_error(self) -> None:
        # xmin > xmax
        with self.assertRaises(ValueError):
            BoundingBox(xmin=150.0, ymin=10.0, xmax=100.0, ymax=100.0)

        # ymin > ymax
        with self.assertRaises(ValueError):
            BoundingBox(xmin=10.0, ymin=200.0, xmax=100.0, ymax=50.0)

    def test_normalized_coordinate_bounds(self) -> None:
        bbox = BoundingBox(xmin=0.1, ymin=0.2, xmax=0.8, ymax=0.9, is_normalized=True)
        self.assertTrue(bbox.is_normalized)

        # Out of bounds (> 1.0)
        with self.assertRaises(ValueError):
            BoundingBox(xmin=0.1, ymin=0.2, xmax=1.5, ymax=0.9, is_normalized=True)

    def test_normalization_and_denormalization(self) -> None:
        page_width, page_height = 1000.0, 2000.0
        bbox = BoundingBox(xmin=100.0, ymin=200.0, xmax=500.0, ymax=800.0)

        normalized = bbox.normalize(page_width, page_height)
        self.assertTrue(normalized.is_normalized)
        self.assertAlmostEqual(normalized.xmin, 0.1)
        self.assertAlmostEqual(normalized.ymin, 0.1)
        self.assertAlmostEqual(normalized.xmax, 0.5)
        self.assertAlmostEqual(normalized.ymax, 0.4)

        denormalized = normalized.denormalize(page_width, page_height)
        self.assertFalse(denormalized.is_normalized)
        self.assertAlmostEqual(denormalized.xmin, 100.0)
        self.assertAlmostEqual(denormalized.ymin, 200.0)
        self.assertAlmostEqual(denormalized.xmax, 500.0)
        self.assertAlmostEqual(denormalized.ymax, 800.0)

    def test_intersection_over_union(self) -> None:
        box1 = BoundingBox(xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0)  # Area = 100
        box2 = BoundingBox(xmin=5.0, ymin=0.0, xmax=15.0, ymax=10.0)  # Area = 100
        # Intersection = [5,0] to [10,10] => Area = 50
        # Union = 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3 ~ 0.3333
        iou = box1.intersection_over_union(box2)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)

        # Disjoint boxes
        box3 = BoundingBox(xmin=20.0, ymin=20.0, xmax=30.0, ymax=30.0)
        self.assertEqual(box1.intersection_over_union(box3), 0.0)

    def test_dict_serialization_roundtrip(self) -> None:
        box = BoundingBox(xmin=12.5, ymin=34.2, xmax=150.8, ymax=200.4)
        box_dict = box.to_dict()
        reconstructed = BoundingBox.from_dict(box_dict)
        self.assertEqual(box, reconstructed)


class TestProvenanceAndFields(unittest.TestCase):
    """Test suite for Provenance metadata, ExtractedField, and validation."""

    def test_provenance_validation(self) -> None:
        bbox = BoundingBox(xmin=10.0, ymin=10.0, xmax=50.0, ymax=20.0)
        prov = Provenance(
            document_id="doc_101",
            page_number=1,
            raw_text="Invoice #12345",
            bounding_box=bbox,
            ocr_confidence=0.96,
            extraction_method=ExtractionMethod.REGEX_PATTERN,
        )
        self.assertEqual(prov.page_number, 1)
        self.assertEqual(prov.ocr_confidence, 0.96)

        # Negative page number
        with self.assertRaises(ValueError):
            Provenance(document_id="doc_101", page_number=0, raw_text="test")

        # Invalid OCR confidence > 1.0
        with self.assertRaises(ValueError):
            Provenance(document_id="doc_101", page_number=1, raw_text="test", ocr_confidence=1.5)

    def test_extracted_field_serialization(self) -> None:
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance(
            document_id="doc_test_1",
            page_number=1,
            raw_text="Acme Corp",
            bounding_box=bbox,
            ocr_confidence=0.99,
            extraction_method=ExtractionMethod.KEY_VALUE_HEURISTIC,
        )
        field = ExtractedField(
            name="vendor_name",
            value="Acme Corp",
            normalized_value="ACME CORP",
            field_type=FieldType.ORGANIZATION,
            provenance=prov,
            extraction_confidence=0.92,
            confidence_source=ConfidenceSource.RULE_SCORE,
            validation_status=ValidationStatus.VALID,
            is_required=True,
        )

        field_dict = field.to_dict()
        reconstructed = ExtractedField.from_dict(field_dict)
        self.assertEqual(reconstructed.name, "vendor_name")
        self.assertEqual(reconstructed.value, "Acme Corp")
        self.assertEqual(reconstructed.normalized_value, "ACME CORP")
        self.assertEqual(reconstructed.field_type, FieldType.ORGANIZATION)
        self.assertEqual(reconstructed.extraction_confidence, 0.92)
        self.assertIsNotNone(reconstructed.provenance)
        self.assertEqual(reconstructed.provenance.ocr_confidence, 0.99)


class TestTableStructures(unittest.TestCase):
    """Test suite for TableCell, ExtractedTable, and record conversion."""

    def test_extracted_table_records(self) -> None:
        headers = ["Item", "Quantity", "Price", "Total"]
        row1 = [
            TableCell(row_index=0, col_index=0, text="Widget A", normalized_value="Widget A"),
            TableCell(row_index=0, col_index=1, text="2", normalized_value=2),
            TableCell(row_index=0, col_index=2, text="$10.00", normalized_value=10.00),
            TableCell(row_index=0, col_index=3, text="$20.00", normalized_value=20.00),
        ]
        row2 = [
            TableCell(row_index=1, col_index=0, text="Service B", normalized_value="Service B"),
            TableCell(row_index=1, col_index=1, text="1", normalized_value=1),
            TableCell(row_index=1, col_index=2, text="$50.00", normalized_value=50.00),
            TableCell(row_index=1, col_index=3, text="$50.00", normalized_value=50.00),
        ]

        table = ExtractedTable(
            table_id="table_001",
            page_number=1,
            headers=headers,
            rows=[row1, row2],
            confidence=0.88,
        )

        records = table.to_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["Item"], "Widget A")
        self.assertEqual(records[0]["Total"], 20.00)
        self.assertEqual(records[1]["Item"], "Service B")
        self.assertEqual(records[1]["Price"], 50.00)

        # Dictionary round-trip
        table_dict = table.to_dict()
        reconstructed = ExtractedTable.from_dict(table_dict)
        self.assertEqual(reconstructed.table_id, "table_001")
        self.assertEqual(len(reconstructed.rows), 2)
        self.assertEqual(reconstructed.headers, headers)


class TestDocumentAndProcessingResult(unittest.TestCase):
    """Test suite for aggregate Document and complete ProcessingResult serialization."""

    def test_full_document_aggregation(self) -> None:
        metadata = DocumentMetadata(
            document_id="doc_001",
            filename="sample_invoice.pdf",
            file_path=Path("data/samples/sample_invoice.pdf"),
            file_type="application/pdf",
            file_size_bytes=102400,
            checksum_sha256="abcdef1234567890",
            page_count=2,
        )
        page1 = DocumentPage(page_number=1, raw_text="Page 1 Header Text")
        page2 = DocumentPage(page_number=2, raw_text="Page 2 Footer Text")
        doc = Document(
            metadata=metadata,
            pages=[page1, page2],
            classified_type=DocumentType.INVOICE,
            classification_confidence=0.95,
        )

        self.assertEqual(doc.id, "doc_001")
        self.assertIn("Page 1 Header Text", doc.full_text)
        self.assertIn("Page 2 Footer Text", doc.full_text)
        self.assertEqual(doc.get_page(1).page_number, 1)
        self.assertIsNone(doc.get_page(3))

    def test_processing_result_json_export_and_import(self) -> None:
        metadata = DocumentMetadata(
            document_id="doc_999",
            filename="test.png",
            file_path=Path("data/input/test.png"),
            file_type="image/png",
            file_size_bytes=45000,
            checksum_sha256="9876543210fedcba",
            page_count=1,
        )
        doc = Document(metadata=metadata, pages=[DocumentPage(page_number=1, raw_text="Content")])

        field1 = ExtractedField(
            name="total",
            value="$100.00",
            normalized_value=100.0,
            field_type=FieldType.CURRENCY,
            extraction_confidence=0.98,
        )

        val_res = ValidationResult(
            rule_id="RULE_TOTAL_EXISTS",
            rule_name="Total Amount Present",
            status=ValidationStatus.VALID,
            severity=SeverityLevel.INFO,
            message="Total amount identified cleanly.",
            affected_fields=["total"],
        )

        summary = ProcessingSummary(
            document_id="doc_999",
            document_type=DocumentType.RECEIPT,
            total_pages=1,
            total_fields_extracted=1,
            total_tables_extracted=0,
            validation_passed_count=1,
            validation_warning_count=0,
            validation_error_count=0,
            overall_confidence_score=0.98,
            review_required=False,
            processing_time_seconds=0.15,
        )

        result = ProcessingResult(
            document=doc,
            fields={"total": field1},
            tables=[],
            validation_results=[val_res],
            review_flags=[],
            summary=summary,
        )

        json_str = result.to_json()
        self.assertTrue(isinstance(json_str, str))
        parsed_dict = json.loads(json_str)
        self.assertEqual(parsed_dict["document"]["metadata"]["document_id"], "doc_999")
        self.assertEqual(parsed_dict["fields"]["total"]["normalized_value"], 100.0)

        reconstructed = ProcessingResult.from_dict(parsed_dict)
        self.assertEqual(reconstructed.document.id, "doc_999")
        self.assertEqual(reconstructed.fields["total"].name, "total")
        self.assertEqual(len(reconstructed.validation_results), 1)


if __name__ == "__main__":
    unittest.main()
