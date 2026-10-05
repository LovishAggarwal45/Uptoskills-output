"""Unit and integration tests for DocumentValidationEngine."""

import unittest

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.extraction.models import ExtractionResult
from src.tables.models import LineItem, Table, TableExtractionResult, TableValidationResult
from src.validation.models import ValidationCategory, ValidationReport
from src.validation.validator import DocumentValidationEngine


class TestDocumentValidationEngine(unittest.TestCase):
    """Test suite for DocumentValidationEngine coordinator."""

    def setUp(self) -> None:
        self.engine = DocumentValidationEngine()

    def test_engine_updates_field_statuses_in_place(self) -> None:
        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-2024-001", field_type=FieldType.IDENTIFIER),
            "invoice_date": ExtractedField(name="invoice_date", value="invalid-date", field_type=FieldType.DATE),
            "vendor_name": ExtractedField(name="vendor_name", value="Acme Corp", field_type=FieldType.ORGANIZATION),
            "subtotal": ExtractedField(name="subtotal", value=100.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=10.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=110.0, field_type=FieldType.CURRENCY),
        }

        report = self.engine.validate_document(
            document_id="doc_inv_101",
            fields=fields,
            tables=[],
            document_type=DocumentType.INVOICE,
        )

        self.assertIsInstance(report, ValidationReport)
        self.assertEqual(report.overall_status, ValidationStatus.INVALID)
        self.assertEqual(fields["invoice_number"].validation_status, ValidationStatus.VALID)
        self.assertEqual(fields["invoice_date"].validation_status, ValidationStatus.INVALID)
        self.assertGreater(len(fields["invoice_date"].validation_messages), 0)

    def test_validate_extraction_result_and_table_result(self) -> None:
        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-9999", field_type=FieldType.IDENTIFIER),
            "invoice_date": ExtractedField(name="invoice_date", value="2024-06-01", field_type=FieldType.DATE),
            "due_date": ExtractedField(name="due_date", value="2024-06-30", field_type=FieldType.DATE),
            "vendor_name": ExtractedField(name="vendor_name", value="Global Tech", field_type=FieldType.ORGANIZATION),
            "subtotal": ExtractedField(name="subtotal", value=200.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=20.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=220.0, field_type=FieldType.CURRENCY),
        }
        ext_res = ExtractionResult(
            document_id="doc_ext_456",
            document_type=DocumentType.INVOICE,
            fields=fields,
        )

        li1 = LineItem(description="Item 1", quantity=2.0, unit_price=50.0, amount=100.0)
        li2 = LineItem(description="Item 2", quantity=2.0, unit_price=50.0, amount=100.0)
        tbl = Table(
            table_id="table_1",
            page_number=1,
            headers=["Description", "Qty", "Price", "Total"],
            line_items=[li1, li2],
            validation_result=TableValidationResult(is_valid=True, row_checks_passed=2),
        )
        tbl_res = TableExtractionResult(
            document_id="doc_ext_456",
            tables=[tbl],
            line_items=[li1, li2],
        )

        report = self.engine.validate_extraction_result(ext_res, tbl_res)
        self.assertEqual(report.overall_status, ValidationStatus.VALID)
        self.assertEqual(report.validation_score, 1.0)
        self.assertTrue(report.is_valid)


if __name__ == "__main__":
    unittest.main()
