"""Unit tests for InvoiceValidator, ReceiptValidator, and FormValidator."""

import unittest

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, ValidationStatus
from src.validation.form_validators import FormValidator
from src.validation.invoice_validators import InvoiceValidator
from src.validation.receipt_validators import ReceiptValidator


class TestSpecializedValidators(unittest.TestCase):
    """Test suite for document-type tailored validator suites."""

    def test_invoice_validator_full_valid(self) -> None:
        validator = InvoiceValidator()
        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-2024-001", field_type=FieldType.IDENTIFIER),
            "invoice_date": ExtractedField(name="invoice_date", value="2024-01-10", field_type=FieldType.DATE),
            "due_date": ExtractedField(name="due_date", value="2024-02-10", field_type=FieldType.DATE),
            "vendor_name": ExtractedField(name="vendor_name", value="Acme Inc", field_type=FieldType.ORGANIZATION),
            "subtotal": ExtractedField(name="subtotal", value=100.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=10.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=110.0, field_type=FieldType.CURRENCY),
        }
        issues = validator.validate_document(fields, [], DocumentType.INVOICE)
        self.assertGreaterEqual(len(issues), 4)
        self.assertTrue(all(i.status == ValidationStatus.VALID for i in issues))

    def test_receipt_validator_valid(self) -> None:
        validator = ReceiptValidator()
        fields = {
            "merchant_name": ExtractedField(name="merchant_name", value="Local Cafe", field_type=FieldType.ORGANIZATION),
            "transaction_date": ExtractedField(name="transaction_date", value="2024-04-05", field_type=FieldType.DATE),
            "subtotal": ExtractedField(name="subtotal", value=15.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=1.5, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=16.5, field_type=FieldType.CURRENCY),
        }
        issues = validator.validate_document(fields, [], DocumentType.RECEIPT)
        self.assertTrue(all(i.status == ValidationStatus.VALID for i in issues))

    def test_form_validator_valid(self) -> None:
        validator = FormValidator()
        fields = {
            "applicant_name": ExtractedField(name="applicant_name", value="Jane Smith", field_type=FieldType.STRING),
            "date_of_birth": ExtractedField(name="date_of_birth", value="1995-08-12", field_type=FieldType.DATE),
        }
        issues = validator.validate_document(fields, [], DocumentType.FORM)
        self.assertTrue(all(i.status == ValidationStatus.VALID for i in issues))


if __name__ == "__main__":
    unittest.main()
