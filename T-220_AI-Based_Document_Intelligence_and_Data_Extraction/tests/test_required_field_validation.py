"""Unit tests for required field presence validation."""

import unittest

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.validation.models import ValidationCategory
from src.validation.required_field_validator import RequiredFieldValidator


class TestRequiredFieldValidation(unittest.TestCase):
    """Test suite for RequiredFieldValidator across document categories."""

    def setUp(self) -> None:
        self.validator = RequiredFieldValidator()

    def test_invoice_all_required_present(self) -> None:
        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-9921", field_type=FieldType.IDENTIFIER),
            "invoice_date": ExtractedField(name="invoice_date", value="2024-03-15", field_type=FieldType.DATE),
            "vendor_name": ExtractedField(name="vendor_name", value="Acme Supplies Ltd", field_type=FieldType.ORGANIZATION),
            "total": ExtractedField(name="total", value=540.00, field_type=FieldType.CURRENCY),
        }

        issues = self.validator.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 4)
        for issue in issues:
            self.assertEqual(issue.status, ValidationStatus.VALID)
            self.assertEqual(issue.category, ValidationCategory.REQUIRED_FIELD)
            self.assertEqual(issue.severity, SeverityLevel.INFO)

    def test_invoice_missing_total_and_empty_vendor(self) -> None:
        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-9921", field_type=FieldType.IDENTIFIER),
            "invoice_date": ExtractedField(name="invoice_date", value="2024-03-15", field_type=FieldType.DATE),
            "vendor_name": ExtractedField(name="vendor_name", value="   ", field_type=FieldType.ORGANIZATION),
            # total is missing entirely
        }

        issues = self.validator.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 4)

        invalid_issues = [i for i in issues if i.status == ValidationStatus.INVALID]
        self.assertEqual(len(invalid_issues), 2)
        invalid_fields = [i.affected_fields[0] for i in invalid_issues]
        self.assertIn("vendor_name", invalid_fields)
        self.assertIn("total", invalid_fields)

    def test_receipt_required_fields(self) -> None:
        fields = {
            "merchant_name": ExtractedField(name="merchant_name", value="Starbucks Coffee", field_type=FieldType.ORGANIZATION),
            "transaction_date": ExtractedField(name="transaction_date", value="2024-05-10", field_type=FieldType.DATE),
            "total": ExtractedField(name="total", value=7.50, field_type=FieldType.CURRENCY),
        }

        issues = self.validator.evaluate(fields, [], DocumentType.RECEIPT)
        self.assertEqual(len(issues), 3)
        self.assertTrue(all(i.status == ValidationStatus.VALID for i in issues))

    def test_form_required_fields(self) -> None:
        fields = {
            "applicant_name": ExtractedField(name="applicant_name", value="John Doe", field_type=FieldType.STRING),
            # date_of_birth is missing
        }

        issues = self.validator.evaluate(fields, [], DocumentType.FORM)
        self.assertEqual(len(issues), 2)
        invalid = [i for i in issues if i.status == ValidationStatus.INVALID]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].affected_fields, ["date_of_birth"])

    def test_unknown_document_type_has_no_required_fields(self) -> None:
        fields = {}
        issues = self.validator.evaluate(fields, [], DocumentType.UNKNOWN)
        self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
