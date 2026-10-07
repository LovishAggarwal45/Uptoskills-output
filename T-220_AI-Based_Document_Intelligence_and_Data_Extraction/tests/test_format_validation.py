"""Unit tests for format validators across date, currency, email, phone, and identifier fields."""

import unittest
from datetime import datetime

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.validation.format_validators import (
    CurrencyFormatValidator,
    DateFormatValidator,
    DocumentFormatValidationRule,
    EmailFormatValidator,
    IdentifierFormatValidator,
    PhoneFormatValidator,
)
from src.validation.models import ValidationCategory


class TestFormatValidation(unittest.TestCase):
    """Test suite for format validation algorithms and edge cases."""

    def setUp(self) -> None:
        self.date_val = DateFormatValidator()
        self.curr_val = CurrencyFormatValidator()
        self.email_val = EmailFormatValidator()
        self.phone_val = PhoneFormatValidator()
        self.id_val = IdentifierFormatValidator()
        self.doc_format_rule = DocumentFormatValidationRule()

    def test_date_format_valid(self) -> None:
        valid_dates = ["2023-11-20", "15/04/2024", "12-05-2023", "April 15, 2024", "15 Apr 2024"]
        for d_str in valid_dates:
            field = ExtractedField(name="invoice_date", value=d_str, field_type=FieldType.DATE)
            issues = self.date_val.validate_field(field)
            self.assertEqual(len(issues), 1, f"Failed for {d_str}")
            self.assertEqual(issues[0].status, ValidationStatus.VALID)

    def test_date_format_invalid_and_out_of_range(self) -> None:
        invalid_cases = [
            ("not-a-date", "cannot be parsed"),
            ("2024-99-99", "cannot be parsed"),
            ("1850-01-01", "anomalous year"),
            ("2250-01-01", "anomalous year"),
        ]
        for val, err_fragment in invalid_cases:
            field = ExtractedField(name="invoice_date", value=val, field_type=FieldType.DATE)
            issues = self.date_val.validate_field(field)
            self.assertEqual(len(issues), 1, f"Expected 1 issue for {val}")
            self.assertEqual(issues[0].status, ValidationStatus.INVALID)
            self.assertIn(err_fragment, issues[0].message)

    def test_currency_format_valid(self) -> None:
        valid_amounts = ["$1,250.50", "150.00", "€45.99", "1200", 99.50]
        for val in valid_amounts:
            field = ExtractedField(name="total", value=val, field_type=FieldType.CURRENCY)
            issues = self.curr_val.validate_field(field)
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].status, ValidationStatus.VALID)

    def test_currency_format_invalid(self) -> None:
        field = ExtractedField(name="total", value="invalid_currency", field_type=FieldType.CURRENCY)
        issues = self.curr_val.validate_field(field)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.INVALID)

    def test_email_format_valid_and_invalid(self) -> None:
        valid_f = ExtractedField(name="vendor_email", value="billing@acme.com", field_type=FieldType.EMAIL)
        issues = self.email_val.validate_field(valid_f)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)

        invalid_f = ExtractedField(name="vendor_email", value="billing@acme", field_type=FieldType.EMAIL)
        issues_inv = self.email_val.validate_field(invalid_f)
        self.assertEqual(issues_inv[0].status, ValidationStatus.INVALID)

    def test_phone_format_valid_and_invalid(self) -> None:
        valid_f = ExtractedField(name="phone_number", value="+1 (555) 234-5678", field_type=FieldType.PHONE)
        issues = self.phone_val.validate_field(valid_f)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)

        invalid_f = ExtractedField(name="phone_number", value="123", field_type=FieldType.PHONE)
        issues_inv = self.phone_val.validate_field(invalid_f)
        self.assertEqual(issues_inv[0].status, ValidationStatus.INVALID)

    def test_identifier_format_valid_and_invalid(self) -> None:
        valid_f = ExtractedField(name="invoice_number", value="INV-2024-8841", field_type=FieldType.IDENTIFIER)
        issues = self.id_val.validate_field(valid_f)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)

        invalid_f = ExtractedField(name="invoice_number", value="#", field_type=FieldType.IDENTIFIER)
        issues_inv = self.id_val.validate_field(invalid_f)
        self.assertEqual(issues_inv[0].status, ValidationStatus.INVALID)

    def test_document_format_validation_rule_integration(self) -> None:
        fields = {
            "invoice_number": ExtractedField(name="invoice_number", value="INV-100", field_type=FieldType.IDENTIFIER),
            "invoice_date": ExtractedField(name="invoice_date", value="2024-02-15", field_type=FieldType.DATE),
            "email": ExtractedField(name="email", value="hello@world.com", field_type=FieldType.EMAIL),
            "phone": ExtractedField(name="phone", value="+1-555-0199", field_type=FieldType.PHONE),
            "total": ExtractedField(name="total", value="$250.00", field_type=FieldType.CURRENCY),
        }
        issues = self.doc_format_rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 5)
        self.assertTrue(all(i.status == ValidationStatus.VALID for i in issues))


if __name__ == "__main__":
    unittest.main()
