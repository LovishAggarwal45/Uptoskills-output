"""Unit tests for cross-field consistency validation rules."""

import unittest

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.validation.cross_field_validators import (
    DateOfBirthSanityRule,
    InvoiceDueDateRule,
    SubtotalTotalRelationshipRule,
)
from src.validation.models import ValidationCategory


class TestCrossFieldValidation(unittest.TestCase):
    """Test suite for logical cross-field consistency relationships."""

    def test_invoice_due_date_chronology_valid(self) -> None:
        rule = InvoiceDueDateRule()
        fields = {
            "invoice_date": ExtractedField(name="invoice_date", value="2024-03-01", field_type=FieldType.DATE),
            "due_date": ExtractedField(name="due_date", value="2024-03-31", field_type=FieldType.DATE),
        }
        issues = rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)
        self.assertEqual(issues[0].category, ValidationCategory.CROSS_FIELD)

    def test_invoice_due_date_chronology_conflict(self) -> None:
        rule = InvoiceDueDateRule()
        fields = {
            "invoice_date": ExtractedField(name="invoice_date", value="2024-04-15", field_type=FieldType.DATE),
            "due_date": ExtractedField(name="due_date", value="2024-04-01", field_type=FieldType.DATE),
        }
        issues = rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.INVALID)
        self.assertEqual(issues[0].severity, SeverityLevel.ERROR)
        self.assertIn("is earlier than invoice date", issues[0].message)

    def test_subtotal_total_relationship_valid(self) -> None:
        rule = SubtotalTotalRelationshipRule(tolerance=0.05)
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=100.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=110.0, field_type=FieldType.CURRENCY),
        }
        issues = rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)

    def test_subtotal_total_relationship_exceeds_total(self) -> None:
        rule = SubtotalTotalRelationshipRule(tolerance=0.05)
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=200.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=150.0, field_type=FieldType.CURRENCY),
        }
        issues = rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.INVALID)
        self.assertEqual(issues[0].severity, SeverityLevel.ERROR)
        self.assertIn("exceeds total", issues[0].message)

    def test_date_of_birth_valid(self) -> None:
        rule = DateOfBirthSanityRule()
        fields = {
            "date_of_birth": ExtractedField(name="date_of_birth", value="1990-05-20", field_type=FieldType.DATE),
        }
        issues = rule.evaluate(fields, [], DocumentType.FORM)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)

    def test_date_of_birth_future_rejected(self) -> None:
        rule = DateOfBirthSanityRule()
        fields = {
            "date_of_birth": ExtractedField(name="date_of_birth", value="2099-01-01", field_type=FieldType.DATE),
        }
        issues = rule.evaluate(fields, [], DocumentType.FORM)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.INVALID)
        self.assertEqual(issues[0].severity, SeverityLevel.ERROR)
        self.assertIn("cannot be in the future", issues[0].message)


if __name__ == "__main__":
    unittest.main()
