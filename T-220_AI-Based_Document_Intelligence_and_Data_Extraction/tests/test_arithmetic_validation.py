"""Unit tests for arithmetic consistency rules across totals, line items, and sums."""

import unittest

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.tables.models import LineItem
from src.validation.arithmetic_validators import (
    LineItemArithmeticRule,
    LineItemsSubtotalSumRule,
    SubtotalTaxTotalRule,
)
from src.validation.models import ValidationCategory


class TestArithmeticValidation(unittest.TestCase):
    """Test suite for mathematical and accounting consistency verification."""

    def test_subtotal_tax_total_valid(self) -> None:
        rule = SubtotalTaxTotalRule(tolerance=0.05)
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=100.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=10.0, field_type=FieldType.CURRENCY),
            "discount": ExtractedField(name="discount", value=5.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=105.0, field_type=FieldType.CURRENCY),
        }
        issues = rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)
        self.assertEqual(issues[0].category, ValidationCategory.ARITHMETIC)
        self.assertEqual(issues[0].expected_value, 105.0)

    def test_subtotal_tax_total_mismatch(self) -> None:
        rule = SubtotalTaxTotalRule(tolerance=0.05)
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=100.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=10.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=150.0, field_type=FieldType.CURRENCY),
        }
        issues = rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.INVALID)
        self.assertEqual(issues[0].severity, SeverityLevel.ERROR)
        self.assertIn("Arithmetic mismatch", issues[0].message)

    def test_line_item_arithmetic_valid_and_mismatch(self) -> None:
        rule = LineItemArithmeticRule(tolerance=0.02)
        li1 = LineItem(description="Widget A", quantity=2.0, unit_price=15.0, amount=30.0)
        li2 = LineItem(description="Widget B", quantity=3.0, unit_price=10.0, amount=40.0)  # should be 30

        context = {"line_items": [li1, li2]}
        issues = rule.evaluate({}, [], DocumentType.INVOICE, context=context)
        self.assertEqual(len(issues), 2)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)
        self.assertEqual(issues[1].status, ValidationStatus.INVALID)
        self.assertEqual(issues[1].severity, SeverityLevel.ERROR)
        self.assertIn("arithmetic mismatch", issues[1].message)

    def test_line_items_subtotal_sum(self) -> None:
        rule = LineItemsSubtotalSumRule(tolerance=0.05)
        li1 = LineItem(description="Item 1", amount=50.0)
        li2 = LineItem(description="Item 2", amount=75.0)
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=125.0, field_type=FieldType.CURRENCY),
        }

        context = {"line_items": [li1, li2]}
        issues = rule.evaluate(fields, [], DocumentType.INVOICE, context=context)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)
        self.assertIn("matches document subtotal", issues[0].message)


if __name__ == "__main__":
    unittest.main()
