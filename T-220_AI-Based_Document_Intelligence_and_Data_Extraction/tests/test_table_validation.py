"""Unit tests for Phase 6 table arithmetic validation and integrity checks."""

import unittest

from src.core.models import BoundingBox
from src.tables.models import LineItem, Table, TableCell, TableColumn, TableRow, TableValueType
from src.tables.validators import TableValidator


class TestTableValidation(unittest.TestCase):
    """Test suite for TableValidator integrity and arithmetic validation."""

    def setUp(self) -> None:
        self.validator = TableValidator(amount_tolerance=0.02, total_tolerance=0.05)
        self.bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=550.0, ymax=130.0)

    def test_validate_table_valid_sums(self) -> None:
        item1 = LineItem(description="Item 1", quantity=2.0, unit_price=100.0, amount=200.0, is_valid_arithmetic=True)
        item2 = LineItem(description="Item 2", quantity=1.0, unit_price=50.0, amount=50.0, is_valid_arithmetic=True)

        # Summary subtotal row ($250.00)
        subtotal_row = TableRow(
            row_index=2,
            cells=[
                TableCell(row_index=2, col_index=0, text="Subtotal"),
                TableCell(row_index=2, col_index=1, text="$250.00", normalized_value=250.0),
            ],
            is_summary=True,
        )

        table = Table(
            table_id="tbl_valid",
            page_number=1,
            headers=["Description", "Amount"],
            columns=[
                TableColumn(index=0, name="Description"),
                TableColumn(index=1, name="Amount"),
            ],
            rows=[subtotal_row],
            line_items=[item1, item2],
            confidence=0.95,
        )

        result = self.validator.validate_table(table)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.row_checks_passed, 2)
        self.assertEqual(result.row_checks_failed, 0)
        self.assertTrue(result.subtotal_valid)
        self.assertEqual(result.calculated_subtotal, 250.0)

    def test_validate_table_subtotal_mismatch(self) -> None:
        item1 = LineItem(description="Item 1", quantity=2.0, unit_price=100.0, amount=200.0, is_valid_arithmetic=True)

        # Reported subtotal is $300.00 instead of $200.00
        subtotal_row = TableRow(
            row_index=1,
            cells=[
                TableCell(row_index=1, col_index=0, text="Subtotal"),
                TableCell(row_index=1, col_index=1, text="$300.00", normalized_value=300.0),
            ],
            is_summary=True,
        )

        table = Table(
            table_id="tbl_mismatch",
            page_number=1,
            headers=["Description", "Amount"],
            columns=[
                TableColumn(index=0, name="Description"),
                TableColumn(index=1, name="Amount"),
            ],
            rows=[subtotal_row],
            line_items=[item1],
            confidence=0.95,
        )

        result = self.validator.validate_table(table)
        self.assertFalse(result.is_valid)
        self.assertFalse(result.subtotal_valid)
        self.assertEqual(len(result.discrepancies), 1)
        self.assertEqual(result.discrepancies[0]["type"], "subtotal_mismatch")


if __name__ == "__main__":
    unittest.main()
