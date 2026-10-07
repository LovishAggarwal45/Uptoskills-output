"""Unit tests for Phase 6 cell parsing, normalization, and value typing."""

import unittest

from src.core.models import BoundingBox
from src.tables.cell_parser import CellParser
from src.tables.models import TableCell, TableColumn, TableValueType


class TestTableCells(unittest.TestCase):
    """Test suite for Table Cell value parsing, normalization, and typing."""

    def test_normalize_cell_currency_value(self) -> None:
        val, vtype = CellParser.normalize_cell_value("$1,250.50", canonical_field="amount")
        self.assertEqual(val, 1250.50)
        self.assertEqual(vtype, TableValueType.CURRENCY)

        val_euro, vtype_euro = CellParser.normalize_cell_value("€ 99.00", canonical_field="unit_price")
        self.assertEqual(val_euro, 99.00)
        self.assertEqual(vtype_euro, TableValueType.CURRENCY)

    def test_normalize_cell_integer_and_decimal_values(self) -> None:
        val_int, vtype_int = CellParser.normalize_cell_value("5", canonical_field="quantity")
        self.assertEqual(val_int, 5.0)
        self.assertEqual(vtype_int, TableValueType.INTEGER)

        val_dec, vtype_dec = CellParser.normalize_cell_value("12.75")
        self.assertEqual(val_dec, 12.75)
        self.assertEqual(vtype_dec, TableValueType.DECIMAL)

    def test_normalize_cell_percentage_and_date(self) -> None:
        val_pct, vtype_pct = CellParser.normalize_cell_value("15%")
        self.assertEqual(val_pct, 15.0)
        self.assertEqual(vtype_pct, TableValueType.PERCENTAGE)

        val_date, vtype_date = CellParser.normalize_cell_value("2026-10-01")
        self.assertEqual(val_date, "2026-10-01")
        self.assertEqual(vtype_date, TableValueType.DATE)

    def test_normalize_empty_cell(self) -> None:
        val, vtype = CellParser.normalize_cell_value("   ")
        self.assertIsNone(val)
        self.assertEqual(vtype, TableValueType.EMPTY)

    def test_populate_cell_values_integration(self) -> None:
        col = TableColumn(index=0, name="Unit Price", canonical_field="unit_price", inferred_type=TableValueType.CURRENCY)
        cell = TableCell(row_index=0, col_index=0, text="$45.00")

        CellParser.populate_cell_values(cell, column=col)
        self.assertEqual(cell.normalized_value, 45.0)
        self.assertEqual(cell.value_type, TableValueType.CURRENCY)


if __name__ == "__main__":
    unittest.main()
