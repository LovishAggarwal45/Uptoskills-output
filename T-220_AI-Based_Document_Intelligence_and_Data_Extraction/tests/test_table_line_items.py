"""Unit tests for Phase 6 canonical line-item extraction and semantic column mapping."""

import unittest

from src.core.models import BoundingBox
from src.tables.line_item_extractor import LineItemExtractor
from src.tables.models import Table, TableCell, TableColumn, TableRow, TableValueType


class TestTableLineItems(unittest.TestCase):
    """Test suite for Line Item Extraction, column mapping, and field derivation."""

    def setUp(self) -> None:
        self.extractor = LineItemExtractor(amount_tolerance=0.02)
        self.columns = [
            TableColumn(index=0, name="Description", canonical_field="description", inferred_type=TableValueType.TEXT),
            TableColumn(index=1, name="Quantity", canonical_field="quantity", inferred_type=TableValueType.INTEGER),
            TableColumn(index=2, name="Unit Price", canonical_field="unit_price", inferred_type=TableValueType.CURRENCY),
            TableColumn(index=3, name="Total Amount", canonical_field="amount", inferred_type=TableValueType.CURRENCY),
        ]
        self.bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=550.0, ymax=130.0)

    def test_extract_line_items_valid_arithmetic(self) -> None:
        row = TableRow(
            row_index=0,
            cells=[
                TableCell(row_index=0, col_index=0, text="Cloud Server Hosting", normalized_value="Cloud Server Hosting", bounding_box=self.bbox),
                TableCell(row_index=0, col_index=1, text="2", normalized_value=2.0, bounding_box=self.bbox),
                TableCell(row_index=0, col_index=2, text="$150.00", normalized_value=150.0, bounding_box=self.bbox),
                TableCell(row_index=0, col_index=3, text="$300.00", normalized_value=300.0, bounding_box=self.bbox),
            ],
            bounding_box=self.bbox,
            confidence=0.98,
        )

        table = Table(
            table_id="tbl_test",
            page_number=1,
            headers=["Description", "Quantity", "Unit Price", "Total Amount"],
            columns=self.columns,
            rows=[row],
            confidence=0.95,
        )

        items = self.extractor.extract_line_items(table, document_id="doc_test")
        self.assertEqual(len(items), 1)

        item = items[0]
        self.assertEqual(item.description, "Cloud Server Hosting")
        self.assertEqual(item.quantity, 2.0)
        self.assertEqual(item.unit_price, 150.0)
        self.assertEqual(item.amount, 300.0)
        self.assertTrue(item.is_valid_arithmetic)
        self.assertEqual(len(item.validation_messages), 0)
        self.assertIsNotNone(item.provenance)
        self.assertEqual(item.provenance.page_number, 1)

    def test_extract_line_item_derives_missing_amount(self) -> None:
        # Amount cell is empty
        row = TableRow(
            row_index=0,
            cells=[
                TableCell(row_index=0, col_index=0, text="SSL Certificate", normalized_value="SSL Certificate"),
                TableCell(row_index=0, col_index=1, text="3", normalized_value=3.0),
                TableCell(row_index=0, col_index=2, text="$40.00", normalized_value=40.0),
                TableCell(row_index=0, col_index=3, text="", normalized_value=None),
            ],
            bounding_box=self.bbox,
            confidence=0.95,
        )

        table = Table(
            table_id="tbl_test",
            page_number=1,
            headers=["Description", "Quantity", "Unit Price", "Total Amount"],
            columns=self.columns,
            rows=[row],
            confidence=0.95,
        )

        items = self.extractor.extract_line_items(table, document_id="doc_test")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].amount, 120.0)  # 3 * 40.0

    def test_extract_line_item_arithmetic_mismatch_flagged(self) -> None:
        # Reported amount $500.00 does not match 2 * $150 = $300
        row = TableRow(
            row_index=0,
            cells=[
                TableCell(row_index=0, col_index=0, text="Hardware Unit", normalized_value="Hardware Unit"),
                TableCell(row_index=0, col_index=1, text="2", normalized_value=2.0),
                TableCell(row_index=0, col_index=2, text="$150.00", normalized_value=150.0),
                TableCell(row_index=0, col_index=3, text="$500.00", normalized_value=500.0),
            ],
            bounding_box=self.bbox,
            confidence=0.95,
        )

        table = Table(
            table_id="tbl_test",
            page_number=1,
            headers=["Description", "Quantity", "Unit Price", "Total Amount"],
            columns=self.columns,
            rows=[row],
            confidence=0.95,
        )

        items = self.extractor.extract_line_items(table, document_id="doc_test")
        self.assertEqual(len(items), 1)
        self.assertFalse(items[0].is_valid_arithmetic)
        self.assertGreaterEqual(len(items[0].validation_messages), 1)


if __name__ == "__main__":
    unittest.main()
