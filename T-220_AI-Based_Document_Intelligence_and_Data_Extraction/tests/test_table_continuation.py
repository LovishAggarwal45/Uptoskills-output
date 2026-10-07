"""Unit tests for Phase 6 multi-page table continuation detection and merging."""

import unittest

from src.core.models import BoundingBox
from src.tables.continuation import TableContinuationDetector
from src.tables.models import LineItem, Table, TableCell, TableColumn, TableRow, TableValueType


class TestTableContinuation(unittest.TestCase):
    """Test suite for TableContinuationDetector multi-page table linking and merging."""

    def setUp(self) -> None:
        self.detector = TableContinuationDetector(min_column_overlap_ratio=0.65)
        self.cols = [
            TableColumn(index=0, name="Description", x_start=50.0, x_end=250.0),
            TableColumn(index=1, name="Quantity", x_start=250.0, x_end=350.0),
            TableColumn(index=2, name="Amount", x_start=350.0, x_end=500.0),
        ]
        self.bbox1 = BoundingBox(xmin=50.0, ymin=500.0, xmax=500.0, ymax=900.0)
        self.bbox2 = BoundingBox(xmin=50.0, ymin=100.0, xmax=500.0, ymax=400.0)

    def test_link_continuations_success(self) -> None:
        t1 = Table(
            table_id="table_p1_1",
            page_number=1,
            headers=["Description", "Quantity", "Amount"],
            columns=list(self.cols),
            rows=[TableRow(row_index=0, cells=[TableCell(row_index=0, col_index=0, text="Item 1")])],
            line_items=[LineItem(description="Item 1", quantity=1.0, amount=10.0, page_number=1)],
            bounding_box=self.bbox1,
            confidence=0.95,
        )

        t2 = Table(
            table_id="table_p2_1",
            page_number=2,
            headers=["Description", "Quantity", "Amount"],
            columns=list(self.cols),
            rows=[TableRow(row_index=0, cells=[TableCell(row_index=0, col_index=0, text="Item 2")])],
            line_items=[LineItem(description="Item 2", quantity=2.0, amount=20.0, page_number=2)],
            bounding_box=self.bbox2,
            confidence=0.95,
        )

        linked = self.detector.link_continuations([t1, t2])
        self.assertEqual(len(linked), 2)
        self.assertFalse(linked[0].is_continuation)
        self.assertTrue(linked[1].is_continuation)
        self.assertEqual(linked[1].continuation_of_table_id, "table_p1_1")

    def test_merge_continued_tables_produces_unified_table(self) -> None:
        t1 = Table(
            table_id="table_p1_1",
            page_number=1,
            headers=["Description", "Quantity", "Amount"],
            columns=list(self.cols),
            rows=[TableRow(row_index=0, cells=[TableCell(row_index=0, col_index=0, text="Item 1")])],
            line_items=[LineItem(description="Item 1", quantity=1.0, amount=10.0, page_number=1)],
            bounding_box=self.bbox1,
            confidence=0.95,
        )

        t2 = Table(
            table_id="table_p2_1",
            page_number=2,
            headers=["Description", "Quantity", "Amount"],
            columns=list(self.cols),
            rows=[TableRow(row_index=0, cells=[TableCell(row_index=0, col_index=0, text="Item 2")])],
            line_items=[LineItem(description="Item 2", quantity=2.0, amount=20.0, page_number=2)],
            bounding_box=self.bbox2,
            confidence=0.95,
        )

        merged = self.detector.merge_continued_tables([t1, t2])
        self.assertEqual(len(merged), 1)

        consolidated = merged[0]
        self.assertEqual(consolidated.table_id, "table_p1_1")
        self.assertEqual(len(consolidated.rows), 2)
        self.assertEqual(len(consolidated.line_items), 2)
        self.assertTrue(consolidated.metadata.get("is_multipage_merged"))


if __name__ == "__main__":
    unittest.main()
