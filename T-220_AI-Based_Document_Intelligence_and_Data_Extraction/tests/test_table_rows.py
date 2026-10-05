"""Unit tests for Phase 6 row parsing, wrapped multi-line descriptions, and summary detection."""

import unittest

from src.core.models import BoundingBox
from src.ocr.models import OCRWord
from src.tables.models import TableColumn, TableValueType
from src.tables.row_parser import RowParser, is_summary_text


class TestTableRows(unittest.TestCase):
    """Test suite for Table Row clustering, multi-line wrapped cells, and summary discrimination."""

    def setUp(self) -> None:
        self.parser = RowParser(line_height_tolerance_px=12.0, max_multiline_gap_factor=1.6)
        self.columns = [
            TableColumn(index=0, name="Description", canonical_field="description", x_start=40.0, x_end=240.0, inferred_type=TableValueType.TEXT),
            TableColumn(index=1, name="Qty", canonical_field="quantity", x_start=240.0, x_end=340.0, inferred_type=TableValueType.INTEGER),
            TableColumn(index=2, name="Price", canonical_field="unit_price", x_start=340.0, x_end=450.0, inferred_type=TableValueType.CURRENCY),
            TableColumn(index=3, name="Amount", canonical_field="amount", x_start=450.0, x_end=580.0, inferred_type=TableValueType.CURRENCY),
        ]

    def test_parse_rows_single_line_items(self) -> None:
        words = [
            # Row 1
            OCRWord(text="Widget A", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=150.0, xmax=120.0, ymax=170.0), page_number=1),
            OCRWord(text="1", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=280.0, ymin=150.0, xmax=290.0, ymax=170.0), page_number=1),
            OCRWord(text="$10.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=380.0, ymin=150.0, xmax=420.0, ymax=170.0), page_number=1),
            OCRWord(text="$10.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=490.0, ymin=150.0, xmax=530.0, ymax=170.0), page_number=1),

            # Row 2
            OCRWord(text="Widget B", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=185.0, xmax=120.0, ymax=205.0), page_number=1),
            OCRWord(text="2", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=280.0, ymin=185.0, xmax=290.0, ymax=205.0), page_number=1),
            OCRWord(text="$15.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=380.0, ymin=185.0, xmax=420.0, ymax=205.0), page_number=1),
            OCRWord(text="$30.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=490.0, ymin=185.0, xmax=530.0, ymax=205.0), page_number=1),
        ]

        rows = self.parser.parse_rows(words=words, columns=self.columns, page_number=1, header_ymax=120.0)
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0].is_summary)
        self.assertEqual(rows[0].cells[0].text, "Widget A")
        self.assertEqual(rows[0].cells[1].text, "1")
        self.assertEqual(rows[1].cells[0].text, "Widget B")

    def test_parse_rows_with_wrapped_multiline_description(self) -> None:
        words = [
            # Row 1 Line 1: Description + Numbers
            OCRWord(text="Enterprise Hosting Plan", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=150.0, xmax=180.0, ymax=170.0), page_number=1),
            OCRWord(text="1", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=280.0, ymin=150.0, xmax=290.0, ymax=170.0), page_number=1),
            OCRWord(text="$500.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=380.0, ymin=150.0, xmax=430.0, ymax=170.0), page_number=1),
            OCRWord(text="$500.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=490.0, ymin=150.0, xmax=540.0, ymax=170.0), page_number=1),

            # Row 1 Line 2 (Wrapped description with NO numeric tokens)
            OCRWord(text="Includes 24/7 Dedicated Support", confidence=0.94, raw_confidence=94.0, bounding_box=BoundingBox(xmin=50.0, ymin=175.0, xmax=220.0, ymax=192.0), page_number=1),

            # Row 2 Line 1
            OCRWord(text="Setup Fee", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=215.0, xmax=120.0, ymax=235.0), page_number=1),
            OCRWord(text="1", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=280.0, ymin=215.0, xmax=290.0, ymax=235.0), page_number=1),
            OCRWord(text="$100.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=380.0, ymin=215.0, xmax=430.0, ymax=235.0), page_number=1),
            OCRWord(text="$100.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=490.0, ymin=215.0, xmax=540.0, ymax=235.0), page_number=1),
        ]

        rows = self.parser.parse_rows(words=words, columns=self.columns, page_number=1, header_ymax=120.0)
        self.assertEqual(len(rows), 2)  # Should have merged into 2 rows, NOT 3 rows

        # Check merged description cell
        row1_desc = rows[0].cells[0].text
        self.assertIn("Enterprise Hosting Plan", row1_desc)
        self.assertIn("Includes 24/7 Dedicated Support", row1_desc)
        self.assertEqual(rows[0].cells[3].text, "$500.00")

    def test_summary_row_detection(self) -> None:
        self.assertTrue(is_summary_text("Subtotal: $600.00"))
        self.assertTrue(is_summary_text("Tax (10%): $60.00"))
        self.assertTrue(is_summary_text("Total Amount Due: $660.00"))
        self.assertTrue(is_summary_text("Balance Due"))
        self.assertFalse(is_summary_text("Custom Software Development Service"))


if __name__ == "__main__":
    unittest.main()
