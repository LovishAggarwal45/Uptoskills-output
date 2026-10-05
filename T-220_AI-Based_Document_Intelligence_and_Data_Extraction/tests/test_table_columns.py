"""Unit tests for Phase 6 column boundary parsing, X-clustering, and type inference."""

import unittest

from src.core.models import BoundingBox
from src.ocr.models import OCRWord
from src.tables.column_parser import ColumnParser
from src.tables.models import TableCell, TableHeader, TableValueType


class TestTableColumns(unittest.TestCase):
    """Test suite for Table Column discovery, type inference, and boundary establishment."""

    def setUp(self) -> None:
        self.parser = ColumnParser(min_column_gap_px=15.0)

    def test_parse_columns_from_header_with_data_words(self) -> None:
        # Header: Description [50..200], Qty [250..300], Price [350..420], Amount [480..560]
        header_cells = [
            TableCell(row_index=0, col_index=0, text="Description", bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=200.0, ymax=120.0), is_header=True),
            TableCell(row_index=0, col_index=1, text="Qty", bounding_box=BoundingBox(xmin=250.0, ymin=100.0, xmax=300.0, ymax=120.0), is_header=True),
            TableCell(row_index=0, col_index=2, text="Unit Price", bounding_box=BoundingBox(xmin=350.0, ymin=100.0, xmax=420.0, ymax=120.0), is_header=True),
            TableCell(row_index=0, col_index=3, text="Amount", bounding_box=BoundingBox(xmin=480.0, ymin=100.0, xmax=560.0, ymax=120.0), is_header=True),
        ]
        header = TableHeader(row_index=0, cells=header_cells, column_names=["Description", "Qty", "Unit Price", "Amount"])

        # Data words below
        data_words = [
            OCRWord(text="Enterprise", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=140.0, xmax=140.0, ymax=160.0), page_number=1),
            OCRWord(text="Software", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=145.0, ymin=140.0, xmax=210.0, ymax=160.0), page_number=1),
            OCRWord(text="2", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=270.0, ymin=140.0, xmax=285.0, ymax=160.0), page_number=1),
            OCRWord(text="$500.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=360.0, ymin=140.0, xmax=415.0, ymax=160.0), page_number=1),
            OCRWord(text="$1,000.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=485.0, ymin=140.0, xmax=555.0, ymax=160.0), page_number=1),
        ]

        columns = self.parser.parse_columns_from_header(header, data_words, page_width=600.0)
        self.assertEqual(len(columns), 4)

        # Verify boundaries are contiguous and non-overlapping
        for i in range(len(columns) - 1):
            self.assertLess(columns[i].x_start, columns[i].x_end)
            self.assertAlmostEqual(columns[i].x_end, columns[i + 1].x_start, delta=0.1)

        # Inferred types
        self.assertEqual(columns[0].inferred_type, TableValueType.TEXT)
        self.assertEqual(columns[1].inferred_type, TableValueType.INTEGER)
        self.assertEqual(columns[2].inferred_type, TableValueType.CURRENCY)
        self.assertEqual(columns[3].inferred_type, TableValueType.CURRENCY)

        # Alignments
        self.assertEqual(columns[0].alignment, "left")
        self.assertEqual(columns[1].alignment, "right")
        self.assertEqual(columns[2].alignment, "right")
        self.assertEqual(columns[3].alignment, "right")

    def test_parse_columns_from_words_headerless(self) -> None:
        # Headerless words forming 2 columns
        words = [
            OCRWord(text="Item A", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=120.0, ymax=120.0), page_number=1),
            OCRWord(text="$100.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=300.0, ymin=100.0, xmax=360.0, ymax=120.0), page_number=1),
            OCRWord(text="Item B", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=130.0, xmax=120.0, ymax=150.0), page_number=1),
            OCRWord(text="$200.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=300.0, ymin=130.0, xmax=360.0, ymax=150.0), page_number=1),
        ]

        columns = self.parser.parse_columns_from_words(words, page_width=500.0)
        self.assertEqual(len(columns), 2)
        self.assertEqual(columns[0].inferred_type, TableValueType.TEXT)
        self.assertEqual(columns[1].inferred_type, TableValueType.CURRENCY)


if __name__ == "__main__":
    unittest.main()
