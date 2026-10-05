"""Unit tests for Phase 6 table region detection and false positive suppression."""

import unittest

from src.core.models import BoundingBox
from src.ocr.models import OCRWord
from src.tables.detector import RuleBasedTableDetector


class TestTableDetection(unittest.TestCase):
    """Test suite for Table Region Detection and false positive filtering."""

    def setUp(self) -> None:
        self.detector = RuleBasedTableDetector()

    def test_detect_table_with_header(self) -> None:
        words = [
            # Header line
            OCRWord(text="Item Description", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=180.0, ymax=120.0), page_number=1),
            OCRWord(text="Qty", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=250.0, ymin=100.0, xmax=280.0, ymax=120.0), page_number=1),
            OCRWord(text="Price", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=350.0, ymin=100.0, xmax=390.0, ymax=120.0), page_number=1),
            OCRWord(text="Total", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=450.0, ymin=100.0, xmax=490.0, ymax=120.0), page_number=1),

            # Data Row 1
            OCRWord(text="Consulting Services", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=140.0, xmax=190.0, ymax=160.0), page_number=1),
            OCRWord(text="10", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=255.0, ymin=140.0, xmax=275.0, ymax=160.0), page_number=1),
            OCRWord(text="$150.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=350.0, ymin=140.0, xmax=400.0, ymax=160.0), page_number=1),
            OCRWord(text="$1,500.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=445.0, ymin=140.0, xmax=510.0, ymax=160.0), page_number=1),
        ]

        regions = self.detector.detect_regions(words=words, page_number=1)
        self.assertEqual(len(regions), 1)

        reg = regions[0]
        self.assertIsNotNone(reg.header)
        self.assertGreaterEqual(reg.confidence_score, 0.70)
        self.assertEqual(len(reg.row_boxes), 1)

    def test_termination_line_stops_table_region(self) -> None:
        words = [
            # Header line
            OCRWord(text="Item", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=100.0, ymax=120.0), page_number=1),
            OCRWord(text="Total", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=400.0, ymin=100.0, xmax=450.0, ymax=120.0), page_number=1),

            # Row 1
            OCRWord(text="Product A", confidence=0.95, raw_confidence=95.0, bounding_box=BoundingBox(xmin=50.0, ymin=140.0, xmax=120.0, ymax=160.0), page_number=1),
            OCRWord(text="$100.00", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=400.0, ymin=140.0, xmax=450.0, ymax=160.0), page_number=1),

            # Termination Line (Terms and Conditions)
            OCRWord(text="Terms and Conditions apply to all orders.", confidence=0.90, raw_confidence=90.0, bounding_box=BoundingBox(xmin=50.0, ymin=180.0, xmax=350.0, ymax=200.0), page_number=1),

            # Noise below termination line
            OCRWord(text="Late payment penalty of 1.5% per month.", confidence=0.90, raw_confidence=90.0, bounding_box=BoundingBox(xmin=50.0, ymin=220.0, xmax=350.0, ymax=240.0), page_number=1),
        ]

        regions = self.detector.detect_regions(words=words, page_number=1)
        self.assertEqual(len(regions), 1)
        # Verify region does not include the text below terms & conditions
        self.assertLessEqual(regions[0].bounding_box.ymax, 170.0)

    def test_empty_words_returns_empty_regions(self) -> None:
        regions = self.detector.detect_regions(words=[], page_number=1)
        self.assertEqual(len(regions), 0)


if __name__ == "__main__":
    unittest.main()
