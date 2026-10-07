"""Unit tests for Phase 6 table header detection and canonical synonym matching."""

import unittest

from src.core.models import BoundingBox
from src.ocr.models import OCRWord
from src.tables.header_detector import (
    CANONICAL_HEADER_SYNONYMS,
    HeaderDetector,
    match_canonical_header,
)


class TestTableHeaders(unittest.TestCase):
    """Test suite for Table Header detection and canonical category matching."""

    def test_canonical_header_synonym_matching(self) -> None:
        # Description synonyms
        self.assertEqual(match_canonical_header("Description"), "description")
        self.assertEqual(match_canonical_header("Item Description"), "description")
        self.assertEqual(match_canonical_header("Particulars"), "description")
        self.assertEqual(match_canonical_header("Services"), "description")

        # Quantity synonyms
        self.assertEqual(match_canonical_header("Qty"), "quantity")
        self.assertEqual(match_canonical_header("Quantity"), "quantity")
        self.assertEqual(match_canonical_header("Units"), "quantity")
        self.assertEqual(match_canonical_header("Hours"), "quantity")
        self.assertEqual(match_canonical_header("Count"), "quantity")

        # Unit price synonyms
        self.assertEqual(match_canonical_header("Unit Price"), "unit_price")
        self.assertEqual(match_canonical_header("Price / Unit"), "unit_price")
        self.assertEqual(match_canonical_header("Rate"), "unit_price")
        self.assertEqual(match_canonical_header("Cost"), "unit_price")

        # Amount synonyms
        self.assertEqual(match_canonical_header("Amount"), "amount")
        self.assertEqual(match_canonical_header("Ext Price"), "amount")
        self.assertEqual(match_canonical_header("Line Total"), "amount")
        self.assertEqual(match_canonical_header("Total Due"), "amount")

        # Item code synonyms
        self.assertEqual(match_canonical_header("Item #"), "item_code")
        self.assertEqual(match_canonical_header("SKU"), "item_code")
        self.assertEqual(match_canonical_header("Part No."), "item_code")
        self.assertEqual(match_canonical_header("Code"), "item_code")

        # Tax & Discount
        self.assertEqual(match_canonical_header("VAT"), "tax")
        self.assertEqual(match_canonical_header("Sales Tax"), "tax")
        self.assertEqual(match_canonical_header("Discount"), "discount")

    def test_detect_header_candidates_success(self) -> None:
        detector = HeaderDetector(min_header_columns=2)

        # Build word tokens for header line
        words = [
            OCRWord(text="Item", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=90.0, ymax=120.0), page_number=1),
            OCRWord(text="Description", confidence=0.98, raw_confidence=98.0, bounding_box=BoundingBox(xmin=95.0, ymin=100.0, xmax=180.0, ymax=120.0), page_number=1),
            OCRWord(text="Qty", confidence=0.97, raw_confidence=97.0, bounding_box=BoundingBox(xmin=250.0, ymin=100.0, xmax=280.0, ymax=120.0), page_number=1),
            OCRWord(text="Price", confidence=0.96, raw_confidence=96.0, bounding_box=BoundingBox(xmin=350.0, ymin=100.0, xmax=390.0, ymax=120.0), page_number=1),
            OCRWord(text="Total", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=450.0, ymin=100.0, xmax=490.0, ymax=120.0), page_number=1),
        ]

        headers = detector.detect_header_candidates(words=words, page_number=1)
        self.assertEqual(len(headers), 1)

        hdr = headers[0]
        self.assertEqual(len(hdr.cells), 4)  # "Item Description", "Qty", "Price", "Total"
        self.assertGreaterEqual(hdr.confidence, 0.80)
        self.assertEqual(hdr.cells[0].text, "Item Description")
        self.assertEqual(hdr.cells[1].text, "Qty")
        self.assertEqual(hdr.cells[2].text, "Price")
        self.assertEqual(hdr.cells[3].text, "Total")
    
    def test_ordinary_text_with_tax_and_total_is_not_a_header(self) -> None:
        detector = HeaderDetector(min_header_columns=2)

        # These words appear in ordinary prose, not in a table header.
        words = [
            OCRWord(
                text="Tax",
                confidence=0.98,
                raw_confidence=98.0,
                bounding_box=BoundingBox(
                    xmin=50.0, ymin=200.0, xmax=80.0, ymax=220.0
                ),
                page_number=1,
            ),
            OCRWord(
                text="Total",
                confidence=0.98,
                raw_confidence=98.0,
                bounding_box=BoundingBox(
                    xmin=350.0, ymin=200.0, xmax=400.0, ymax=220.0
                ),
                page_number=1,
            ),
        ]

        headers = detector.detect_header_candidates(
            words=words, page_number=1
        )

        self.assertEqual(
            len(headers),
            0,
            "Ordinary prose should not automatically become a table header.",
        )

    def test_single_word_or_non_header_rejected(self) -> None:
        detector = HeaderDetector(min_header_columns=2)

        # Single word header line
        words = [
            OCRWord(text="INVOICE", confidence=0.99, raw_confidence=99.0, bounding_box=BoundingBox(xmin=50.0, ymin=50.0, xmax=150.0, ymax=70.0), page_number=1),
        ]

        headers = detector.detect_header_candidates(words=words, page_number=1)
        self.assertEqual(len(headers), 0)


if __name__ == "__main__":
    unittest.main()
