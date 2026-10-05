"""Unit tests for OCR domain data models, reading order, and text reconstruction."""

import unittest

from src.core.models import BoundingBox, OCRTextRegion
from src.ocr.models import (
    DocumentOCRResult,
    OCRBlock,
    OCRLevel,
    OCRLine,
    OCRWord,
    PageOCRResult,
)
from src.ocr.postprocessing import (
    compute_bounding_box_union,
    normalize_tesseract_confidence,
    reconstruct_page_text,
    sort_reading_order,
)


class TestOCRModels(unittest.TestCase):
    """Test suite covering OCR structured data models and conversions."""

    def test_ocr_word_creation_and_conversion(self) -> None:
        bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=150.0, ymax=130.0)
        word = OCRWord(
            text="Invoice",
            confidence=0.965,
            raw_confidence=96.5,
            bounding_box=bbox,
            page_number=1,
            block_num=1,
            line_num=1,
            word_num=1,
            reading_order=1,
        )

        self.assertEqual(word.text, "Invoice")
        self.assertEqual(word.confidence, 0.965)
        self.assertEqual(word.raw_confidence, 96.5)
        self.assertEqual(word.reading_order, 1)

        # Serialization
        word_dict = word.to_dict()
        self.assertEqual(word_dict["text"], "Invoice")
        self.assertEqual(word_dict["level"], "word")
        self.assertEqual(word_dict["confidence"], 0.965)

        # Conversion to Core OCRTextRegion
        text_region = word.to_text_region()
        self.assertIsInstance(text_region, OCRTextRegion)
        self.assertEqual(text_region.text, "Invoice")
        self.assertEqual(text_region.confidence, 0.965)

    def test_ocr_line_and_block_aggregation(self) -> None:
        w1 = OCRWord(
            text="Invoice",
            confidence=0.95,
            raw_confidence=95.0,
            bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=120.0, ymax=130.0),
            page_number=1,
            block_num=1,
            line_num=1,
        )
        w2 = OCRWord(
            text="Number:",
            confidence=0.97,
            raw_confidence=97.0,
            bounding_box=BoundingBox(xmin=130.0, ymin=100.0, xmax=200.0, ymax=130.0),
            page_number=1,
            block_num=1,
            line_num=1,
        )

        union_bbox = compute_bounding_box_union([w1.bounding_box, w2.bounding_box])
        self.assertIsNotNone(union_bbox)
        self.assertEqual(union_bbox.xmin, 50.0)
        self.assertEqual(union_bbox.xmax, 200.0)

        line = OCRLine(
            text="Invoice Number:",
            words=[w1, w2],
            bounding_box=union_bbox,
            confidence=0.96,
            page_number=1,
            block_num=1,
            line_num=1,
        )

        block = OCRBlock(
            text="Invoice Number:",
            lines=[line],
            bounding_box=union_bbox,
            confidence=0.96,
            page_number=1,
            block_num=1,
        )

        self.assertEqual(len(block.lines), 1)
        self.assertEqual(len(block.lines[0].words), 2)
        self.assertEqual(block.lines[0].text, "Invoice Number:")


class TestOCRPostprocessing(unittest.TestCase):
    """Test suite covering confidence normalization, reading order, and text reconstruction."""

    def test_normalize_tesseract_confidence(self) -> None:
        # Standard positive confidence [0-100]
        norm, raw = normalize_tesseract_confidence(95.6)
        self.assertEqual(norm, 0.956)
        self.assertEqual(raw, 95.6)

        norm_zero, raw_zero = normalize_tesseract_confidence(0.0)
        self.assertEqual(norm_zero, 0.0)
        self.assertEqual(raw_zero, 0.0)

        # Missing/invalid confidence (-1)
        norm_neg, raw_neg = normalize_tesseract_confidence(-1.0)
        self.assertIsNone(norm_neg)
        self.assertIsNone(raw_neg)

        # String / non-numeric handling
        norm_str, _ = normalize_tesseract_confidence("88.2")
        self.assertEqual(norm_str, 0.882)

    def test_sort_reading_order(self) -> None:
        # Create words out of reading order
        w_line1_word2 = OCRWord(
            text="WORLD",
            confidence=0.9,
            raw_confidence=90.0,
            bounding_box=BoundingBox(xmin=200.0, ymin=50.0, xmax=280.0, ymax=80.0),
            page_number=1,
        )
        w_line1_word1 = OCRWord(
            text="HELLO",
            confidence=0.9,
            raw_confidence=90.0,
            bounding_box=BoundingBox(xmin=50.0, ymin=52.0, xmax=150.0, ymax=82.0),
            page_number=1,
        )
        w_line2_word1 = OCRWord(
            text="DOCUMIND",
            confidence=0.9,
            raw_confidence=90.0,
            bounding_box=BoundingBox(xmin=50.0, ymin=150.0, xmax=180.0, ymax=180.0),
            page_number=1,
        )

        unarranged = [w_line2_word1, w_line1_word2, w_line1_word1]
        ordered = sort_reading_order(unarranged, line_tolerance_px=10.0)

        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0].text, "HELLO")
        self.assertEqual(ordered[0].reading_order, 1)
        self.assertEqual(ordered[1].text, "WORLD")
        self.assertEqual(ordered[1].reading_order, 2)
        self.assertEqual(ordered[2].text, "DOCUMIND")
        self.assertEqual(ordered[2].reading_order, 3)

    def test_reconstruct_page_text(self) -> None:
        line1 = OCRLine(
            text="DocuMind AI System",
            words=[
                OCRWord("DocuMind", 0.9, 90.0, BoundingBox(10, 10, 80, 30), 1),
                OCRWord("AI", 0.9, 90.0, BoundingBox(90, 10, 110, 30), 1),
                OCRWord("System", 0.9, 90.0, BoundingBox(120, 10, 180, 30), 1),
            ],
            bounding_box=BoundingBox(10, 10, 180, 30),
            confidence=0.9,
            page_number=1,
            block_num=1,
        )
        line2 = OCRLine(
            text="Invoice Number: INV-001",
            words=[
                OCRWord("Invoice", 0.9, 90.0, BoundingBox(10, 40, 70, 60), 1),
                OCRWord("Number:", 0.9, 90.0, BoundingBox(80, 40, 140, 60), 1),
                OCRWord("INV-001", 0.9, 90.0, BoundingBox(150, 40, 220, 60), 1),
            ],
            bounding_box=BoundingBox(10, 40, 220, 60),
            confidence=0.9,
            page_number=1,
            block_num=2,
        )

        reconstructed = reconstruct_page_text([line1, line2], preserve_blocks=True)
        self.assertIn("DocuMind AI System", reconstructed)
        self.assertIn("Invoice Number: INV-001", reconstructed)
        # Verify block separation
        self.assertIn("\n\nInvoice Number: INV-001", reconstructed)


class TestDocumentOCRResult(unittest.TestCase):
    """Test suite covering DocumentOCRResult aggregation across multiple pages."""

    def test_document_ocr_result_aggregation(self) -> None:
        page1 = PageOCRResult(
            page_number=1,
            raw_text="Page 1 Content",
            words=[OCRWord("Page", 0.95, 95.0, BoundingBox(10, 10, 50, 30), 1)],
            engine_name="tesseract",
            mean_confidence=0.95,
            execution_time_seconds=0.12,
        )
        page2 = PageOCRResult(
            page_number=2,
            raw_text="Page 2 Content",
            words=[OCRWord("Page", 0.85, 85.0, BoundingBox(10, 10, 50, 30), 2)],
            engine_name="tesseract",
            mean_confidence=0.85,
            execution_time_seconds=0.15,
        )

        doc_result = DocumentOCRResult(
            document_id="doc_multi_001",
            pages=[page1, page2],
            full_text=f"{page1.raw_text}\n\n--- Page Break ---\n\n{page2.raw_text}",
            total_words=2,
            total_lines=0,
            total_blocks=0,
            mean_confidence=0.90,
            engine_name="tesseract",
            total_duration_seconds=0.27,
        )

        self.assertEqual(doc_result.document_id, "doc_multi_001")
        self.assertEqual(len(doc_result.pages), 2)
        self.assertEqual(doc_result.mean_confidence, 0.90)
        self.assertIn("Page 1 Content", doc_result.full_text)
        self.assertIn("Page 2 Content", doc_result.full_text)

        doc_dict = doc_result.to_dict()
        self.assertEqual(doc_dict["total_pages"], 2)
        self.assertEqual(doc_dict["total_words"], 2)


if __name__ == "__main__":
    unittest.main()
