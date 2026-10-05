"""Unit tests for DocumentOCRProcessor multi-page orchestration and page updates."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    OCRTextRegion,
)
from src.ocr.base import BaseOCREngine
from src.ocr.models import OCRLine, OCRWord, PageOCRResult
from src.ocr.processor import DocumentOCRProcessor


class MockTestOCREngine(BaseOCREngine):
    """Deterministic mock OCR engine strictly for testing multi-page orchestration."""

    @property
    def engine_name(self) -> str:
        return "mock_test_engine"

    def process_image(self, image_input, page_number: int = 1) -> PageOCRResult:
        if page_number == 99:
            raise RuntimeError("Simulated single page OCR failure.")

        word1 = OCRWord(
            text=f"Page{page_number}",
            confidence=0.95,
            raw_confidence=95.0,
            bounding_box=BoundingBox(10, 10, 80, 30),
            page_number=page_number,
        )
        word2 = OCRWord(
            text="Invoice",
            confidence=0.90,
            raw_confidence=90.0,
            bounding_box=BoundingBox(90, 10, 160, 30),
            page_number=page_number,
        )
        line = OCRLine(
            text=f"Page{page_number} Invoice",
            words=[word1, word2],
            bounding_box=BoundingBox(10, 10, 160, 30),
            confidence=0.925,
            page_number=page_number,
        )

        return PageOCRResult(
            page_number=page_number,
            raw_text=f"Page{page_number} Invoice",
            lines=[line],
            words=[word1, word2],
            text_regions=[line.to_text_region()],
            engine_name="mock_test_engine",
            mean_confidence=0.925,
            execution_time_seconds=0.01,
        )


class TestOCRProcessor(unittest.TestCase):
    """Test suite covering DocumentOCRProcessor coordination and DocumentPage mutation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.doc_dir = self.root_path / "doc_test_100"
        self.doc_dir.mkdir(parents=True)

        # Create dummy page images
        self.page1_img = self.doc_dir / "page_001_original.png"
        self.page2_img = self.doc_dir / "page_002_original.png"
        Image.new("RGB", (200, 200), "white").save(self.page1_img)
        Image.new("RGB", (200, 200), "white").save(self.page2_img)

        # Create dummy preprocessed image for page 1
        self.page1_proc_img = self.doc_dir / "page_001_processed.png"
        Image.new("L", (200, 200), 255).save(self.page1_proc_img)

        self.mock_engine = MockTestOCREngine()
        self.processor = DocumentOCRProcessor(
            engine=self.mock_engine,
            use_preprocessed_images=True,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_multi_page_document_processing(self) -> None:
        meta = DocumentMetadata(
            document_id="doc_test_100",
            filename="invoice_multi.pdf",
            file_path=self.root_path / "invoice_multi.pdf",
            file_type="application/pdf",
            file_size_bytes=5000,
            checksum_sha256="1234567890abcdef",
            page_count=2,
        )
        page1 = DocumentPage(page_number=1, image_path=self.page1_img)
        page2 = DocumentPage(page_number=2, image_path=self.page2_img)
        doc = Document(metadata=meta, pages=[page1, page2])

        result = self.processor.process_document(doc, processed_dir=self.root_path)

        # Verify DocumentOCRResult
        self.assertEqual(result.document_id, "doc_test_100")
        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.total_words, 4)
        self.assertAlmostEqual(result.mean_confidence, 0.925)
        self.assertIn("Page1 Invoice", result.full_text)
        self.assertIn("Page2 Invoice", result.full_text)

        # Verify DocumentPage items were updated in-place
        self.assertEqual(page1.raw_text, "Page1 Invoice")
        self.assertEqual(len(page1.ocr_text_regions), 1)
        self.assertEqual(page1.metadata["ocr_engine"], "mock_test_engine")

        self.assertEqual(page2.raw_text, "Page2 Invoice")
        self.assertEqual(len(page2.ocr_text_regions), 1)

    def test_single_page_failure_records_error_without_aborting(self) -> None:
        meta = DocumentMetadata(
            document_id="doc_test_101",
            filename="error_doc.pdf",
            file_path=self.root_path / "error_doc.pdf",
            file_type="application/pdf",
            file_size_bytes=5000,
            checksum_sha256="abcdef1234567890",
            page_count=2,
        )
        page1 = DocumentPage(page_number=1, image_path=self.page1_img)
        page_fail = DocumentPage(page_number=99, image_path=self.page2_img)
        doc = Document(metadata=meta, pages=[page1, page_fail])

        result = self.processor.process_document(doc, processed_dir=self.root_path)

        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.pages[0].raw_text, "Page1 Invoice")
        self.assertEqual(result.pages[1].raw_text, "")
        self.assertTrue(result.metadata["has_errors"])
        self.assertIn(99, result.metadata["page_errors"])


if __name__ == "__main__":
    unittest.main()
