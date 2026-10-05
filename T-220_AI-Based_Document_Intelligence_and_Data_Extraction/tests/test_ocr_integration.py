"""Integration test running real OCR end-to-end when Tesseract is available, or cleanly verifying unavailable state."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.core.config import DocuMindConfig
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.exceptions import OCREngineUnavailableError
from src.ocr.processor import DocumentOCRProcessor
from src.ocr.tesseract_engine import TesseractOCREngine
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor


class TestOCRIntegration(unittest.TestCase):
    """End-to-end integration test: Ingestion -> Preprocessing -> Real Tesseract OCR."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.input_dir = self.root_path / "input"
        self.processed_dir = self.root_path / "processed"
        self.input_dir.mkdir(parents=True)
        self.processed_dir.mkdir(parents=True)

        self.config = DocuMindConfig()
        self.loader = DocumentLoader(
            config=self.config.ingestion,
            processed_dir=self.processed_dir,
        )
        self.preprocessor = OpenCVImagePreprocessor(
            config=self.config.preprocessing,
            quality_config=self.config.quality,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_invoice_image(self, path: Path) -> Path:
        """Create a clear, high-contrast synthetic document image with known text lines."""
        canvas = np.ones((800, 700, 3), dtype=np.uint8) * 255

        cv2.putText(canvas, "DOCUMIND AI", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas, "Invoice Number: INV-1001", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Date: 2026-09-20", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Item: Cloud Platform Subscription", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(canvas, "Total: 1250.00", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imwrite(str(path), canvas)
        return path

    def test_real_tesseract_ocr_integration_or_documented_skip(self) -> None:
        avail = check_ocr_availability(self.config.ocr)

        if not avail.is_ready:
            # Verify that calling real OCR raises OCREngineUnavailableError with diagnostic details
            engine = TesseractOCREngine(config=self.config.ocr)
            dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            with self.assertRaises(OCREngineUnavailableError) as ctx:
                engine.process_image(dummy_img, page_number=1)

            self.assertIn("tesseract", ctx.exception.engine_name)
            self.skipTest(
                f"Skipping live OCR execution: Tesseract binary not installed on host ({avail.status_message})"
            )

        # If Tesseract IS installed on host, execute full real OCR pipeline
        invoice_path = self.input_dir / "sample_invoice.png"
        self._create_synthetic_invoice_image(invoice_path)

        # 1. Ingestion
        document = self.loader.ingest(invoice_path, document_id="doc_ocr_live_01")
        self.assertEqual(len(document.pages), 1)

        # 2. Preprocessing
        self.preprocessor.preprocess_page(
            page=document.pages[0],
            output_dir=self.processed_dir / document.id,
        )

        # 3. Real OCR
        processor = DocumentOCRProcessor(
            config=self.config.ocr,
            use_preprocessed_images=True,
        )
        ocr_result = processor.process_document(document, processed_dir=self.processed_dir)

        # Verify OCR output properties
        self.assertGreater(ocr_result.total_words, 0)
        self.assertIsNotNone(ocr_result.mean_confidence)
        self.assertGreater(ocr_result.mean_confidence, 0.0)

        # Check recognized key tokens
        raw_text_upper = ocr_result.full_text.upper()
        self.assertIn("DOCUMIND", raw_text_upper)
        self.assertIn("INV-1001", raw_text_upper)
        self.assertIn("1250", raw_text_upper)

        # Verify DocumentPage mutation
        page = document.pages[0]
        self.assertTrue(len(page.ocr_text_regions) > 0)
        self.assertEqual(page.raw_text, ocr_result.pages[0].raw_text)


if __name__ == "__main__":
    unittest.main()
