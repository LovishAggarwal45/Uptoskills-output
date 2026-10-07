"""End-to-end integration test: Ingestion -> Preprocessing -> Real Tesseract OCR -> Classification."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.classification.classifier import RuleBasedDocumentClassifier
from src.core.config import DocuMindConfig
from src.core.types import DocumentType
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.processor import DocumentOCRProcessor
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor


class TestClassificationIntegration(unittest.TestCase):
    """End-to-end integration pipeline test including document classification."""

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
        self.classifier = RuleBasedDocumentClassifier(config=self.config.classification)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_invoice_image(self, path: Path) -> Path:
        """Create a clear synthetic invoice image with standard billing headers."""
        canvas = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        cv2.putText(canvas, "TAX INVOICE", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, "Invoice Number: INV-2026-9810", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Invoice Date: 2026-10-01", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Bill To: Apex Global Logistics Inc", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Payment Terms: NET 30", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Subtotal: 4500.00", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Total Due: 4950.00", (60, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imwrite(str(path), canvas)
        return path

    def test_full_pipeline_ingestion_ocr_and_classification(self) -> None:
        avail = check_ocr_availability(self.config.ocr)
        if not avail.is_ready:
            self.skipTest(f"Skipping live OCR classification integration test: {avail.status_message}")

        # 1. Create synthetic document image
        invoice_path = self.input_dir / "invoice_test.png"
        self._create_synthetic_invoice_image(invoice_path)

        # 2. Ingest
        document = self.loader.ingest(invoice_path, document_id="doc_int_class_01")
        self.assertEqual(len(document.pages), 1)

        # 3. Preprocess
        self.preprocessor.preprocess_page(
            page=document.pages[0],
            output_dir=self.processed_dir / document.id,
        )

        # 4. OCR
        ocr_processor = DocumentOCRProcessor(
            config=self.config.ocr,
            use_preprocessed_images=True,
        )
        ocr_result = ocr_processor.process_document(document, processed_dir=self.processed_dir)
        self.assertGreater(ocr_result.total_words, 0)

        # 5. Classification
        classification_result = self.classifier.classify(document)

        # Verify classification outcome
        self.assertEqual(classification_result.document_type, DocumentType.INVOICE)
        self.assertGreaterEqual(classification_result.confidence, 0.60)
        self.assertEqual(document.classified_type, DocumentType.INVOICE)
        self.assertEqual(document.classification_confidence, classification_result.confidence)
        self.assertEqual(document.classification_method, "rule_based")

        # Verify evidence contains real bounding box provenance from OCR
        ev_list = classification_result.get_top_evidence()
        self.assertGreater(len(ev_list), 0)
        first_ev = ev_list[0]
        self.assertIsNotNone(first_ev.bounding_box)
        self.assertIsNotNone(first_ev.ocr_confidence)
        self.assertGreater(first_ev.ocr_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
