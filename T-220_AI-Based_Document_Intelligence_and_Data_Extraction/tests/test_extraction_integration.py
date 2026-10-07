"""End-to-end integration test: Ingestion -> Preprocessing -> Real Tesseract OCR -> Classification -> Extraction."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.classification.classifier import RuleBasedDocumentClassifier
from src.core.config import DocuMindConfig
from src.core.types import DocumentType, FieldType
from src.extraction.extractor import DocumentExtractor
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.processor import DocumentOCRProcessor
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor


class TestExtractionIntegration(unittest.TestCase):
    """Full end-to-end integration test through Phase 5 Structured Extraction."""

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
        self.extractor = DocumentExtractor(config=self.config.extraction)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_invoice_image(self, path: Path) -> Path:
        """Create a clear synthetic invoice image with standard billing headers."""
        canvas = np.ones((1000, 800, 3), dtype=np.uint8) * 255

        cv2.putText(canvas, "COMMERCIAL INVOICE", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, "Invoice Number: INV-2026-9810", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Invoice Date: 2026-10-01", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Due Date: 2026-10-31", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Bill To: Apex Global Logistics Inc", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Payment Terms: NET 30", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Purchase Order: PO-98124", (60, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Tax ID: VAT-98765432", (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Subtotal: $4,500.00", (60, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Tax: $450.00", (60, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, "Total Due: $4,950.00", (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imwrite(str(path), canvas)
        return path

    def test_full_pipeline_ingestion_ocr_classification_and_extraction(self) -> None:
        avail = check_ocr_availability(self.config.ocr)
        if not avail.is_ready:
            self.skipTest(f"Skipping live OCR extraction integration test: {avail.status_message}")

        # 1. Create document
        invoice_path = self.input_dir / "invoice_e2e.png"
        self._create_synthetic_invoice_image(invoice_path)

        # 2. Ingest
        document = self.loader.ingest(invoice_path, document_id="doc_int_extract_01")
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
        self.assertEqual(classification_result.document_type, DocumentType.INVOICE)

        # 6. Structured Extraction
        extraction_result = self.extractor.extract_document(document, classification_result)

        # Verify Extraction Results
        self.assertEqual(extraction_result.document_id, "doc_int_extract_01")
        self.assertEqual(extraction_result.document_type, DocumentType.INVOICE)
        self.assertGreater(len(extraction_result.fields), 0)

        # Verify Invoice Number
        inv_num_f = extraction_result.get_field("invoice_number")
        self.assertIsNotNone(inv_num_f)
        self.assertEqual(inv_num_f.normalized_value, "INV-2026-9810")
        self.assertEqual(inv_num_f.field_type, FieldType.IDENTIFIER)
        self.assertGreaterEqual(inv_num_f.extraction_confidence, 0.70)
        self.assertIsNotNone(inv_num_f.provenance.ocr_confidence)

        # Verify Total Amount
        total_f = extraction_result.get_field("total")
        self.assertIsNotNone(total_f)
        self.assertEqual(total_f.normalized_value, 4950.00)
        self.assertEqual(total_f.field_type, FieldType.CURRENCY)

        # Verify Subtotal
        subtotal_f = extraction_result.get_field("subtotal")
        self.assertIsNotNone(subtotal_f)
        self.assertEqual(subtotal_f.normalized_value, 4500.00)

        # Verify Dates
        inv_date_f = extraction_result.get_field("invoice_date")
        self.assertIsNotNone(inv_date_f)
        self.assertEqual(inv_date_f.normalized_value, "2026-10-01")

        # Verify Entities (money, date, etc.)
        self.assertGreater(len(extraction_result.entities), 0)

        # Verify JSON Export
        json_data = extraction_result.to_json()
        self.assertIn("INV-2026-9810", json_data)
        self.assertIn("4950.0", json_data)


if __name__ == "__main__":
    unittest.main()
