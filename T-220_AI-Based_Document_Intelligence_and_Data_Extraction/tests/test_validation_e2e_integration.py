"""End-to-end integration test: Ingestion -> Preprocessing -> Real Tesseract OCR -> Classification -> Extraction -> Tables -> Validation."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.classification.classifier import RuleBasedDocumentClassifier
from src.core.config import DocuMindConfig
from src.core.types import DocumentType, ValidationStatus
from src.extraction.extractor import DocumentExtractor
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.processor import DocumentOCRProcessor
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor
from src.tables import TableExtractor
from src.validation.validator import DocumentValidationEngine


class TestValidationE2EIntegration(unittest.TestCase):
    """End-to-end pipeline verification test validating real OCR output and document validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.input_dir = self.root_path / "input"
        self.processed_dir = self.root_path / "processed"
        self.input_dir.mkdir(parents=True)
        self.processed_dir.mkdir(parents=True)

        self.config = DocuMindConfig()
        self.config.preprocessing.apply_deskew = False
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
        self.table_extractor = TableExtractor(config=self.config.tables)
        self.validation_engine = DocumentValidationEngine(config=self.config.validation)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_invoice_image(self, path: Path) -> Path:
        """Create a clear synthetic invoice image with standard billing headers and table."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (1000, 900), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_header = ImageFont.truetype("arial.ttf", 18)
            font_body = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font_title = font_header = font_body = ImageFont.load_default()

        # Headers
        draw.text((60, 40), "COMMERCIAL INVOICE", fill=(0, 0, 0), font=font_title)
        draw.text((60, 80), "Vendor: Apex Global Logistics Inc", fill=(0, 0, 0), font=font_body)
        draw.text((60, 110), "Invoice Number: INV-2026-7788", fill=(0, 0, 0), font=font_body)
        draw.text((60, 140), "Invoice Date: 2026-10-01", fill=(0, 0, 0), font=font_body)
        draw.text((60, 170), "Due Date: 2026-10-31", fill=(0, 0, 0), font=font_body)
        draw.text((60, 200), "Customer: Acme Corporation Ltd", fill=(0, 0, 0), font=font_body)

        # Table Header
        draw.text((60, 260), "Description", fill=(0, 0, 0), font=font_header)
        draw.text((450, 260), "Quantity", fill=(0, 0, 0), font=font_header)
        draw.text((600, 260), "Unit Price", fill=(0, 0, 0), font=font_header)
        draw.text((780, 260), "Amount", fill=(0, 0, 0), font=font_header)
        draw.line([(60, 290), (900, 290)], fill=(0, 0, 0), width=2)

        # Rows
        draw.text((60, 310), "Enterprise Cloud Hosting", fill=(0, 0, 0), font=font_body)
        draw.text((470, 310), "2", fill=(0, 0, 0), font=font_body)
        draw.text((620, 310), "$150.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 310), "$300.00", fill=(0, 0, 0), font=font_body)

        draw.text((60, 350), "Database Managed Backup", fill=(0, 0, 0), font=font_body)
        draw.text((470, 350), "1", fill=(0, 0, 0), font=font_body)
        draw.text((620, 350), "$80.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 350), "$80.00", fill=(0, 0, 0), font=font_body)

        draw.text((60, 390), "Network Security Firewall", fill=(0, 0, 0), font=font_body)
        draw.text((470, 390), "1", fill=(0, 0, 0), font=font_body)
        draw.text((620, 390), "$120.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 390), "$120.00", fill=(0, 0, 0), font=font_body)

        draw.line([(60, 430), (900, 430)], fill=(0, 0, 0), width=1)

        # Totals
        draw.text((600, 460), "Subtotal:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 460), "$500.00", fill=(0, 0, 0), font=font_header)

        draw.text((600, 500), "Tax:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 500), "$50.00", fill=(0, 0, 0), font=font_header)

        draw.text((600, 540), "Total Amount:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 540), "$550.00", fill=(0, 0, 0), font=font_header)

        img.save(str(path))
        return path

    def test_full_pipeline_ingestion_to_validation(self) -> None:
        avail = check_ocr_availability(self.config.ocr)
        if not avail.is_ready:
            self.skipTest(f"Skipping live OCR validation integration test: {avail.status_message}")

        # 1. Create document
        invoice_path = self.input_dir / "invoice_e2e.png"
        self._create_synthetic_invoice_image(invoice_path)

        # 2. Ingest
        document = self.loader.ingest(invoice_path, document_id="doc_int_val_01")
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
        self.assertEqual(extraction_result.document_type, DocumentType.INVOICE)
        self.assertIn("total", extraction_result.fields)

        # 7. Table Extraction
        table_result = self.table_extractor.extract_document(document)

        # 8. Document Validation & Consistency Engine
        val_report = self.validation_engine.validate_extraction_result(extraction_result, table_result)

        self.assertIsNotNone(val_report)
        self.assertEqual(val_report.document_id, "doc_int_val_01")
        self.assertEqual(val_report.document_type, DocumentType.INVOICE)
        self.assertIn(val_report.overall_status, (ValidationStatus.VALID, ValidationStatus.WARNING))
        self.assertGreaterEqual(val_report.validation_score, 0.70)
        self.assertEqual(val_report.error_count, 0)
        self.assertGreater(len(val_report.issues), 0)


if __name__ == "__main__":
    unittest.main()
