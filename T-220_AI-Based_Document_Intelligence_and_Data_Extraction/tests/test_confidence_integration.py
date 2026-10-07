"""End-to-end integration test: Ingestion -> Preprocessing -> OCR -> Classification -> Extraction -> Tables -> Validation -> Confidence & Human Review Routing."""

import tempfile
import unittest
from pathlib import Path

from src.classification.classifier import RuleBasedDocumentClassifier
from src.confidence.review_router import ReviewRouter
from src.confidence.models import ConfidenceBand, ReviewPriority, ReviewStatus
from src.core.config import DocuMindConfig
from src.core.types import DocumentType, ValidationStatus
from src.extraction.extractor import DocumentExtractor
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.processor import DocumentOCRProcessor
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor
from src.tables import TableExtractor
from src.validation.validator import DocumentValidationEngine


class TestConfidenceIntegration(unittest.TestCase):
    """End-to-end integration test verifying multi-tier confidence calculation and review routing with real OCR."""

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
        self.router = ReviewRouter(config=self.config.confidence)

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
        draw.text((60, 40), "TAX INVOICE", fill=(0, 0, 0), font=font_title)
        draw.text((60, 80), "Vendor: Global Cloud Technologies LLC", fill=(0, 0, 0), font=font_body)
        draw.text((60, 110), "Invoice Number: INV-2026-9021", fill=(0, 0, 0), font=font_body)
        draw.text((60, 140), "Invoice Date: 2026-10-01", fill=(0, 0, 0), font=font_body)
        draw.text((60, 170), "Due Date: 2026-10-31", fill=(0, 0, 0), font=font_body)
        draw.text((60, 200), "Customer: Acme Enterprise Corp", fill=(0, 0, 0), font=font_body)

        # Table Header
        draw.text((60, 260), "Description", fill=(0, 0, 0), font=font_header)
        draw.text((450, 260), "Quantity", fill=(0, 0, 0), font=font_header)
        draw.text((600, 260), "Unit Price", fill=(0, 0, 0), font=font_header)
        draw.text((780, 260), "Amount", fill=(0, 0, 0), font=font_header)
        draw.line([(60, 290), (900, 290)], fill=(0, 0, 0), width=2)

        # Rows
        draw.text((60, 310), "Dedicated Compute Node", fill=(0, 0, 0), font=font_body)
        draw.text((470, 310), "2", fill=(0, 0, 0), font=font_body)
        draw.text((620, 310), "$200.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 310), "$400.00", fill=(0, 0, 0), font=font_body)

        draw.text((60, 350), "High Speed Block Storage", fill=(0, 0, 0), font=font_body)
        draw.text((470, 350), "1", fill=(0, 0, 0), font=font_body)
        draw.text((620, 350), "$100.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 350), "$100.00", fill=(0, 0, 0), font=font_body)

        draw.line([(60, 390), (900, 390)], fill=(0, 0, 0), width=1)

        # Totals
        draw.text((600, 420), "Subtotal:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 420), "$500.00", fill=(0, 0, 0), font=font_header)

        draw.text((600, 460), "Tax:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 460), "$50.00", fill=(0, 0, 0), font=font_header)

        draw.text((600, 500), "Total Due:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 500), "$550.00", fill=(0, 0, 0), font=font_header)

        img.save(str(path))
        return path

    def test_full_pipeline_ingestion_to_review_routing(self) -> None:
        """Run complete 8-phase pipeline and verify confidence scoring & review router."""
        avail = check_ocr_availability(self.config.ocr)
        if not avail.is_ready:
            self.skipTest(f"Skipping live OCR integration test: {avail.status_message}")

        # 1. Ingest
        invoice_path = self.input_dir / "invoice_phase8_e2e.png"
        self._create_synthetic_invoice_image(invoice_path)
        document = self.loader.ingest(invoice_path, document_id="doc_p8_e2e_01")

        # 2. Preprocess
        self.preprocessor.preprocess_page(
            page=document.pages[0],
            output_dir=self.processed_dir / document.id,
        )

        # 3. Real OCR
        ocr_processor = DocumentOCRProcessor(
            config=self.config.ocr,
            use_preprocessed_images=True,
        )
        ocr_result = ocr_processor.process_document(document, processed_dir=self.processed_dir)
        self.assertGreater(ocr_result.total_words, 0)

        # 4. Classification
        classification_result = self.classifier.classify(document)
        self.assertEqual(classification_result.document_type, DocumentType.INVOICE)

        # 5. Extraction
        extraction_result = self.extractor.extract_document(document, classification_result)
        self.assertIn("total", extraction_result.fields)

        # 6. Tables
        table_result = self.table_extractor.extract_document(document)

        # 7. Validation
        val_report = self.validation_engine.validate_extraction_result(extraction_result, table_result)
        self.assertEqual(val_report.error_count, 0)

        # 8. Multi-Tier Confidence & Review Routing
        routing_result = self.router.route_document(
            document=document,
            fields=extraction_result.fields,
            tables=table_result.tables,
            validation_report=val_report,
        )

        # Assertions
        self.assertIsNotNone(routing_result)
        self.assertEqual(routing_result.document_id, "doc_p8_e2e_01")
        self.assertGreater(routing_result.document_confidence.overall_confidence, 0.70)
        self.assertIn(routing_result.document_confidence.confidence_band, (ConfidenceBand.HIGH, ConfidenceBand.MEDIUM))
        self.assertIsNotNone(routing_result.queue_item)

        # Verify field confidences
        for f_name, f_conf in routing_result.field_confidences.items():
            self.assertGreater(f_conf.aggregated_confidence, 0.0)
            self.assertIsNotNone(f_conf.confidence_band)
            self.assertIsNotNone(f_conf.explanation)

        # Verify review queue item
        queue_item = routing_result.queue_item
        self.assertEqual(queue_item.document_id, "doc_p8_e2e_01")
        self.assertIn(queue_item.priority, (ReviewPriority.LOW, ReviewPriority.MEDIUM, ReviewPriority.HIGH))
        self.assertIsNotNone(queue_item.status)

        # Verify JSON serialization works seamlessly
        json_output = routing_result.to_json()
        self.assertIn("doc_p8_e2e_01", json_output)
        self.assertIn("overall_confidence", json_output)


if __name__ == "__main__":
    unittest.main()
