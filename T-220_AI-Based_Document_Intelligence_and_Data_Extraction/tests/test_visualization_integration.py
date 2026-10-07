"""End-to-end integration tests verifying the full document pipeline through visual overlay generation."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from src.core.config import DocuMindConfig
from src.core.models import BoundingBox, Document, DocumentMetadata, DocumentPage
from src.core.types import DocumentType
from src.classification.classifier import RuleBasedDocumentClassifier
from src.confidence.review_router import ReviewRouter
from src.extraction.extractor import DocumentExtractor
from src.ingestion.document_loader import DocumentLoader
from src.ocr.availability import check_ocr_availability
from src.ocr.processor import DocumentOCRProcessor
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor
from src.tables.extractor import TableExtractor
from src.validation.validator import DocumentValidationEngine
from src.visualization.base import DocumentVisualizer
from src.visualization.models import VisualizationResult


class TestVisualizationIntegration(unittest.TestCase):
    """Integration test suite executing full pipeline from image through visual explainability."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        self.input_dir = self.base_dir / "data" / "input"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.output_vis_dir = self.base_dir / "outputs" / "visualizations"

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_vis_dir.mkdir(parents=True, exist_ok=True)

        # Create a synthetic invoice image fixture
        self.sample_image_path = self.input_dir / "test_invoice_vis.png"
        self._create_synthetic_invoice_image(self.sample_image_path)

        self.config = DocuMindConfig()
        self.config.storage.processed_dir = str(self.processed_dir)
        self.config.visualization.output.directory = str(self.output_vis_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_invoice_image(self, target_path: Path) -> None:
        """Draw a synthetic invoice document image for integration testing."""
        img = Image.new("RGB", (800, 1000), color="white")
        draw = ImageDraw.Draw(img)

        # Header
        draw.text((50, 40), "TAX INVOICE", fill="black")
        draw.text((50, 80), "Invoice Number: INV-2026-9901", fill="black")
        draw.text((50, 110), "Invoice Date: 2026-10-01", fill="black")
        draw.text((50, 140), "Due Date: 2026-10-31", fill="black")
        draw.text((50, 170), "Vendor: Acme Global Logistics LLC", fill="black")

        # Table Header
        draw.text((50, 240), "Description", fill="black")
        draw.text((300, 240), "Qty", fill="black")
        draw.text((450, 240), "Price", fill="black")
        draw.text((600, 240), "Amount", fill="black")
        draw.line([(50, 265), (750, 265)], fill="black", width=1)

        # Table Rows
        draw.text((50, 280), "Cloud Server Infrastructure", fill="black")
        draw.text((300, 280), "2", fill="black")
        draw.text((450, 280), "$1,000.00", fill="black")
        draw.text((600, 280), "$2,000.00", fill="black")

        draw.text((50, 320), "Dedicated Support Plan", fill="black")
        draw.text((300, 320), "1", fill="black")
        draw.text((450, 320), "$500.00", fill="black")
        draw.text((600, 320), "$500.00", fill="black")

        # Summary
        draw.line([(50, 360), (750, 360)], fill="black", width=1)
        draw.text((450, 380), "Subtotal:", fill="black")
        draw.text((600, 380), "$2,500.00", fill="black")
        draw.text((450, 410), "Tax:", fill="black")
        draw.text((600, 410), "$250.00", fill="black")
        draw.text((450, 440), "Total Due:", fill="black")
        draw.text((600, 440), "$2,750.00", fill="black")

        img.save(target_path, format="PNG")

    def test_full_pipeline_ingestion_to_visualization(self) -> None:
        """Verify full execution: Ingestion -> OCR -> Classification -> Extraction -> Tables -> Validation -> Confidence -> Visualization."""
        # 1. Ingestion
        loader = DocumentLoader(config=self.config.ingestion, processed_dir=self.processed_dir)
        doc = loader.ingest(self.sample_image_path, output_dir=self.processed_dir)
        self.assertEqual(len(doc.pages), 1)

        # 2. Preprocessing
        preprocessor = OpenCVImagePreprocessor(config=self.config.preprocessing)
        for p in doc.pages:
            res = preprocessor.preprocess_page(p, output_dir=self.processed_dir / doc.id)
            p.image_path = res.processed_image_path

        # 3. OCR (real Tesseract if available, else synthetic fallback)
        avail = check_ocr_availability(self.config.ocr)
        if avail.is_ready:
            ocr_proc = DocumentOCRProcessor(config=self.config.ocr)
            ocr_result = ocr_proc.process_document(doc, processed_dir=self.processed_dir)
        else:
            # Synthetic OCR words if Tesseract unavailable in environment
            words = [
                OCRWord("TAX", 0.99, 99.0, BoundingBox(50, 40, 90, 60), 1),
                OCRWord("INVOICE", 0.99, 99.0, BoundingBox(95, 40, 160, 60), 1),
                OCRWord("INV-2026-9901", 0.98, 98.0, BoundingBox(180, 80, 300, 100), 1),
                OCRWord("2026-10-01", 0.98, 98.0, BoundingBox(160, 110, 260, 130), 1),
                OCRWord("$2,750.00", 0.98, 98.0, BoundingBox(600, 440, 690, 460), 1),
            ]
            doc.pages[0].raw_text = "TAX INVOICE\nInvoice Number: INV-2026-9901\nInvoice Date: 2026-10-01\nTotal Due: $2,750.00"
            doc.pages[0].metadata["ocr_words"] = words
            ocr_result = None

        # 4. Classification
        classifier = RuleBasedDocumentClassifier(config=self.config.classification)
        clf_result = classifier.classify(doc)
        doc.classified_type = clf_result.document_type
        doc.classification_confidence = clf_result.confidence

        # 5. Extraction
        extractor = DocumentExtractor(config=self.config.extraction)
        ext_result = extractor.extract(doc, clf_result)

        # 6. Table Extraction
        table_extractor = TableExtractor(config=self.config.tables)
        tbl_result = table_extractor.extract(doc)

        # 7. Validation
        val_engine = DocumentValidationEngine(config=self.config.validation)
        val_report = val_engine.validate_document(
            document_id=doc.id,
            document_type=doc.classified_type,
            fields=ext_result.fields,
            tables=tbl_result.tables,
        )

        # 8. Confidence & Review Routing
        router = ReviewRouter(config=self.config.confidence)
        routing_result = router.route_document(
            document=doc,
            fields=ext_result.fields,
            tables=tbl_result.tables,
            validation_report=val_report,
        )

        # 9. Visualization & Visual Explainability (Phase 9)
        visualizer = DocumentVisualizer(config=self.config.visualization)
        vis_result = visualizer.visualize_document(
            document=doc,
            fields=ext_result.fields,
            entities=ext_result.entities,
            tables=tbl_result.tables,
            validation_report=val_report,
            routing_result=routing_result,
            ocr_result=ocr_result,
            output_dir=self.output_vis_dir,
        )

        # Verifications
        self.assertIsInstance(vis_result, VisualizationResult)
        self.assertEqual(vis_result.document_id, doc.id)
        self.assertEqual(len(vis_result.page_visualizations), 1)
        self.assertEqual(vis_result.page_visualizations[0].rendering_status, "rendered")

        # Verify overlay artifact file exists
        overlay_path = vis_result.page_visualizations[0].overlay_image_path
        self.assertIsNotNone(overlay_path)
        self.assertTrue(overlay_path.exists())

        # Verify overlay image properties
        with Image.open(overlay_path) as img:
            self.assertEqual(img.width, vis_result.page_visualizations[0].image_width)
            self.assertGreaterEqual(img.height, vis_result.page_visualizations[0].image_height)

        # Verify evidence manifest exists
        manifest_path = vis_result.artifact_paths.get("evidence_manifest")
        self.assertIsNotNone(manifest_path)
        self.assertTrue(manifest_path.exists())

        # Verify summary file exists
        summary_path = vis_result.artifact_paths.get("visualization_summary")
        self.assertIsNotNone(summary_path)
        self.assertTrue(summary_path.exists())

        # Verify original source image was preserved
        self.assertTrue(self.sample_image_path.exists())


if __name__ == "__main__":
    unittest.main()
