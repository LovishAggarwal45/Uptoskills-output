"""End-to-end integration test connecting Document Ingestion and Image Preprocessing."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.core.config import DocuMindConfig, PreprocessingConfig, QualityConfig
from src.core.models import Document, DocumentPage
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.validators import compute_sha256
from src.preprocessing.deskew import rotate_image
from src.preprocessing.opencv_preprocessor import OpenCVImagePreprocessor


class TestIngestionPreprocessingIntegration(unittest.TestCase):
    """Integration test suite executing full Ingestion -> Staging -> Preprocessing -> Quality flow."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.input_dir = self.root_path / "input"
        self.processed_dir = self.root_path / "processed"
        self.input_dir.mkdir(parents=True)
        self.processed_dir.mkdir(parents=True)

        self.config = DocuMindConfig()
        self.config.preprocessing.enabled = True
        self.config.preprocessing.apply_grayscale = True
        self.config.preprocessing.apply_deskew = True
        self.config.preprocessing.apply_denoising = True
        self.config.preprocessing.apply_contrast_enhancement = True
        self.config.preprocessing.resize_max_dimension = 2000

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

    def _create_synthetic_skewed_invoice(self, path: Path) -> Path:
        """Render a realistic synthetic invoice image with text lines and slight skew."""
        canvas = np.ones((1200, 900, 3), dtype=np.uint8) * 255

        # Header
        cv2.putText(canvas, "ACME SUPPLIES INC. - INVOICE", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas, "Invoice #: INV-2026-8891", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(canvas, "Date: 2026-10-01", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        # Line items
        for y in range(300, 700, 60):
            cv2.putText(canvas, "Widget Description Item Model X", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(canvas, "$120.00", (700, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Total line
        cv2.putText(canvas, "Total Due: $600.00", (550, 850), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Apply -3.0 degree skew
        skewed = rotate_image(canvas, -3.0)
        cv2.imwrite(str(path), skewed)
        return path

    def test_complete_ingestion_and_preprocessing_chain(self) -> None:
        # Step 1: Prepare synthetic source document
        source_doc_path = self.input_dir / "sample_invoice.png"
        self._create_synthetic_skewed_invoice(source_doc_path)
        original_hash = compute_sha256(source_doc_path)

        # Step 2: Ingestion & Staging
        document = self.loader.ingest(
            file_path=source_doc_path,
            document_id="doc_integration_001",
        )

        # Verify source file integrity was strictly preserved
        self.assertEqual(compute_sha256(source_doc_path), original_hash)

        self.assertIsInstance(document, Document)
        self.assertEqual(document.id, "doc_integration_001")
        self.assertEqual(len(document.pages), 1)

        page_1 = document.pages[0]
        self.assertEqual(page_1.page_number, 1)
        self.assertTrue(Path(page_1.image_path).exists())
        self.assertEqual(Path(page_1.image_path).name, "page_001_original.png")

        # Step 3: Preprocessing
        prep_result = self.preprocessor.preprocess_page(
            page=page_1,
            output_dir=self.processed_dir / document.id,
        )

        self.assertEqual(prep_result.page_number, 1)
        self.assertTrue(prep_result.is_modified)
        self.assertTrue(Path(prep_result.processed_image_path).exists())
        self.assertEqual(Path(prep_result.processed_image_path).name, "page_001_processed.png")

        # Verify applied operations include deskew and grayscale
        self.assertTrue(any("deskew" in op for op in prep_result.applied_operations))
        self.assertTrue(any("grayscale" in op for op in prep_result.applied_operations))

        # Verify quality metrics are populated and flagged as heuristic
        metrics = prep_result.metrics
        self.assertIn("initial_quality", metrics)
        self.assertIn("final_quality", metrics)
        self.assertTrue(metrics["initial_quality"]["is_heuristic"])
        self.assertTrue(metrics["final_quality"]["is_heuristic"])

        # Check that both original and processed image files coexist on disk
        self.assertTrue((self.processed_dir / document.id / "page_001_original.png").exists())
        self.assertTrue((self.processed_dir / document.id / "page_001_processed.png").exists())


if __name__ == "__main__":
    unittest.main()
