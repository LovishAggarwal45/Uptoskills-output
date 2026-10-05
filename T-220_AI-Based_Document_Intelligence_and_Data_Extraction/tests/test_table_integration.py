"""End-to-end integration tests for Table intelligence using real OCR and synthetic documents."""

import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.config import DocuMindConfig, OCRConfig
from src.core.models import Document, DocumentMetadata, DocumentPage
from src.ocr.availability import check_ocr_availability
from src.ocr.tesseract_engine import TesseractOCREngine
from src.tables.extractor import TableExtractor


class TestTableIntegration(unittest.TestCase):
    """End-to-end integration test: Synthetic document rendering -> Real OCR -> Table intelligence."""

    def setUp(self) -> None:
        self.config = DocuMindConfig()
        self.ocr_avail = check_ocr_availability(self.config.ocr)

    def _generate_synthetic_invoice_table_image(self) -> Image.Image:
        """Create a clean, high-resolution document image with a clear 4-column data table."""
        # 1000 x 800 white canvas
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Standard system font or default
        try:
            font_title = ImageFont.truetype("arial.ttf", 26)
            font_header = ImageFont.truetype("arial.ttf", 20)
            font_body = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_body = ImageFont.load_default()

        # Document Header
        draw.text((60, 50), "COMMERCIAL INVOICE", fill=(0, 0, 0), font=font_title)
        draw.text((60, 90), "Invoice Number: INV-2026-7788", fill=(0, 0, 0), font=font_body)
        draw.text((60, 115), "Invoice Date: 2026-10-01", fill=(0, 0, 0), font=font_body)

        # Table Column Coordinates
        # Col 0: Description (x: 60)
        # Col 1: Quantity    (x: 450)
        # Col 2: Unit Price  (x: 600)
        # Col 3: Amount      (x: 780)

        # Header Row (y: 180)
        draw.text((60, 180), "Description", fill=(0, 0, 0), font=font_header)
        draw.text((450, 180), "Quantity", fill=(0, 0, 0), font=font_header)
        draw.text((600, 180), "Unit Price", fill=(0, 0, 0), font=font_header)
        draw.text((780, 180), "Amount", fill=(0, 0, 0), font=font_header)
        draw.line([(60, 210), (900, 210)], fill=(0, 0, 0), width=2)

        # Row 1 (y: 230)
        draw.text((60, 230), "Enterprise Cloud Hosting", fill=(0, 0, 0), font=font_body)
        draw.text((470, 230), "2", fill=(0, 0, 0), font=font_body)
        draw.text((620, 230), "$150.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 230), "$300.00", fill=(0, 0, 0), font=font_body)

        # Row 2 (y: 280)
        draw.text((60, 280), "Database Managed Backup", fill=(0, 0, 0), font=font_body)
        draw.text((470, 280), "1", fill=(0, 0, 0), font=font_body)
        draw.text((620, 280), "$80.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 280), "$80.00", fill=(0, 0, 0), font=font_body)

        # Row 3 (y: 330)
        draw.text((60, 330), "Network Security Firewall", fill=(0, 0, 0), font=font_body)
        draw.text((470, 330), "1", fill=(0, 0, 0), font=font_body)
        draw.text((620, 330), "$120.00", fill=(0, 0, 0), font=font_body)
        draw.text((800, 330), "$120.00", fill=(0, 0, 0), font=font_body)

        draw.line([(60, 370), (900, 370)], fill=(0, 0, 0), width=1)

        # Summary Row (y: 390)
        draw.text((600, 390), "Subtotal:", fill=(0, 0, 0), font=font_header)
        draw.text((800, 390), "$500.00", fill=(0, 0, 0), font=font_header)

        return img

    def test_real_ocr_to_table_extraction_pipeline(self) -> None:
        """Verify complete pipeline from synthetic image through real OCR and table extraction."""
        if not self.ocr_avail.is_ready:
            self.skipTest("Tesseract OCR is not installed or available in this environment.")

        # 1. Generate image
        img = self._generate_synthetic_invoice_table_image()
        np_img = np.array(img)

        # 2. Run real OCR
        engine = TesseractOCREngine(config=self.config.ocr)
        page_ocr = engine.process_image(np_img, page_number=1)

        self.assertGreater(len(page_ocr.words), 10)
        self.assertIn("Description", page_ocr.raw_text)

        # 3. Construct DocumentPage with OCR results
        meta = DocumentMetadata(
            document_id="doc_integration_test_01",
            filename="synthetic_table_invoice.png",
            file_path=Path("synthetic_table_invoice.png"),
            file_type="image/png",
            file_size_bytes=len(np_img),
            checksum_sha256="test_sha",
            page_count=1,
        )

        page = DocumentPage(
            page_number=1,
            width=float(img.width),
            height=float(img.height),
            ocr_text_regions=page_ocr.text_regions,
            raw_text=page_ocr.raw_text,
            metadata={"ocr_words": page_ocr.words},
        )

        doc = Document(metadata=meta, pages=[page])

        # 4. Run Table Intelligence Extraction
        table_extractor = TableExtractor(config=self.config.tables)
        result = table_extractor.extract(doc)

        # 5. Verify extracted table and line items
        self.assertEqual(len(result.tables), 1)
        table = result.tables[0]

        # Verify columns
        self.assertGreaterEqual(len(table.columns), 3)

        # Verify line items extracted from REAL OCR
        self.assertGreaterEqual(len(table.line_items), 2)

        # Verify arithmetic validation executed on real OCR data
        self.assertIsNotNone(table.validation_result)
        self.assertGreaterEqual(table.validation_result.row_checks_passed, 1)

        # Verify explainable composite confidence
        self.assertGreaterEqual(table.confidence, 0.70)
        self.assertIn("header_recognition", table.confidence_breakdown)
        self.assertIn("ocr_confidence", table.confidence_breakdown)


if __name__ == "__main__":
    unittest.main()
