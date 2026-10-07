"""Unit and integration tests for multi-format universal ingestion in DocuMind AI."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from pypdf import PdfWriter

from src.core.config import IngestionConfig
from src.core.exceptions import (
    EmptyFileError,
    FileSizeExceededError,
    UnsupportedFileTypeError,
)
from src.core.models import Document
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.validators import (
    MAGIC_BYTES_SIGNATURES,
    detect_poppler_path,
    sniff_file_format,
    validate_file_not_empty,
    validate_file_size_limit,
)


class TestUniversalIngestion(unittest.TestCase):
    """Test multi-format document loading, frame extraction, and validation rules."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = IngestionConfig(max_file_size_bytes=10 * 1024 * 1024)
        self.loader = DocumentLoader(config=self.config, processed_dir=self.output_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sniff_file_format_all_types(self) -> None:
        """Verify sniff_file_format detects PDF, PNG, JPG, TIFF, and BMP header bytes."""
        # 1. PDF
        pdf_p = self.temp_dir / "sample.pdf"
        pdf_p.write_bytes(MAGIC_BYTES_SIGNATURES["application/pdf"] + b" dummy pdf content")
        mime, ext = sniff_file_format(pdf_p)
        self.assertEqual(mime, "application/pdf")
        self.assertEqual(ext, ".pdf")

        # 2. PNG
        png_p = self.temp_dir / "sample.png"
        png_p.write_bytes(MAGIC_BYTES_SIGNATURES["image/png"] + b" dummy png content")
        mime, ext = sniff_file_format(png_p)
        self.assertEqual(mime, "image/png")
        self.assertEqual(ext, ".png")

        # 3. JPEG
        jpg_p = self.temp_dir / "sample.jpg"
        jpg_p.write_bytes(MAGIC_BYTES_SIGNATURES["image/jpeg"] + b" dummy jpeg content")
        mime, ext = sniff_file_format(jpg_p)
        self.assertEqual(mime, "image/jpeg")
        self.assertEqual(ext, ".jpg")

        # 4. TIFF (little-endian and big-endian)
        tif_le = self.temp_dir / "sample_le.tiff"
        tif_le.write_bytes(MAGIC_BYTES_SIGNATURES["image/tiff_le"] + b" dummy tiff content")
        mime, ext = sniff_file_format(tif_le)
        self.assertEqual(mime, "image/tiff")
        self.assertEqual(ext, ".tiff")

        tif_be = self.temp_dir / "sample_be.tiff"
        tif_be.write_bytes(MAGIC_BYTES_SIGNATURES["image/tiff_be"] + b" dummy tiff content")
        mime, ext = sniff_file_format(tif_be)
        self.assertEqual(mime, "image/tiff")

        # 5. BMP
        bmp_p = self.temp_dir / "sample.bmp"
        bmp_p.write_bytes(MAGIC_BYTES_SIGNATURES["image/bmp"] + b" dummy bmp content")
        mime, ext = sniff_file_format(bmp_p)
        self.assertEqual(mime, "image/bmp")
        self.assertEqual(ext, ".bmp")

    def test_spoofed_file_extension_rejection(self) -> None:
        """Verify that spoofed extensions (e.g. text/exe file named as .png or .pdf) are rejected."""
        spoofed = self.temp_dir / "malicious.png"
        spoofed.write_bytes(b"MZ\x90\x00 This is an executable disguised as PNG")
        with self.assertRaises(UnsupportedFileTypeError):
            sniff_file_format(spoofed)

    def test_empty_file_rejection(self) -> None:
        """Verify that 0-byte files raise EmptyFileError."""
        empty_p = self.temp_dir / "empty.pdf"
        empty_p.write_bytes(b"")
        with self.assertRaises(EmptyFileError):
            validate_file_not_empty(empty_p)

    def test_file_size_exceeded_rejection(self) -> None:
        """Verify that oversized files raise FileSizeExceededError."""
        oversized_p = self.temp_dir / "large.pdf"
        oversized_p.write_bytes(b"x" * 2000)
        with self.assertRaises(FileSizeExceededError):
            validate_file_size_limit(oversized_p, max_bytes=1000)

    def test_ingest_png_image(self) -> None:
        """Verify ingesting a valid PNG image generates a single DocumentPage with correct dimensions."""
        img_p = self.temp_dir / "test_doc.png"
        img = Image.new("RGB", (300, 400), color="white")
        img.save(img_p, format="PNG")

        doc = self.loader.ingest(img_p, document_id="doc_test_png", output_dir=self.output_dir)
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.id, "doc_test_png")
        self.assertEqual(doc.metadata.page_count, 1)
        self.assertEqual(len(doc.pages), 1)
        self.assertEqual(doc.pages[0].width, 300.0)
        self.assertEqual(doc.pages[0].height, 400.0)
        self.assertTrue(Path(doc.pages[0].image_path).exists())

    def test_ingest_jpeg_image(self) -> None:
        """Verify ingesting a JPEG image generates a clean standardized PNG page image."""
        img_p = self.temp_dir / "test_doc.jpg"
        img = Image.new("RGB", (250, 350), color="blue")
        img.save(img_p, format="JPEG")

        doc = self.loader.ingest(img_p, document_id="doc_test_jpg", output_dir=self.output_dir)
        self.assertEqual(doc.metadata.page_count, 1)
        self.assertEqual(doc.pages[0].width, 250.0)
        self.assertEqual(doc.pages[0].height, 350.0)
        self.assertTrue(Path(doc.pages[0].image_path).exists())

    def test_ingest_bmp_image(self) -> None:
        """Verify ingesting a BMP image converts and preserves page metadata."""
        img_p = self.temp_dir / "test_doc.bmp"
        img = Image.new("RGB", (200, 200), color="red")
        img.save(img_p, format="BMP")

        doc = self.loader.ingest(img_p, document_id="doc_test_bmp", output_dir=self.output_dir)
        self.assertEqual(doc.metadata.page_count, 1)
        self.assertEqual(doc.pages[0].width, 200.0)
        self.assertTrue(Path(doc.pages[0].image_path).exists())

    def test_ingest_multipage_tiff(self) -> None:
        """Verify ingesting a multi-frame TIFF extracts all pages into numbered image files."""
        tif_p = self.temp_dir / "multipage.tiff"
        frame1 = Image.new("RGB", (200, 300), color="white")
        frame2 = Image.new("RGB", (200, 300), color="yellow")
        frame3 = Image.new("RGB", (200, 300), color="green")

        frame1.save(
            tif_p,
            format="TIFF",
            save_all=True,
            append_images=[frame2, frame3],
        )

        doc = self.loader.ingest(tif_p, document_id="doc_test_tiff", output_dir=self.output_dir)
        self.assertEqual(doc.metadata.page_count, 3)
        self.assertEqual(len(doc.pages), 3)
        for i, page in enumerate(doc.pages, start=1):
            self.assertEqual(page.page_number, i)
            self.assertTrue(Path(page.image_path).exists())


if __name__ == "__main__":
    unittest.main()
