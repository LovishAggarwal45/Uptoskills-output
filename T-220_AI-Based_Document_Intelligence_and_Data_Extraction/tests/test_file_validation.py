"""Unit tests for file existence, readability, size limits, and magic byte sniffing."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.core.exceptions import (
    EmptyFileError,
    FileSizeExceededError,
    UnsupportedFileTypeError,
)
from src.ingestion.validators import (
    compute_sha256,
    sniff_file_format,
    validate_file_exists_and_readable,
    validate_file_not_empty,
    validate_file_size_limit,
)


class TestFileValidation(unittest.TestCase):
    """Test suite covering file validation, size limits, and format sniffing."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_file_raises_filenotfound(self) -> None:
        missing_path = self.dir_path / "non_existent_doc.pdf"
        with self.assertRaises(FileNotFoundError):
            validate_file_exists_and_readable(missing_path)

    def test_empty_file_raises_emptyfileerror(self) -> None:
        empty_path = self.dir_path / "empty_doc.png"
        empty_path.touch()
        self.assertTrue(empty_path.exists())

        with self.assertRaises(EmptyFileError):
            validate_file_not_empty(empty_path)

    def test_file_size_exceeded_raises_error(self) -> None:
        large_path = self.dir_path / "large_doc.jpg"
        with open(large_path, "wb") as f:
            f.write(b"\x00" * 1024)  # 1 KB

        # Limit to 500 bytes
        with self.assertRaises(FileSizeExceededError):
            validate_file_size_limit(large_path, max_bytes=500)

        # Within 2 KB limit
        size = validate_file_size_limit(large_path, max_bytes=2048)
        self.assertEqual(size, 1024)

    def test_magic_bytes_png_sniffing(self) -> None:
        png_path = self.dir_path / "test_sample.png"
        img = Image.new("RGB", (50, 50), color="white")
        img.save(png_path, format="PNG")

        mime_type, ext = sniff_file_format(png_path)
        self.assertEqual(mime_type, "image/png")
        self.assertEqual(ext, ".png")

    def test_magic_bytes_jpeg_sniffing(self) -> None:
        jpg_path = self.dir_path / "test_sample.jpg"
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(jpg_path, format="JPEG")

        mime_type, ext = sniff_file_format(jpg_path)
        self.assertEqual(mime_type, "image/jpeg")
        self.assertEqual(ext, ".jpg")

    def test_magic_bytes_pdf_sniffing(self) -> None:
        pdf_path = self.dir_path / "dummy.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.7\n%Fake PDF binary header\n")

        mime_type, ext = sniff_file_format(pdf_path)
        self.assertEqual(mime_type, "application/pdf")
        self.assertEqual(ext, ".pdf")

    def test_unsupported_file_format_rejected(self) -> None:
        txt_path = self.dir_path / "document.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("This is a plain text file without supported magic bytes.")

        with self.assertRaises(UnsupportedFileTypeError):
            sniff_file_format(txt_path)

    def test_sha256_checksum_deterministic(self) -> None:
        file_path = self.dir_path / "data.bin"
        with open(file_path, "wb") as f:
            f.write(b"DocuMind AI Provenance Test Data")

        hash1 = compute_sha256(file_path)
        hash2 = compute_sha256(file_path)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)


if __name__ == "__main__":
    unittest.main()
