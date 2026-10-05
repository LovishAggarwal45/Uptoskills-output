"""Unit tests for PDF structure parsing, page counting, and Poppler dependency handling."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pypdf import PageObject, PdfWriter

from src.core.config import IngestionConfig
from src.core.exceptions import CorruptedDocumentError, PopplerNotFoundError
from src.ingestion.pdf_ingestion import PDFIngestionHandler
from src.ingestion.validators import detect_poppler_path, validate_pdf_structure


class TestPDFIngestion(unittest.TestCase):
    """Test suite for PDF metadata extraction and rasterization dependency management."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.input_dir = self.root_path / "input"
        self.output_dir = self.root_path / "processed"
        self.input_dir.mkdir()
        self.output_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_synthetic_pdf(self, path: Path, num_pages: int = 1) -> Path:
        """Generate a valid, minimal multi-page PDF for test assertions."""
        writer = PdfWriter()
        for i in range(num_pages):
            page = PageObject.create_blank_page(width=612, height=792)  # Standard Letter size
            writer.add_page(page)

        writer.add_metadata({"/Title": "DocuMind Test PDF", "/Author": "DocuMind Unit Tests"})

        with open(path, "wb") as f:
            writer.write(f)
        return path

    def test_single_page_pdf_structure_validation(self) -> None:
        pdf_path = self.input_dir / "single_page.pdf"
        self._create_synthetic_pdf(pdf_path, num_pages=1)

        page_count, is_encrypted, metadata = validate_pdf_structure(pdf_path)
        self.assertEqual(page_count, 1)
        self.assertFalse(is_encrypted)
        self.assertEqual(metadata.get("Title"), "DocuMind Test PDF")

    def test_multi_page_pdf_structure_validation(self) -> None:
        pdf_path = self.input_dir / "multi_page_invoice.pdf"
        self._create_synthetic_pdf(pdf_path, num_pages=3)

        page_count, is_encrypted, _ = validate_pdf_structure(pdf_path)
        self.assertEqual(page_count, 3)
        self.assertFalse(is_encrypted)

    def test_corrupted_pdf_structure_raises_error(self) -> None:
        corrupted_path = self.input_dir / "invalid.pdf"
        with open(corrupted_path, "wb") as f:
            f.write(b"%PDF-1.4\nBrokenTruncatedPDFStreamWithoutTrailer")

        with self.assertRaises(CorruptedDocumentError):
            validate_pdf_structure(corrupted_path)

    def test_detect_poppler_path_with_valid_directory(self) -> None:
        """Test detect_poppler_path when given a directory containing pdftoppm."""
        bin_dir = self.root_path / "mock_poppler_bin"
        bin_dir.mkdir()
        dummy_exe = bin_dir / "pdftoppm.exe" if os.name == "nt" else bin_dir / "pdftoppm"
        dummy_exe.touch()

        detected = detect_poppler_path(str(bin_dir))
        self.assertIsNotNone(detected)
        self.assertEqual(Path(detected).resolve(), bin_dir.resolve())

    def test_detect_poppler_path_with_executable_file_path(self) -> None:
        """Test detect_poppler_path when given a path directly pointing to pdftoppm.exe."""
        bin_dir = self.root_path / "mock_poppler_exec"
        bin_dir.mkdir()
        dummy_exe = bin_dir / "pdftoppm.exe" if os.name == "nt" else bin_dir / "pdftoppm"
        dummy_exe.touch()

        detected = detect_poppler_path(str(dummy_exe))
        self.assertIsNotNone(detected)
        # Should return the parent directory containing the executable
        self.assertEqual(Path(detected).resolve(), bin_dir.resolve())

    def test_detect_poppler_path_with_nested_windows_release_structure(self) -> None:
        """Test detect_poppler_path with Windows release layout containing Library/bin/pdftoppm.exe."""
        root_poppler = self.root_path / "poppler-release"
        lib_bin = root_poppler / "Library" / "bin"
        lib_bin.mkdir(parents=True)
        dummy_exe = lib_bin / "pdftoppm.exe"
        dummy_exe.touch()

        detected = detect_poppler_path(str(root_poppler))
        self.assertIsNotNone(detected)
        self.assertEqual(Path(detected).resolve(), lib_bin.resolve())

    def test_detect_poppler_path_with_quoted_windows_path(self) -> None:
        """Test detect_poppler_path strips outer quotes commonly present in Windows configs."""
        bin_dir = self.root_path / "mock_poppler_quoted"
        bin_dir.mkdir()
        dummy_exe = bin_dir / "pdftoppm.exe" if os.name == "nt" else bin_dir / "pdftoppm"
        dummy_exe.touch()

        quoted = f'"{bin_dir}"'
        detected = detect_poppler_path(quoted)
        self.assertIsNotNone(detected)
        self.assertEqual(Path(detected).resolve(), bin_dir.resolve())

    def test_detect_poppler_path_with_environment_variable(self) -> None:
        """Test detect_poppler_path resolves from POPPLER_PATH environment variable."""
        bin_dir = self.root_path / "mock_poppler_env"
        bin_dir.mkdir()
        dummy_exe = bin_dir / "pdftoppm.exe" if os.name == "nt" else bin_dir / "pdftoppm"
        dummy_exe.touch()

        with patch.dict(os.environ, {"POPPLER_PATH": str(bin_dir)}):
            detected = detect_poppler_path(None)
            self.assertIsNotNone(detected)
            self.assertEqual(Path(detected).resolve(), bin_dir.resolve())

    def test_detect_poppler_path_missing_returns_none(self) -> None:
        """Test detect_poppler_path returns None when poppler is not found."""
        with patch.dict(os.environ, {}, clear=True):
            detected = detect_poppler_path("/definitely/nonexistent/poppler/path/12345")
            self.assertIsNone(detected)

    def test_poppler_diagnostic_and_handling(self) -> None:
        # Provide an explicitly nonexistent poppler path
        handler = PDFIngestionHandler(
            config=IngestionConfig(poppler_path="/nonexistent/poppler/bin")
        )
        is_available, _ = handler.check_poppler_availability()

        pdf_path = self.input_dir / "sample.pdf"
        self._create_synthetic_pdf(pdf_path, num_pages=1)

        if not is_available:
            with self.assertRaises(PopplerNotFoundError) as ctx:
                handler.ingest_pdf(
                    file_path=pdf_path,
                    document_id="doc_pdf_001",
                    output_dir=self.output_dir,
                )
            self.assertIn("Poppler", str(ctx.exception))
            self.assertIn("pdftoppm", str(ctx.exception))
        else:
            # If Poppler happens to be installed in PATH on host
            doc = handler.ingest_pdf(
                file_path=pdf_path,
                document_id="doc_pdf_001",
                output_dir=self.output_dir,
            )
            self.assertEqual(len(doc.pages), 1)

    def test_pdf_ingestion_with_system_poppler(self) -> None:
        """Test PDF ingestion end-to-end when Poppler is configured/detected."""
        handler = PDFIngestionHandler()
        is_available, poppler_bin = handler.check_poppler_availability()
        if not is_available:
            self.skipTest("Poppler not installed in current environment")

        pdf_path = self.input_dir / "invoice_multi.pdf"
        self._create_synthetic_pdf(pdf_path, num_pages=2)

        doc = handler.ingest_pdf(
            file_path=pdf_path,
            document_id="doc_test_system_poppler",
            output_dir=self.output_dir,
        )
        self.assertEqual(len(doc.pages), 2)
        self.assertEqual(doc.page_count, 2)
        for page in doc.pages:
            self.assertTrue(Path(page.image_path).exists())
            self.assertGreater(page.width, 0)
            self.assertGreater(page.height, 0)


if __name__ == "__main__":
    unittest.main()

