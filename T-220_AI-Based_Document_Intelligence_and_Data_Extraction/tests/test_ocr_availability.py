"""Unit tests for OCR engine availability detection and environment diagnostics."""

import os
import tempfile
import unittest
from pathlib import Path

from src.core.config import OCRConfig
from src.ocr.availability import (
    OCRAvailabilityResult,
    check_ocr_availability,
    find_tesseract_binary,
)


class TestOCRAvailability(unittest.TestCase):
    """Test suite covering Tesseract discovery, version checking, and availability diagnostics."""

    def test_find_tesseract_binary_with_custom_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".exe" if os.name == "nt" else "", delete=False) as f:
            temp_path = f.name

        try:
            # Make file executable
            os.chmod(temp_path, 0o755)
            found = find_tesseract_binary(temp_path)
            self.assertEqual(Path(found).resolve(), Path(temp_path).resolve())
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_check_ocr_availability_diagnostic_output(self) -> None:
        cfg = OCRConfig(languages=["eng"])
        avail = check_ocr_availability(cfg)

        self.assertIsInstance(avail, OCRAvailabilityResult)
        self.assertEqual(avail.requested_language, "eng")
        self.assertIsInstance(avail.available_languages, list)

        avail_dict = avail.to_dict()
        self.assertIn("is_wrapper_available", avail_dict)
        self.assertIn("is_engine_available", avail_dict)
        self.assertIn("is_ready", avail_dict)
        self.assertIn("status_message", avail_dict)

        # Environment-aware assertions
        if avail.is_engine_available:
            # Tesseract binary is installed and detected on host
            self.assertTrue(avail.is_wrapper_available)
            self.assertIsNotNone(avail.executable_path)
            self.assertTrue(Path(avail.executable_path).exists())
            self.assertTrue(avail.is_language_available)
            self.assertTrue(avail.is_ready)
            self.assertIn("READY", avail.status_message)
            self.assertIsNotNone(avail.engine_version)
        else:
            # Tesseract binary is not present on host
            self.assertFalse(avail.is_ready)
            self.assertIn("Tesseract", avail.status_message)

    def test_check_ocr_availability_missing_language(self) -> None:
        cfg_bad_lang = OCRConfig(languages=["nonexistent_lang_xyz_999"])
        avail_bad = check_ocr_availability(cfg_bad_lang)

        self.assertIsInstance(avail_bad, OCRAvailabilityResult)
        self.assertFalse(avail_bad.is_language_available)
        self.assertFalse(avail_bad.is_ready)

        if avail_bad.is_engine_available:
            self.assertIn("nonexistent_lang_xyz_999", avail_bad.status_message)


if __name__ == "__main__":
    unittest.main()
