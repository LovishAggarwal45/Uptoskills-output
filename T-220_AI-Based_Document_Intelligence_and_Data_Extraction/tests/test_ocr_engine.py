"""Unit tests for OCR Engine Factory, Tesseract data parsing, and exception handling."""

import unittest
from pathlib import Path
from PIL import Image

from src.core.config import OCRConfig
from src.ocr.base import BaseOCREngine
from src.ocr.engine_factory import OCREngineFactory
from src.ocr.exceptions import (
    OCRConfigurationError,
    OCREngineUnavailableError,
    OCRInvalidImageError,
)
from src.ocr.models import PageOCRResult
from src.ocr.tesseract_engine import TesseractOCREngine


class DummyCustomEngine(BaseOCREngine):
    """Dummy engine to test factory extensibility."""

    @property
    def engine_name(self) -> str:
        return "dummy_custom"

    def process_image(self, image_input, page_number: int = 1) -> PageOCRResult:
        return PageOCRResult(page_number=page_number, raw_text="Dummy", engine_name="dummy_custom")


class TestOCREngineFactory(unittest.TestCase):
    """Test suite covering OCR factory resolution and custom backend registration."""

    def test_create_default_tesseract_engine(self) -> None:
        engine = OCREngineFactory.create_engine("tesseract")
        self.assertIsInstance(engine, TesseractOCREngine)
        self.assertEqual(engine.engine_name, "tesseract")

    def test_unsupported_engine_raises_configuration_error(self) -> None:
        with self.assertRaises(OCRConfigurationError):
            OCREngineFactory.create_engine("nonexistent_ocr_engine")

    def test_register_and_create_custom_engine(self) -> None:
        OCREngineFactory.register_engine("dummy_custom", DummyCustomEngine)
        engine = OCREngineFactory.create_engine("dummy_custom")
        self.assertIsInstance(engine, DummyCustomEngine)
        self.assertEqual(engine.engine_name, "dummy_custom")


class TestTesseractEngineParsing(unittest.TestCase):
    """Test suite covering Tesseract data dictionary parsing and grouping logic."""

    def setUp(self) -> None:
        self.engine = TesseractOCREngine()

    def test_parse_tesseract_data_dictionary(self) -> None:
        # Synthetic pytesseract.image_to_data Output.DICT payload
        mock_data = {
            "level": [5, 5, 5, 5],
            "page_num": [1, 1, 1, 1],
            "block_num": [1, 1, 2, 2],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
            "word_num": [1, 2, 1, 2],
            "left": [100, 220, 100, 200],
            "top": [50, 50, 120, 120],
            "width": [100, 80, 90, 60],
            "height": [30, 30, 25, 25],
            "conf": [95.5, 92.0, 89.0, 94.2],
            "text": ["INVOICE", "TOTAL", "Date:", "2026-10-01"],
        }

        words = self.engine._parse_tesseract_data(mock_data, page_number=1)
        self.assertEqual(len(words), 4)
        self.assertEqual(words[0].text, "INVOICE")
        self.assertEqual(words[0].confidence, 0.955)
        self.assertEqual(words[0].bounding_box.xmin, 100.0)
        self.assertEqual(words[0].bounding_box.xmax, 200.0)
        self.assertEqual(words[1].text, "TOTAL")
        self.assertEqual(words[3].text, "2026-10-01")

        lines, blocks = self.engine._group_words_into_lines_and_blocks(words, page_number=1)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, "INVOICE TOTAL")
        self.assertEqual(lines[1].text, "Date: 2026-10-01")
        self.assertEqual(len(blocks), 2)

    def test_invalid_image_raises_error(self) -> None:
        with self.assertRaises(OCRInvalidImageError):
            self.engine._prepare_pil_image("/nonexistent/image.png")

        with self.assertRaises(OCRInvalidImageError):
            self.engine._prepare_pil_image(image_input=12345)  # Invalid type


if __name__ == "__main__":
    unittest.main()
