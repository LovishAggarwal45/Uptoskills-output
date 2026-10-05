"""Unit tests for configuration management and YAML parsing."""

import tempfile
import unittest
from pathlib import Path

from src.core.config import (
    DocuMindConfig,
    IngestionConfig,
    OCRConfig,
    ValidationConfig,
    load_config,
)


class TestConfig(unittest.TestCase):
    """Test suite for DocuMind AI configuration schema and loader."""

    def test_default_config_instantiation(self) -> None:
        config = DocuMindConfig()
        self.assertEqual(config.project_name, "DocuMind AI")
        self.assertEqual(config.version, "0.1.0")
        self.assertIn(".pdf", config.ingestion.supported_extensions)
        self.assertEqual(config.ocr.engine_type, "tesseract")
        self.assertEqual(config.validation.subtotal_tax_tolerance, 0.05)
        self.assertEqual(config.confidence.review_confidence_threshold, 0.70)

    def test_load_default_yaml_file(self) -> None:
        yaml_path = Path("configs/default.yaml")
        if not yaml_path.exists():
            self.skipTest("configs/default.yaml not present in relative path.")

        config = load_config(yaml_path)
        self.assertEqual(config.project_name, "DocuMind AI")
        self.assertEqual(config.ingestion.render_dpi, 300)
        self.assertIn("invoice_number", config.extraction.invoice_required_fields)

    def test_custom_yaml_loading(self) -> None:
        custom_yaml = """
project_name: "DocuMind Custom"
version: "0.2.0"
environment: "production"
ocr:
  engine_type: "easyocr"
  languages:
    - "eng"
    - "spa"
  min_confidence_threshold: 0.50
validation:
  strict_mode: true
  subtotal_tax_tolerance: 0.01
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(custom_yaml)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            self.assertEqual(config.project_name, "DocuMind Custom")
            self.assertEqual(config.environment, "production")
            self.assertEqual(config.ocr.engine_type, "easyocr")
            self.assertEqual(config.ocr.languages, ["eng", "spa"])
            self.assertTrue(config.validation.strict_mode)
            self.assertEqual(config.validation.subtotal_tax_tolerance, 0.01)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_to_dict_and_to_yaml_roundtrip(self) -> None:
        config = DocuMindConfig()
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["ocr"]["engine_type"], "tesseract")

        yaml_str = config.to_yaml()
        self.assertIn("project_name: DocuMind AI", yaml_str)


if __name__ == "__main__":
    unittest.main()
