"""Unit tests for VisualizationConfig schema and dictionary deserialization."""

import unittest

from src.core.config import DocuMindConfig, VisualizationConfig


class TestVisualizationConfig(unittest.TestCase):
    """Test suite for visualization configuration schema."""

    def test_default_config_instantiation(self) -> None:
        """Verify default visualization configuration parameters."""
        config = VisualizationConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.output.directory, "outputs/visualizations")
        self.assertEqual(config.output.format, "png")
        self.assertTrue(config.output.save_manifest)
        self.assertTrue(config.layers.ocr)
        self.assertTrue(config.layers.fields)
        self.assertTrue(config.layers.tables)
        self.assertTrue(config.layers.validation)
        self.assertTrue(config.layers.review)
        self.assertEqual(config.rendering.line_width, 2)
        self.assertAlmostEqual(config.rendering.opacity, 0.85)

    def test_custom_dict_parsing(self) -> None:
        """Verify parsing custom nested dictionary into VisualizationConfig."""
        data = {
            "enabled": False,
            "output": {
                "directory": "custom_outputs/vis",
                "format": "jpg",
                "save_manifest": False,
            },
            "layers": {
                "ocr": False,
                "tables": False,
            },
            "rendering": {
                "line_width": 4,
                "opacity": 0.5,
            },
        }

        config = VisualizationConfig.from_dict(data)
        self.assertFalse(config.enabled)
        self.assertEqual(config.output.directory, "custom_outputs/vis")
        self.assertEqual(config.output.format, "jpg")
        self.assertFalse(config.output.save_manifest)
        self.assertFalse(config.layers.ocr)
        self.assertFalse(config.layers.tables)
        self.assertTrue(config.layers.fields)  # Default
        self.assertEqual(config.rendering.line_width, 4)
        self.assertAlmostEqual(config.rendering.opacity, 0.5)

    def test_documind_config_has_visualization(self) -> None:
        """Verify master DocuMindConfig includes visualization block."""
        master = DocuMindConfig()
        self.assertIsNotNone(master.visualization)
        self.assertTrue(master.visualization.enabled)


if __name__ == "__main__":
    unittest.main()
