"""Unit tests for image quality heuristics and diagnostic warnings."""

import unittest
import cv2
import numpy as np

from src.core.config import QualityConfig
from src.preprocessing.quality import compute_image_quality_heuristics


class TestQualityMetrics(unittest.TestCase):
    """Test suite covering heuristic image quality diagnostics."""

    def test_dimensions_and_metadata_metrics(self) -> None:
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 200
        metrics = compute_image_quality_heuristics(img)

        self.assertTrue(metrics["is_heuristic"])
        self.assertEqual(metrics["width"], 1200)
        self.assertEqual(metrics["height"], 800)
        self.assertEqual(metrics["aspect_ratio"], 1.5)
        self.assertAlmostEqual(metrics["megapixels"], 0.96)
        self.assertAlmostEqual(metrics["mean_brightness"], 200.0)

    def test_blur_detection_with_laplacian_variance(self) -> None:
        # 1. Create a sharp text image
        sharp_img = np.ones((400, 400), dtype=np.uint8) * 255
        cv2.putText(sharp_img, "SHARP CRISP TEXT", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

        # 2. Artificially blur the image with large Gaussian kernel
        blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)

        config = QualityConfig(blur_threshold=100.0)

        sharp_metrics = compute_image_quality_heuristics(sharp_img, config)
        blurry_metrics = compute_image_quality_heuristics(blurry_img, config)

        # Sharpness score of sharp image must be significantly higher than blurred image
        self.assertGreater(sharp_metrics["sharpness_laplacian_var"], blurry_metrics["sharpness_laplacian_var"])
        self.assertTrue(blurry_metrics["has_warnings"])
        self.assertTrue(any("blur" in w.lower() for w in blurry_metrics["quality_warnings"]))

    def test_low_contrast_warning(self) -> None:
        # Uniform gray image with negligible contrast
        flat_img = np.ones((400, 400), dtype=np.uint8) * 128
        config = QualityConfig(min_contrast=30.0)

        metrics = compute_image_quality_heuristics(flat_img, config)
        self.assertTrue(metrics["has_warnings"])
        self.assertTrue(any("contrast" in w.lower() for w in metrics["quality_warnings"]))

    def test_exposure_warnings(self) -> None:
        config = QualityConfig(min_brightness=40.0, max_brightness=235.0)

        # Under-exposed image (near black)
        dark_img = np.ones((400, 400), dtype=np.uint8) * 15
        dark_metrics = compute_image_quality_heuristics(dark_img, config)
        self.assertTrue(any("under-exposed" in w.lower() for w in dark_metrics["quality_warnings"]))

        # Over-exposed image (blown out white)
        washed_img = np.ones((400, 400), dtype=np.uint8) * 250
        washed_metrics = compute_image_quality_heuristics(washed_img, config)
        self.assertTrue(any("over-exposed" in w.lower() for w in washed_metrics["quality_warnings"]))

    def test_low_resolution_warning(self) -> None:
        tiny_img = np.ones((200, 300), dtype=np.uint8) * 255
        config = QualityConfig(min_width=600, min_height=600)

        metrics = compute_image_quality_heuristics(tiny_img, config)
        self.assertTrue(any("resolution" in w.lower() for w in metrics["quality_warnings"]))


if __name__ == "__main__":
    unittest.main()
