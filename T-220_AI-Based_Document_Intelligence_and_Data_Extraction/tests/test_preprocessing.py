"""Unit tests for OpenCV preprocessing operations, deskewing, and image enhancements."""

import math
import unittest
import cv2
import numpy as np

from src.preprocessing.deskew import deskew_image, estimate_skew_angle, rotate_image
from src.preprocessing.operations import (
    clean_borders,
    denoise_image,
    enhance_contrast,
    resize_image,
    threshold_image,
    to_grayscale,
)


class TestPreprocessingOperations(unittest.TestCase):
    """Test suite covering individual image transformation operators."""

    def test_to_grayscale_conversions(self) -> None:
        # 1. Standard 3-channel BGR
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :] = (50, 100, 150)
        gray_bgr = to_grayscale(bgr)
        self.assertEqual(len(gray_bgr.shape), 2)
        self.assertEqual(gray_bgr.shape, (100, 100))

        # 2. 4-channel BGRA with alpha
        bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        bgra[:, :] = (0, 0, 255, 128)  # Semi-transparent red
        gray_bgra = to_grayscale(bgra)
        self.assertEqual(len(gray_bgra.shape), 2)

        # 3. Already grayscale
        gray = np.ones((50, 50), dtype=np.uint8) * 128
        gray_out = to_grayscale(gray)
        self.assertEqual(gray_out.shape, (50, 50))

    def test_resize_image_aspect_ratio(self) -> None:
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # Max dimension 1000 -> width 2000 scaled down to 1000, height 1000 scaled to 500
        resized, scale = resize_image(img, max_dimension=1000)
        self.assertEqual(resized.shape[0], 500)
        self.assertEqual(resized.shape[1], 1000)
        self.assertAlmostEqual(scale, 0.5)

        # Explicit target width
        resized_w, scale_w = resize_image(img, target_width=800)
        self.assertEqual(resized_w.shape[1], 800)
        self.assertEqual(resized_w.shape[0], 400)
        self.assertAlmostEqual(scale_w, 0.4)

    def test_denoise_image(self) -> None:
        noisy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        denoised = denoise_image(noisy, h=10.0)
        self.assertEqual(denoised.shape, (100, 100))

    def test_enhance_contrast_clahe(self) -> None:
        low_contrast = np.ones((100, 100), dtype=np.uint8) * 120
        # Add slight variation
        low_contrast[20:80, 20:80] = 130
        enhanced = enhance_contrast(low_contrast, clip_limit=2.0)
        self.assertEqual(enhanced.shape, (100, 100))
        # Standard deviation / contrast should increase
        self.assertGreaterEqual(np.std(enhanced), np.std(low_contrast))

    def test_threshold_image_otsu_and_adaptive(self) -> None:
        gradient = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))

        # Otsu binarization
        otsu = threshold_image(gradient, method="otsu")
        unique_vals = set(np.unique(otsu))
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Adaptive Gaussian
        adaptive = threshold_image(gradient, method="adaptive_gaussian")
        unique_adaptive = set(np.unique(adaptive))
        self.assertTrue(unique_adaptive.issubset({0, 255}))

    def test_clean_borders(self) -> None:
        img = np.zeros((100, 100), dtype=np.uint8)  # Black image
        cleaned = clean_borders(img, border_fraction=0.05)
        # Margins (5px on each side) should now be 255 (white)
        self.assertTrue(np.all(cleaned[:5, :] == 255))
        self.assertTrue(np.all(cleaned[95:, :] == 255))
        self.assertTrue(np.all(cleaned[:, :5] == 255))
        self.assertTrue(np.all(cleaned[:, 95:] == 255))
        # Center should remain black
        self.assertTrue(np.all(cleaned[10:90, 10:90] == 0))


class TestDeskew(unittest.TestCase):
    """Test suite covering orientation skew detection and rotation."""

    def _generate_skewed_text_image(self, angle_deg: float) -> np.ndarray:
        """Create a synthetic high-contrast text image rotated by an exact angle."""
        canvas = np.ones((600, 800), dtype=np.uint8) * 255

        # Draw multiple parallel horizontal text lines
        for y in range(150, 450, 40):
            cv2.putText(
                canvas,
                "INVOICE LINE ITEM DESCRIPTION FOR OCR EVALUATION",
                (50, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                0,
                2,
                cv2.LINE_AA,
            )

        # Rotate canvas by angle_deg
        return rotate_image(canvas, angle_deg)

    def test_rotate_image_expands_canvas(self) -> None:
        img = np.ones((400, 400), dtype=np.uint8) * 255
        rotated = rotate_image(img, 45.0)
        # At 45 degrees, bounding canvas expands to sqrt(2) * 400 ~ 565
        self.assertGreater(rotated.shape[0], 400)
        self.assertGreater(rotated.shape[1], 400)

    def test_deskew_correction_on_skewed_document(self) -> None:
        target_skew = -5.0
        skewed = self._generate_skewed_text_image(target_skew)

        deskewed, detected_angle, was_deskewed = deskew_image(
            skewed,
            min_angle_threshold=0.5,
            max_angle_threshold=45.0,
        )

        self.assertTrue(was_deskewed)
        # Detected angle should be close to the applied target skew (-5.0 +/- 1.5)
        self.assertAlmostEqual(detected_angle, target_skew, delta=2.0)

    def test_negligible_angle_is_not_rotated(self) -> None:
        unskewed = np.ones((400, 400), dtype=np.uint8) * 255
        deskewed, angle, was_deskewed = deskew_image(unskewed, min_angle_threshold=0.5)

        self.assertFalse(was_deskewed)
        self.assertEqual(deskewed.shape, unskewed.shape)


if __name__ == "__main__":
    unittest.main()
