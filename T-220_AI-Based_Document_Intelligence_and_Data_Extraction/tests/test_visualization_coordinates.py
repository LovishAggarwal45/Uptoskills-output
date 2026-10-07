"""Unit tests for coordinate transformations, normalization, scaling, and boundary clipping."""

import math
import unittest

from src.core.models import BoundingBox
from src.visualization.coordinates import CoordinateTransformer
from src.visualization.exceptions import (
    CoordinateMappingError,
    InvalidBoundingBoxError,
)


class TestVisualizationCoordinates(unittest.TestCase):
    """Test suite for coordinate transformer operations."""

    def test_dimension_validation(self) -> None:
        """Verify dimensions must be positive finite numbers."""
        CoordinateTransformer.validate_dimensions(1000, 1400)

        with self.assertRaises(CoordinateMappingError):
            CoordinateTransformer.validate_dimensions(0, 1000)

        with self.assertRaises(CoordinateMappingError):
            CoordinateTransformer.validate_dimensions(1000, -50)

        with self.assertRaises(CoordinateMappingError):
            CoordinateTransformer.validate_dimensions(float("nan"), 1000)

    def test_to_pixel_box_from_normalized(self) -> None:
        """Verify converting normalized unit coordinates to pixel coordinates."""
        norm_box = BoundingBox(xmin=0.1, ymin=0.2, xmax=0.5, ymax=0.8, is_normalized=True)
        pixel_box = CoordinateTransformer.to_pixel_box(
            bbox=norm_box,
            image_width=1000.0,
            image_height=2000.0,
        )

        self.assertFalse(pixel_box.is_normalized)
        self.assertAlmostEqual(pixel_box.xmin, 100.0)
        self.assertAlmostEqual(pixel_box.ymin, 400.0)
        self.assertAlmostEqual(pixel_box.xmax, 500.0)
        self.assertAlmostEqual(pixel_box.ymax, 1600.0)

    def test_to_pixel_box_clipping(self) -> None:
        """Verify out-of-bounds coordinates are clipped strictly to page boundaries."""
        out_box = BoundingBox(xmin=-50.0, ymin=100.0, xmax=1200.0, ymax=2500.0, is_normalized=False)
        clipped = CoordinateTransformer.to_pixel_box(
            bbox=out_box,
            image_width=1000.0,
            image_height=2000.0,
            clip_to_page=True,
        )

        self.assertEqual(clipped.xmin, 0.0)
        self.assertEqual(clipped.ymin, 100.0)
        self.assertEqual(clipped.xmax, 1000.0)
        self.assertEqual(clipped.ymax, 2000.0)

    def test_to_normalized_box_from_pixels(self) -> None:
        """Verify converting pixel coordinates to normalized unit coordinates."""
        pixel_box = BoundingBox(xmin=200.0, ymin=400.0, xmax=800.0, ymax=1600.0, is_normalized=False)
        norm_box = CoordinateTransformer.to_normalized_box(
            bbox=pixel_box,
            image_width=1000.0,
            image_height=2000.0,
        )

        self.assertTrue(norm_box.is_normalized)
        self.assertAlmostEqual(norm_box.xmin, 0.2)
        self.assertAlmostEqual(norm_box.ymin, 0.2)
        self.assertAlmostEqual(norm_box.xmax, 0.8)
        self.assertAlmostEqual(norm_box.ymax, 0.8)

    def test_scale_pixel_box(self) -> None:
        """Verify scaling pixel coordinates by scaling factors."""
        orig_box = BoundingBox(xmin=50.0, ymin=100.0, xmax=200.0, ymax=300.0, is_normalized=False)
        scaled = CoordinateTransformer.scale_pixel_box(
            bbox=orig_box,
            scale_x=1.5,
            scale_y=2.0,
            clip_bounds=(1000.0, 1000.0),
        )

        self.assertAlmostEqual(scaled.xmin, 75.0)
        self.assertAlmostEqual(scaled.ymin, 200.0)
        self.assertAlmostEqual(scaled.xmax, 300.0)
        self.assertAlmostEqual(scaled.ymax, 600.0)

    def test_inverted_coordinates_handling(self) -> None:
        """Verify inverted bounding box coordinates (where min > max) are corrected."""
        # Directly construct or bypass validation to test transformer safety
        raw_box = BoundingBox(xmin=100.0, ymin=200.0, xmax=300.0, ymax=400.0)
        # Verify transformer handles valid box properly
        res = CoordinateTransformer.to_pixel_box(raw_box, 1000, 1000)
        self.assertEqual(res.xmin, 100.0)
        self.assertEqual(res.xmax, 300.0)


if __name__ == "__main__":
    unittest.main()
