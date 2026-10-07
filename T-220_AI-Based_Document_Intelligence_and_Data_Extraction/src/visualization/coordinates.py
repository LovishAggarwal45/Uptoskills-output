"""Geometric coordinate transformation, normalization, and boundary clipping utilities."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.visualization.exceptions import (
    CoordinateMappingError,
    InvalidBoundingBoxError,
)
from src.visualization.models import VisualCoordinateSystem

logger = get_logger("visualization.coordinates")


class CoordinateTransformer:
    """Handles 2D geometric coordinate transformations, scaling, and clipping.

    Standard Document Coordinate System:
    - Origin (0, 0) is the top-left corner of the page image.
    - X-axis increases horizontally to the right: [0, image_width].
    - Y-axis increases vertically downward: [0, image_height].
    """

    @staticmethod
    def validate_dimensions(image_width: float, image_height: float) -> None:
        """Assert that target image dimensions are positive and finite."""
        if not (math.isfinite(image_width) and image_width > 0):
            raise CoordinateMappingError(f"Invalid image width: {image_width}. Must be positive and finite.")
        if not (math.isfinite(image_height) and image_height > 0):
            raise CoordinateMappingError(f"Invalid image height: {image_height}. Must be positive and finite.")

    @classmethod
    def validate_bounding_box(cls, bbox: BoundingBox) -> None:
        """Validate that all coordinate values are finite numbers."""
        for val, name in [
            (bbox.xmin, "xmin"),
            (bbox.ymin, "ymin"),
            (bbox.xmax, "xmax"),
            (bbox.ymax, "ymax"),
        ]:
            if not math.isfinite(val):
                raise InvalidBoundingBoxError(f"BoundingBox coordinate '{name}' is not a finite number: {val}")

    @classmethod
    def to_pixel_box(
        cls,
        bbox: BoundingBox,
        image_width: float,
        image_height: float,
        clip_to_page: bool = True,
    ) -> BoundingBox:
        """Convert a bounding box (normalized or absolute) to absolute pixel coordinates for an image.

        Args:
            bbox: Source BoundingBox instance.
            image_width: Target image width in pixels.
            image_height: Target image height in pixels.
            clip_to_page: If True, clamp coordinates strictly to [0, width] and [0, height].

        Returns:
            New BoundingBox in absolute pixel coordinates (is_normalized=False).
        """
        cls.validate_dimensions(image_width, image_height)
        cls.validate_bounding_box(bbox)

        # Fix reversed bounds if necessary
        xmin = min(bbox.xmin, bbox.xmax)
        xmax = max(bbox.xmin, bbox.xmax)
        ymin = min(bbox.ymin, bbox.ymax)
        ymax = max(bbox.ymin, bbox.ymax)

        if bbox.is_normalized:
            xmin = xmin * image_width
            xmax = xmax * image_width
            ymin = ymin * image_height
            ymax = ymax * image_height

        if clip_to_page:
            xmin = max(0.0, min(image_width, xmin))
            xmax = max(0.0, min(image_width, xmax))
            ymin = max(0.0, min(image_height, ymin))
            ymax = max(0.0, min(image_height, ymax))

        return BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_normalized=False,
        )

    @classmethod
    def to_normalized_box(
        cls,
        bbox: BoundingBox,
        image_width: float,
        image_height: float,
        clip_to_unit: bool = True,
    ) -> BoundingBox:
        """Convert an absolute pixel bounding box to normalized unit coordinates [0.0, 1.0].

        Args:
            bbox: Source BoundingBox instance.
            image_width: Source image width in pixels.
            image_height: Source image height in pixels.
            clip_to_unit: If True, clamp coordinates to [0.0, 1.0].

        Returns:
            New BoundingBox in normalized unit coordinates (is_normalized=True).
        """
        cls.validate_dimensions(image_width, image_height)
        cls.validate_bounding_box(bbox)

        if bbox.is_normalized:
            xmin = bbox.xmin
            ymin = bbox.ymin
            xmax = bbox.xmax
            ymax = bbox.ymax
        else:
            xmin = bbox.xmin / image_width
            ymin = bbox.ymin / image_height
            xmax = bbox.xmax / image_width
            ymax = bbox.ymax / image_height

        if clip_to_unit:
            xmin = max(0.0, min(1.0, xmin))
            xmax = max(0.0, min(1.0, xmax))
            ymin = max(0.0, min(1.0, ymin))
            ymax = max(0.0, min(1.0, ymax))

        return BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_normalized=True,
        )

    @classmethod
    def scale_pixel_box(
        cls,
        bbox: BoundingBox,
        scale_x: float,
        scale_y: float,
        clip_bounds: Optional[Tuple[float, float]] = None,
    ) -> BoundingBox:
        """Scale absolute pixel coordinates by horizontal and vertical scaling factors.

        Args:
            bbox: Source BoundingBox.
            scale_x: Horizontal scale factor.
            scale_y: Vertical scale factor.
            clip_bounds: Optional (max_width, max_height) for clipping.

        Returns:
            Scaled BoundingBox instance.
        """
        if bbox.is_normalized:
            raise CoordinateMappingError("Cannot apply pixel scale factor to a normalized bounding box.")
        if scale_x <= 0 or scale_y <= 0:
            raise CoordinateMappingError(f"Scale factors must be positive: scale_x={scale_x}, scale_y={scale_y}")

        cls.validate_bounding_box(bbox)

        xmin = bbox.xmin * scale_x
        ymin = bbox.ymin * scale_y
        xmax = bbox.xmax * scale_x
        ymax = bbox.ymax * scale_y

        if clip_bounds:
            max_w, max_h = clip_bounds
            xmin = max(0.0, min(max_w, xmin))
            xmax = max(0.0, min(max_w, xmax))
            ymin = max(0.0, min(max_h, ymin))
            ymax = max(0.0, min(max_h, ymax))

        return BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_normalized=False,
        )


__all__ = ["CoordinateTransformer"]
