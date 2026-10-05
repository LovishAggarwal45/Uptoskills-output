"""Document deskewing algorithms and geometric rotation."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.logging import get_logger
from src.preprocessing.operations import to_grayscale

logger = get_logger("preprocessing.deskew")


def estimate_skew_angle(
    image: np.ndarray,
    method: str = "min_area_rect",
) -> float:
    """Estimate the orientation skew angle (in degrees) of text lines in a document.

    Args:
        image: Grayscale or color document image.
        method: 'min_area_rect' (default) or 'hough_lines'.

    Returns:
        Estimated skew angle in degrees (-45.0 to 45.0).
    """
    if image is None or image.size == 0:
        return 0.0

    gray = to_grayscale(image)

    if method.lower() == "hough_lines":
        return _estimate_skew_hough(gray)
    else:
        return _estimate_skew_min_area_rect(gray)


def _estimate_skew_min_area_rect(gray: np.ndarray) -> float:
    """Estimate skew angle using the minimum area bounding rectangle of foreground pixels."""
    # Invert binary threshold to isolate foreground text components
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological closing with horizontal kernel to connect adjacent letters into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(morph)
    if coords is None or len(coords) < 50:
        return 0.0

    rect = cv2.minAreaRect(coords)
    (cx, cy), (w, h), angle = rect

    # OpenCV minAreaRect returns angle with respect to horizontal axis
    if w < h:
        angle = 90.0 + angle

    # Normalize to [-45, 45] range
    if angle < -45.0:
        angle += 90.0
    elif angle > 45.0:
        angle -= 90.0

    return round(float(-angle), 2)


def _estimate_skew_hough(gray: np.ndarray) -> float:
    """Estimate skew angle using probabilistic Hough line transform on detected edges."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=100,
        minLineLength=gray.shape[1] // 8,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        deg = math.degrees(math.atan2(dy, dx))
        # Filter to near-horizontal lines within +/- 45 degrees
        if abs(deg) <= 45.0:
            angles.append(deg)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    return round(median_angle, 2)


def rotate_image(
    image: np.ndarray,
    angle: float,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Rotate image by the given angle around its center with automatic canvas expansion.

    Args:
        image: Numpy image array.
        angle: Rotation angle in degrees (positive = counter-clockwise, negative = clockwise).
        background_color: Fill color for newly exposed corners.

    Returns:
        Rotated image array with expanded canvas to prevent edge clipping.
    """
    if abs(angle) < 1e-4:
        return image.copy()

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Compute rotation matrix (cv2.getRotationMatrix2D uses positive angle = counter-clockwise)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions of the rotated canvas
    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust transformation matrix to account for translation to new center
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    fill_val = 255 if len(image.shape) == 2 else background_color

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_val,
    )
    return rotated


def deskew_image(
    image: np.ndarray,
    min_angle_threshold: float = 0.5,
    max_angle_threshold: float = 45.0,
    method: str = "min_area_rect",
) -> Tuple[np.ndarray, float, bool]:
    """Detect and correct document page orientation skew.

    Args:
        image: Input document image.
        min_angle_threshold: Angles smaller than this (degrees) are considered negligible and not rotated.
        max_angle_threshold: Maximum plausible skew angle to correct.
        method: Estimation method ('min_area_rect' or 'hough_lines').

    Returns:
        Tuple of (deskewed_image, estimated_angle_degrees, deskew_applied_boolean).
    """
    angle = estimate_skew_angle(image, method=method)

    if abs(angle) < min_angle_threshold or abs(angle) > max_angle_threshold:
        logger.debug(f"Deskew skipped (angle={angle:.2f}° within negligible threshold {min_angle_threshold}°)")
        return image.copy(), angle, False

    logger.info(f"Applying deskew rotation of {-angle:.2f}° to correct detected {angle:.2f}° skew")
    rotated = rotate_image(image, -angle)
    return rotated, angle, True
