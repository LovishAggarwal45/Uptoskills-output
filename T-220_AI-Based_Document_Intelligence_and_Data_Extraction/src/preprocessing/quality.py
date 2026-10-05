"""Heuristic image quality diagnostics and blur/contrast/exposure assessment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.core.config import QualityConfig
from src.preprocessing.operations import to_grayscale


def compute_image_quality_heuristics(
    image: np.ndarray,
    config: Optional[QualityConfig] = None,
) -> Dict[str, Any]:
    """Compute lightweight heuristic image quality metrics and generate diagnostic warnings.

    Note: These are deterministic mathematical heuristics, NOT machine-learning probabilities.

    Args:
        image: Numpy image array.
        config: Optional QualityConfig containing diagnostic thresholds.

    Returns:
        Dictionary containing dimensions, brightness, contrast, sharpness, and warning flags.
    """
    if image is None or image.size == 0:
        return {
            "is_heuristic": True,
            "error": "Empty image array",
            "quality_warnings": ["Image array is empty or unreadable"],
        }

    cfg = config or QualityConfig()
    h, w = image.shape[:2]
    gray = to_grayscale(image)

    # 1. Grayscale statistical distributions
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    # 2. Sharpness / Blur estimation via Variance of Laplacian
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Foreground / Background content distribution
    # Calculate percentage of dark pixels (typical for printed text on white paper)
    dark_pixels = np.sum(gray < 128)
    total_pixels = h * w
    dark_pixel_ratio = float(dark_pixels / total_pixels) if total_pixels > 0 else 0.0

    # 4. Heuristic Diagnostic Warnings
    warnings: List[str] = []

    if cfg.enabled:
        # Sharpness / Blur check
        if laplacian_var < cfg.blur_threshold:
            warnings.append(
                f"Potential blur detected: Sharpness metric ({laplacian_var:.1f}) is below threshold ({cfg.blur_threshold:.1f})."
            )

        # Contrast check
        if contrast_std < cfg.min_contrast:
            warnings.append(
                f"Low image contrast: Contrast standard deviation ({contrast_std:.1f}) is below threshold ({cfg.min_contrast:.1f})."
            )

        # Exposure checks
        if mean_brightness < cfg.min_brightness:
            warnings.append(
                f"Image appears under-exposed (too dark): Mean brightness ({mean_brightness:.1f}) is below threshold ({cfg.min_brightness:.1f})."
            )
        elif mean_brightness > cfg.max_brightness:
            warnings.append(
                f"Image appears over-exposed (washed out): Mean brightness ({mean_brightness:.1f}) exceeds threshold ({cfg.max_brightness:.1f})."
            )

        # Resolution checks
        if w < cfg.min_width or h < cfg.min_height:
            warnings.append(
                f"Low resolution: Image dimensions ({w}x{h}) are smaller than recommended minimum ({cfg.min_width}x{cfg.min_height}) for reliable OCR."
            )

    return {
        "is_heuristic": True,
        "width": w,
        "height": h,
        "aspect_ratio": round(w / float(h), 3) if h > 0 else 0.0,
        "megapixels": round((w * h) / 1_000_000.0, 3),
        "mean_brightness": round(mean_brightness, 2),
        "contrast_std": round(contrast_std, 2),
        "sharpness_laplacian_var": round(laplacian_var, 2),
        "dark_pixel_ratio": round(dark_pixel_ratio, 4),
        "has_warnings": len(warnings) > 0,
        "quality_warnings": warnings,
    }
