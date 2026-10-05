"""Computer vision operations for document page normalization and enhancement."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger

logger = get_logger("preprocessing.operations")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image array (BGR, BGRA, or already grayscale) to 8-bit single channel grayscale.

    Args:
        image: Numpy image array.

    Returns:
        Grayscale numpy array with shape (H, W).
    """
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot convert empty image array to grayscale.")

    if len(image.shape) == 2:
        return image.copy()

    channels = image.shape[2]
    if channels == 4:
        # Handle alpha channel: composite against white background before grayscale
        b, g, r, a = cv2.split(image)
        alpha_norm = a.astype(float) / 255.0
        white_bg = np.ones_like(b, dtype=float) * 255.0
        b_comp = (b.astype(float) * alpha_norm + white_bg * (1.0 - alpha_norm)).astype(np.uint8)
        g_comp = (g.astype(float) * alpha_norm + white_bg * (1.0 - alpha_norm)).astype(np.uint8)
        r_comp = (r.astype(float) * alpha_norm + white_bg * (1.0 - alpha_norm)).astype(np.uint8)
        bgr = cv2.merge([b_comp, g_comp, r_comp])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise PreprocessingError(f"Unsupported number of image channels for grayscale: {channels}")


def resize_image(
    image: np.ndarray,
    max_dimension: Optional[int] = None,
    target_width: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Resize image while strictly preserving aspect ratio.

    Args:
        image: Input numpy image array.
        max_dimension: Maximum allowed width or height in pixels.
        target_width: Explicit target width in pixels (takes precedence over max_dimension).

    Returns:
        Tuple of (resized_image, scale_factor). Scale factor is 1.0 if no resizing occurred.
    """
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot resize empty image array.")

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise PreprocessingError(f"Invalid image dimensions for resize: {w}x{h}")

    scale = 1.0

    if target_width is not None and target_width > 0:
        scale = target_width / float(w)
    elif max_dimension is not None and max_dimension > 0:
        max_dim = max(h, w)
        if max_dim > max_dimension:
            scale = max_dimension / float(max_dim)

    if abs(scale - 1.0) < 1e-4:
        return image.copy(), 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Use INTER_AREA for downscaling (sharpness retention), INTER_CUBIC for upscaling
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    logger.debug(f"Resized image from ({w}x{h}) to ({new_w}x{new_h}) with scale={scale:.4f}")
    return resized, scale


def denoise_image(image: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Apply non-local means or bilateral filtering to eliminate speckle noise while preserving text edges.

    Args:
        image: Input image (grayscale or color).
        h: Filter strength parameter.

    Returns:
        Denoised image.
    """
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot denoise empty image array.")

    try:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(
                src=image,
                h=float(h),
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            return cv2.fastNlMeansDenoisingColored(
                src=image,
                h=float(h),
                hColor=float(h),
                templateWindowSize=7,
                searchWindowSize=21,
            )
    except Exception as e:
        logger.warning(f"FastNLMeans denoising failed ({e}), falling back to Bilateral Filter.")
        return cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance document contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Grayscale or BGR image.
        clip_limit: Threshold for contrast limiting (default 2.0).
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Contrast-enhanced image array.
    """
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot enhance contrast on empty image.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        return clahe.apply(image)
    else:
        # Convert to LAB color space and apply CLAHE strictly on L (lightness) channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def threshold_image(
    image: np.ndarray,
    method: str = "otsu",
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:
    """Binarize document image using Otsu or Adaptive Thresholding.

    Args:
        image: Grayscale image array.
        method: 'otsu', 'adaptive_gaussian', or 'adaptive_mean'.
        block_size: Size of a pixel neighborhood used to calculate a threshold value (must be odd).
        c: Constant subtracted from the mean or weighted mean.

    Returns:
        Binarized (0 or 255) 8-bit image array.
    """
    gray = to_grayscale(image)

    if method.lower() == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    elif method.lower() == "adaptive_gaussian":
        # Ensure block_size is odd and > 1
        bs = block_size if block_size % 2 == 1 and block_size > 1 else 11
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, c
        )
    elif method.lower() == "adaptive_mean":
        bs = block_size if block_size % 2 == 1 and block_size > 1 else 11
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, c
        )
    else:
        raise ValueError(f"Unknown thresholding method: '{method}'. Supported: otsu, adaptive_gaussian, adaptive_mean")


def clean_borders(image: np.ndarray, border_fraction: float = 0.01) -> np.ndarray:
    """Eliminate scanner margin artifacts and dark page edges by filling the outermost perimeter with white.

    Args:
        image: Grayscale or BGR image array.
        border_fraction: Margin percentage (0.01 = 1% from each edge).

    Returns:
        Border-cleaned image array.
    """
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot clean borders on empty image.")

    h, w = image.shape[:2]
    margin_y = max(1, int(round(h * border_fraction)))
    margin_x = max(1, int(round(w * border_fraction)))

    cleaned = image.copy()
    fill_val = 255 if len(image.shape) == 2 else (255, 255, 255)

    # Top and bottom margins
    cleaned[:margin_y, :] = fill_val
    cleaned[h - margin_y :, :] = fill_val

    # Left and right margins
    cleaned[:, :margin_x] = fill_val
    cleaned[:, w - margin_x :] = fill_val

    return cleaned
