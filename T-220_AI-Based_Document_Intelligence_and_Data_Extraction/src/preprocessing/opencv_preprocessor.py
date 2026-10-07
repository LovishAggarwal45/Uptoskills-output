"""OpenCV-based image preprocessor implementing the modular transformation pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from src.core.config import PreprocessingConfig, QualityConfig
from src.core.exceptions import PreprocessingError
from src.core.logging import get_logger
from src.core.models import DocumentPage
from src.preprocessing.base import BaseImagePreprocessor, PreprocessingResult
from src.preprocessing.deskew import deskew_image
from src.preprocessing.operations import (
    clean_borders,
    denoise_image,
    enhance_contrast,
    resize_image,
    threshold_image,
    to_grayscale,
)
from src.preprocessing.quality import compute_image_quality_heuristics

logger = get_logger("preprocessing.opencv")


class OpenCVImagePreprocessor(BaseImagePreprocessor):
    """Production-grade document image preprocessor leveraging OpenCV and NumPy."""

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        quality_config: Optional[QualityConfig] = None,
    ) -> None:
        super().__init__(config=config)
        self.quality_config = quality_config or QualityConfig()

    def preprocess_page(
        self,
        page: DocumentPage,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PreprocessingResult:
        """Apply configured image operations to a single document page without altering the original.

        Args:
            page: DocumentPage instance with valid image_path.
            output_dir: Destination directory for the processed image output.

        Returns:
            PreprocessingResult containing the processed image location, metrics, and applied steps.
        """
        if not page.image_path:
            raise PreprocessingError(
                f"DocumentPage {page.page_number} has no associated image_path."
            )

        source_path = Path(page.image_path)
        if not source_path.exists():
            raise PreprocessingError(
                f"Page image file not found at path: {source_path}"
            )

        start_time = time.perf_counter()

        # Safely load image using OpenCV
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            # Fallback to Pillow if OpenCV cannot decode exotic format
            try:
                with Image.open(source_path) as pil_img:
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise PreprocessingError(
                    f"Failed to load page image '{source_path.name}': {e}"
                ) from e

        orig_h, orig_w = image.shape[:2]
        applied_operations: List[str] = []
        deskew_angle: Optional[float] = None
        scale_factor: float = 1.0

        # Step 0: Calculate initial quality diagnostics
        initial_quality = compute_image_quality_heuristics(image, self.quality_config)

        if not self.config.enabled:
            logger.info(f"Preprocessing disabled by configuration for page {page.page_number}")
            elapsed = time.perf_counter() - start_time
            return PreprocessingResult(
                page_number=page.page_number,
                processed_image_path=source_path,
                applied_operations=[],
                metrics={
                    "initial_quality": initial_quality,
                    "final_quality": initial_quality,
                    "duration_seconds": round(elapsed, 4),
                },
                deskew_angle=None,
                is_modified=False,
            )

        working_image = image.copy()

        # Step 1: Deskew (applied early before scaling or binarization)
        if self.config.apply_deskew:
            working_image, angle, was_deskewed = deskew_image(
                working_image,
                min_angle_threshold=self.config.deskew_min_angle,
                max_angle_threshold=self.config.deskew_max_angle,
            )
            deskew_angle = angle
            if was_deskewed:
                applied_operations.append(f"deskew({angle:.2f}°)")

        # Step 2: Grayscale conversion
        if self.config.apply_grayscale and len(working_image.shape) > 2:
            working_image = to_grayscale(working_image)
            applied_operations.append("grayscale")

        # Step 3: Resize / dimension normalization
        if (
            self.config.resize_max_dimension
            or self.config.target_width
        ):
            working_image, scale = resize_image(
                working_image,
                max_dimension=self.config.resize_max_dimension,
                target_width=self.config.target_width,
            )
            scale_factor = scale
            if abs(scale - 1.0) > 1e-4:
                applied_operations.append(f"resize(scale={scale:.3f})")

        # Step 4: Denoising
        if self.config.apply_denoising:
            working_image = denoise_image(working_image, h=self.config.denoise_h)
            applied_operations.append(f"denoise(h={self.config.denoise_h})")

        # Step 5: Contrast Enhancement (CLAHE)
        if self.config.apply_contrast_enhancement:
            working_image = enhance_contrast(
                working_image,
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=(self.config.clahe_tile_grid_size, self.config.clahe_tile_grid_size),
            )
            applied_operations.append("clahe_contrast")

        # Step 6: Binarization / Thresholding (Optional, default False)
        if self.config.apply_binarization:
            working_image = threshold_image(
                working_image,
                method=self.config.binarization_method,
            )
            applied_operations.append(f"threshold({self.config.binarization_method})")

        # Step 7: Border cleanup (Optional)
        if self.config.apply_border_cleanup:
            working_image = clean_borders(
                working_image,
                border_fraction=self.config.border_cleanup_fraction,
            )
            applied_operations.append("border_cleanup")

        # Determine target output path
        dest_dir = Path(output_dir) if output_dir else source_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        processed_path = dest_dir / f"page_{page.page_number:03d}_processed.png"

        # Save processed image
        cv2.imwrite(str(processed_path), working_image)

        # Calculate final quality metrics
        final_quality = compute_image_quality_heuristics(working_image, self.quality_config)
        elapsed = time.perf_counter() - start_time

        final_h, final_w = working_image.shape[:2]
        is_modified = len(applied_operations) > 0

        logger.info(
            f"Preprocessed page {page.page_number}: ({orig_w}x{orig_h} -> {final_w}x{final_h}) "
            f"applied={applied_operations} in {elapsed:.3f}s"
        )

        return PreprocessingResult(
            page_number=page.page_number,
            processed_image_path=processed_path,
            applied_operations=applied_operations,
            metrics={
                "original_dimensions": {"width": orig_w, "height": orig_h},
                "processed_dimensions": {"width": final_w, "height": final_h},
                "scale_factor": round(scale_factor, 4),
                "duration_seconds": round(elapsed, 4),
                "initial_quality": initial_quality,
                "final_quality": final_quality,
            },
            deskew_angle=deskew_angle,
            is_modified=is_modified,
        )
