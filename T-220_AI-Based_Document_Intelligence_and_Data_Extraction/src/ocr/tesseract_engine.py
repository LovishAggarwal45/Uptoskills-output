"""Concrete Tesseract OCR engine implementation using pytesseract."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src.core.config import OCRConfig
from src.core.logging import get_logger
from src.core.models import BoundingBox, OCRTextRegion
from src.ocr.availability import check_ocr_availability, find_tesseract_binary
from src.ocr.base import BaseOCREngine
from src.ocr.exceptions import (
    OCREngineUnavailableError,
    OCRInvalidImageError,
    OCRProcessingError,
)
from src.ocr.models import OCRBlock, OCRLevel, OCRLine, OCRWord, PageOCRResult
from src.ocr.postprocessing import (
    compute_bounding_box_union,
    normalize_tesseract_confidence,
    reconstruct_page_text,
    sort_reading_order,
)

logger = get_logger("ocr.tesseract")


class TesseractOCREngine(BaseOCREngine):
    """Production Tesseract OCR backend providing word, line, and block structural recognition."""

    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        super().__init__(config=config)
        self._tesseract_cmd = find_tesseract_binary(self.config.tesseract_cmd)
        if self._tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
            except ImportError:
                pass

    @property
    def engine_name(self) -> str:
        return "tesseract"

    def process_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        page_number: int = 1,
    ) -> PageOCRResult:
        """Execute Tesseract OCR recognition on an image.

        Args:
            image_input: File path, numpy array, or PIL Image.
            page_number: 1-indexed page number.

        Returns:
            PageOCRResult with structured word, line, and block regions.

        Raises:
            OCREngineUnavailableError: If Tesseract binary is not installed or detected.
            OCRInvalidImageError: If the input image is invalid or empty.
            OCRProcessingError: If Tesseract fails during OCR processing.
        """
        # Step 1: Verify OCR availability
        avail = self.get_availability()
        if not avail.is_ready:
            raise OCREngineUnavailableError(
                engine_name=self.engine_name,
                message=f"Tesseract OCR is unavailable: {avail.status_message}",
                details=avail.to_dict(),
            )

        import pytesseract

        # Step 2: Load and validate image
        pil_img = self._prepare_pil_image(image_input)
        img_w, img_h = pil_img.size
        if img_w <= 0 or img_h <= 0:
            raise OCRInvalidImageError(
                f"Image for page {page_number} has invalid dimensions: {img_w}x{img_h}"
            )

        lang = self.config.languages[0] if self.config.languages else "eng"
        custom_config = f"--psm {self.config.page_segmentation_mode} --oem {self.config.ocr_engine_mode}"

        logger.info(
            f"Executing Tesseract OCR on page {page_number} ({img_w}x{img_h}, lang='{lang}', psm={self.config.page_segmentation_mode})"
        )

        start_time = time.perf_counter()

        # Step 3: Run Tesseract image_to_data
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            raise OCRProcessingError(
                message=f"Tesseract execution failed on page {page_number}: {str(e)}",
                page_number=page_number,
                engine_name=self.engine_name,
                details={"original_error": str(e), "duration_seconds": round(elapsed, 4)},
            ) from e

        elapsed = time.perf_counter() - start_time

        # Step 4: Parse tokens into structured models
        raw_words = self._parse_tesseract_data(data, page_number)
        ordered_words = sort_reading_order(raw_words)

        # Step 5: Construct lines, blocks, and unified text regions
        lines, blocks = self._group_words_into_lines_and_blocks(ordered_words, page_number)
        reconstructed_text = reconstruct_page_text(lines)

        # Convert lines/words to standard Core OCRTextRegion items
        text_regions = [line.to_text_region() for line in lines]

        mean_conf = (
            sum(w.confidence for w in ordered_words) / len(ordered_words)
            if ordered_words
            else None
        )

        logger.info(
            f"Completed OCR on page {page_number}: {len(ordered_words)} words, {len(lines)} lines, "
            f"mean_conf={round(mean_conf, 2) if mean_conf is not None else 0.0} in {elapsed:.3f}s"
        )

        return PageOCRResult(
            page_number=page_number,
            raw_text=reconstructed_text,
            blocks=blocks,
            lines=lines,
            words=ordered_words,
            text_regions=text_regions,
            engine_name=self.engine_name,
            engine_version=avail.engine_version,
            language=lang,
            mean_confidence=mean_conf,
            execution_time_seconds=round(elapsed, 4),
            provenance_metadata={
                "psm": self.config.page_segmentation_mode,
                "oem": self.config.ocr_engine_mode,
                "image_width": img_w,
                "image_height": img_h,
            },
        )

    def _prepare_pil_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> Image.Image:
        """Standardize various input types into a valid PIL Image."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise OCRInvalidImageError(f"Image path does not exist: {path}")
            try:
                return Image.open(path)
            except Exception as e:
                raise OCRInvalidImageError(f"Could not open image at {path}: {e}") from e

        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise OCRInvalidImageError("Numpy image array is empty.")
            if len(image_input.shape) == 2:
                return Image.fromarray(image_input, mode="L")
            elif len(image_input.shape) == 3:
                channels = image_input.shape[2]
                if channels == 3:
                    # Assume BGR from OpenCV, convert to RGB
                    rgb = image_input[:, :, ::-1]
                    return Image.fromarray(rgb, mode="RGB")
                elif channels == 4:
                    rgba = image_input[:, :, [2, 1, 0, 3]]
                    return Image.fromarray(rgba, mode="RGBA")
                else:
                    raise OCRInvalidImageError(f"Unsupported numpy channels: {channels}")

        elif isinstance(image_input, Image.Image):
            return image_input

        raise OCRInvalidImageError(f"Unsupported image input type: {type(image_input)}")

    def _parse_tesseract_data(
        self,
        data: Dict[str, List[Any]],
        page_number: int,
    ) -> List[OCRWord]:
        """Convert pytesseract dictionary output into typed OCRWord objects."""
        words: List[OCRWord] = []
        n_boxes = len(data.get("text", []))

        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            if not text:
                continue

            raw_conf_str = data["conf"][i]
            norm_conf, raw_conf = normalize_tesseract_confidence(raw_conf_str)

            # Ignore tokens with completely invalid confidence or empty text
            if norm_conf is None or norm_conf < self.config.min_confidence_threshold:
                continue

            left = float(data["left"][i])
            top = float(data["top"][i])
            width = float(data["width"][i])
            height = float(data["height"][i])

            if width <= 0 or height <= 0:
                continue

            bbox = BoundingBox(
                xmin=left,
                ymin=top,
                xmax=left + width,
                ymax=top + height,
                is_normalized=False,
            )

            word = OCRWord(
                text=text,
                confidence=norm_conf,
                raw_confidence=raw_conf if raw_conf is not None else 0.0,
                bounding_box=bbox,
                page_number=page_number,
                block_num=int(data.get("block_num", [0])[i]),
                par_num=int(data.get("par_num", [0])[i]),
                line_num=int(data.get("line_num", [0])[i]),
                word_num=int(data.get("word_num", [0])[i]),
            )
            words.append(word)

        return words

    def _group_words_into_lines_and_blocks(
        self,
        words: List[OCRWord],
        page_number: int,
    ) -> Tuple[List[OCRLine], List[OCRBlock]]:
        """Group recognized words into hierarchical OCRLine and OCRBlock structures."""
        if not words:
            return [], []

        # Group words by (block_num, par_num, line_num)
        line_groups: Dict[Tuple[int, int, int], List[OCRWord]] = {}
        for w in words:
            key = (w.block_num, w.par_num, w.line_num)
            line_groups.setdefault(key, []).append(w)

        lines: List[OCRLine] = []
        for (b_num, p_num, l_num), line_words in line_groups.items():
            line_words.sort(key=lambda w: w.bounding_box.xmin)
            line_text = " ".join(w.text for w in line_words)
            line_boxes = [w.bounding_box for w in line_words]
            line_union_box = compute_bounding_box_union(line_boxes)

            if not line_union_box:
                continue

            line_conf = sum(w.confidence for w in line_words) / len(line_words)
            line = OCRLine(
                text=line_text,
                words=line_words,
                bounding_box=line_union_box,
                confidence=round(line_conf, 4),
                page_number=page_number,
                block_num=b_num,
                par_num=p_num,
                line_num=l_num,
            )
            lines.append(line)

        # Sort lines vertically in reading order
        lines.sort(key=lambda l: (l.block_num, l.bounding_box.ymin, l.bounding_box.xmin))

        # Group lines into blocks
        block_groups: Dict[int, List[OCRLine]] = {}
        for line in lines:
            block_groups.setdefault(line.block_num, []).append(line)

        blocks: List[OCRBlock] = []
        for b_num, block_lines in block_groups.items():
            block_text = "\n".join(l.text for l in block_lines)
            block_boxes = [l.bounding_box for l in block_lines]
            block_union_box = compute_bounding_box_union(block_boxes)

            if not block_union_box:
                continue

            block_conf = sum(l.confidence for l in block_lines) / len(block_lines)
            block = OCRBlock(
                text=block_text,
                lines=block_lines,
                bounding_box=block_union_box,
                confidence=round(block_conf, 4),
                page_number=page_number,
                block_num=b_num,
            )
            blocks.append(block)

        blocks.sort(key=lambda b: (b.bounding_box.ymin, b.bounding_box.xmin))
        return lines, blocks
