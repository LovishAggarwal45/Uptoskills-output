"""Post-processing algorithms for OCR confidence normalization, reading order, and text reconstruction."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.core.models import BoundingBox
from src.ocr.models import OCRBlock, OCRLine, OCRWord


def normalize_tesseract_confidence(raw_conf: float) -> Tuple[Optional[float], Optional[float]]:
    """Normalize Tesseract confidence scores (0-100 scale) to [0.0, 1.0].

    Tesseract returns -1.0 or negative values for non-text / whitespace tokens.

    Args:
        raw_conf: Raw confidence score from Tesseract.

    Returns:
        Tuple of (normalized_confidence, raw_confidence). Returns (None, None) if raw_conf is invalid (< 0).
    """
    try:
        conf_float = float(raw_conf)
    except (ValueError, TypeError):
        return None, None

    if conf_float < 0.0:
        return None, None

    # Tesseract scale is 0.0 - 100.0
    normalized = max(0.0, min(1.0, conf_float / 100.0))
    return round(normalized, 4), round(conf_float, 2)


def compute_bounding_box_union(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    """Compute the minimal bounding box enclosing a list of child bounding boxes.

    Args:
        boxes: List of BoundingBox instances.

    Returns:
        Union BoundingBox or None if list is empty.
    """
    if not boxes:
        return None

    is_norm = boxes[0].is_normalized
    min_x = min(b.xmin for b in boxes)
    min_y = min(b.ymin for b in boxes)
    max_x = max(b.xmax for b in boxes)
    max_y = max(b.ymax for b in boxes)

    return BoundingBox(
        xmin=min_x,
        ymin=min_y,
        xmax=max_x,
        ymax=max_y,
        is_normalized=is_norm,
    )


def sort_reading_order(words: List[OCRWord], line_tolerance_px: float = 8.0) -> List[OCRWord]:
    """Sort recognized words into natural top-to-bottom, left-to-right reading order.

    Algorithm:
    1. Cluster words into horizontal lines where vertical centers are within line_tolerance_px.
    2. Sort line clusters vertically by their mean Y-center.
    3. Sort words within each line cluster horizontally by xmin.
    4. Assign sequential reading_order indices (1, 2, ..., N).

    Args:
        words: List of detected OCRWord items.
        line_tolerance_px: Vertical distance tolerance in pixels for grouping into the same line.

    Returns:
        Sorted list of OCRWord instances with updated reading_order attributes.
    """
    if not words:
        return []

    # Sort primarily by vertical top coordinate
    sorted_by_y = sorted(words, key=lambda w: (w.bounding_box.ymin, w.bounding_box.xmin))

    lines: List[List[OCRWord]] = []
    for word in sorted_by_y:
        y_center = (word.bounding_box.ymin + word.bounding_box.ymax) / 2.0
        placed = False

        for line in lines:
            line_y_center = sum((w.bounding_box.ymin + w.bounding_box.ymax) / 2.0 for w in line) / len(line)
            if abs(y_center - line_y_center) <= line_tolerance_px:
                line.append(word)
                placed = True
                break

        if not placed:
            lines.append([word])

    # Sort each line cluster horizontally (left to right)
    ordered_words: List[OCRWord] = []
    order_idx = 1
    for line in lines:
        line.sort(key=lambda w: w.bounding_box.xmin)
        for w in line:
            w.reading_order = order_idx
            ordered_words.append(w)
            order_idx += 1

    return ordered_words


def reconstruct_page_text(lines: List[OCRLine], preserve_blocks: bool = True) -> str:
    """Reconstruct continuous page text preserving line breaks and paragraph structure.

    Args:
        lines: List of recognized OCRLine items in reading order.
        preserve_blocks: Whether to insert double newlines between distinct block IDs.

    Returns:
        Clean, formatted text string.
    """
    if not lines:
        return ""

    text_parts: List[str] = []
    prev_block_num: Optional[int] = None

    for line in lines:
        line_str = " ".join(w.text for w in line.words if w.text.strip()).strip()
        if not line_str:
            continue

        if preserve_blocks and prev_block_num is not None and line.block_num != prev_block_num:
            text_parts.append("\n" + line_str)
        else:
            text_parts.append(line_str)

        prev_block_num = line.block_num

    return "\n".join(text_parts)
