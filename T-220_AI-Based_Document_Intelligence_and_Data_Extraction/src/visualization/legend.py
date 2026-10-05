"""Visual legend, color palette mapping, and accessible symbol definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from src.core.types import SeverityLevel, ValidationStatus
from src.confidence.models import ConfidenceBand
from src.visualization.models import AnnotationType


# Standard Semantic Color Palette (RGBA)
PALETTE: Dict[str, Tuple[int, int, int, int]] = {
    # Neutral OCR
    "ocr_word": (120, 120, 120, 180),
    "ocr_word_fill": (120, 120, 120, 25),
    "ocr_line": (100, 100, 100, 200),
    "ocr_block": (80, 80, 80, 220),

    # Fields & Entities (Blue / Teal)
    "field": (30, 144, 255, 255),          # Dodger Blue
    "field_fill": (30, 144, 255, 45),
    "entity": (0, 180, 180, 255),           # Teal
    "entity_fill": (0, 180, 180, 45),

    # Tables (Purple)
    "table_bounds": (138, 43, 226, 255),    # Blue Violet
    "table_bounds_fill": (138, 43, 226, 30),
    "table_cell": (147, 112, 219, 220),     # Medium Purple
    "table_cell_fill": (147, 112, 219, 35),
    "table_math_err": (220, 20, 60, 255),   # Crimson
    "table_math_err_fill": (220, 20, 60, 65),

    # Validation Statuses
    "valid": (34, 139, 34, 255),            # Forest Green
    "valid_fill": (34, 139, 34, 40),
    "warning": (255, 140, 0, 255),          # Dark Orange / Amber
    "warning_fill": (255, 140, 0, 50),
    "error": (220, 20, 60, 255),            # Crimson Red
    "error_fill": (220, 20, 60, 60),
    "critical": (178, 34, 34, 255),         # Firebrick Red
    "critical_fill": (178, 34, 34, 75),

    # Confidence Bands
    "conf_high": (34, 139, 34, 255),
    "conf_medium": (255, 165, 0, 255),
    "conf_low": (255, 69, 0, 255),
    "conf_very_low": (178, 34, 34, 255),

    # Review Alerts
    "review_target": (220, 20, 60, 255),
    "review_target_fill": (220, 20, 60, 50),
}

# Accessible badges for non-color dependent legibility
BADGE_SYMBOLS: Dict[AnnotationType, str] = {
    AnnotationType.OCR_WORD: "[OCR]",
    AnnotationType.OCR_LINE: "[LINE]",
    AnnotationType.OCR_BLOCK: "[BLK]",
    AnnotationType.EXTRACTED_FIELD: "[FLD]",
    AnnotationType.EXTRACTED_ENTITY: "[ENT]",
    AnnotationType.TABLE_BOUNDS: "[TBL]",
    AnnotationType.TABLE_HEADER: "[HDR]",
    AnnotationType.TABLE_ROW: "[ROW]",
    AnnotationType.TABLE_CELL: "[CELL]",
    AnnotationType.TABLE_LINE_ITEM: "[ITEM]",
    AnnotationType.VALIDATION_PASS: "[PASS]",
    AnnotationType.VALIDATION_WARNING: "[WARN]",
    AnnotationType.VALIDATION_ERROR: "[ERR]",
    AnnotationType.VALIDATION_CRITICAL: "[CRIT]",
    AnnotationType.CONFIDENCE_HIGH: "[HIGH]",
    AnnotationType.CONFIDENCE_MEDIUM: "[MED]",
    AnnotationType.CONFIDENCE_LOW: "[LOW]",
    AnnotationType.CONFIDENCE_VERY_LOW: "[V.LOW]",
    AnnotationType.REVIEW_TARGET: "[REVIEW]",
    AnnotationType.REVIEW_CRITICAL: "[URGENT]",
    AnnotationType.UNLOCATED_ISSUE: "[DOC-ISSUE]",
}


def get_color_for_annotation(
    annotation_type: AnnotationType,
    severity: Optional[SeverityLevel] = None,
    validation_status: Optional[ValidationStatus] = None,
    confidence_band: Optional[ConfidenceBand] = None,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Determine border and fill RGBA colors for a given annotation context.

    Returns:
        (border_rgba, fill_rgba)
    """
    # 1. Critical & Errors take top priority
    if severity == SeverityLevel.CRITICAL or validation_status == ValidationStatus.INVALID:
        return PALETTE["error"], PALETTE["error_fill"]

    if severity == SeverityLevel.WARNING or validation_status == ValidationStatus.WARNING:
        return PALETTE["warning"], PALETTE["warning_fill"]

    if validation_status == ValidationStatus.VALID:
        return PALETTE["valid"], PALETTE["valid_fill"]

    # 2. Review targets
    if annotation_type in (AnnotationType.REVIEW_TARGET, AnnotationType.REVIEW_CRITICAL):
        return PALETTE["review_target"], PALETTE["review_target_fill"]

    # 3. Tables
    if annotation_type in (
        AnnotationType.TABLE_BOUNDS,
        AnnotationType.TABLE_HEADER,
        AnnotationType.TABLE_ROW,
        AnnotationType.TABLE_CELL,
        AnnotationType.TABLE_LINE_ITEM,
    ):
        return PALETTE["table_bounds"], PALETTE["table_cell_fill"]

    # 4. Fields & Entities
    if annotation_type == AnnotationType.EXTRACTED_FIELD:
        return PALETTE["field"], PALETTE["field_fill"]
    if annotation_type == AnnotationType.EXTRACTED_ENTITY:
        return PALETTE["entity"], PALETTE["entity_fill"]

    # 5. OCR tokens
    if annotation_type in (AnnotationType.OCR_WORD, AnnotationType.OCR_LINE, AnnotationType.OCR_BLOCK):
        return PALETTE["ocr_word"], PALETTE["ocr_word_fill"]

    # Default fallback
    return PALETTE["field"], PALETTE["field_fill"]


class LegendRenderer:
    """Renders a structured, accessible visual legend bar onto an image canvas."""

    @classmethod
    def render_legend_banner(
        cls,
        canvas_width: int,
        line_height: int = 24,
    ) -> Image.Image:
        """Create a standalone banner image representing the visual legend.

        Args:
            canvas_width: Width of the banner image in pixels.
            line_height: Height of the text entries.

        Returns:
            PIL Image representing the legend.
        """
        banner_h = 44
        legend_img = Image.new("RGBA", (canvas_width, banner_h), (245, 245, 245, 240))
        draw = ImageDraw.Draw(legend_img)

        # Draw top divider
        draw.line([(0, 0), (canvas_width, 0)], fill=(200, 200, 200, 255), width=1)

        font = ImageFont.load_default()

        # Legend items: (label, border_color, fill_color, badge)
        items = [
            ("Extracted Field", PALETTE["field"], PALETTE["field_fill"], "[FLD]"),
            ("Table Structure", PALETTE["table_bounds"], PALETTE["table_cell_fill"], "[TBL]"),
            ("Warning / Low Conf", PALETTE["warning"], PALETTE["warning_fill"], "[WARN]"),
            ("Validation Error / Review", PALETTE["error"], PALETTE["error_fill"], "[REVIEW]"),
            ("OCR Evidence", PALETTE["ocr_word"], PALETTE["ocr_word_fill"], "[OCR]"),
        ]

        x_offset = 12
        y_offset = 12
        chip_w = 16
        chip_h = 16

        for label, bcolor, fcolor, badge in items:
            # Draw color chip box
            draw.rectangle(
                [x_offset, y_offset, x_offset + chip_w, y_offset + chip_h],
                fill=fcolor,
                outline=bcolor,
                width=2,
            )
            # Draw text
            text_str = f"{badge} {label}"
            draw.text((x_offset + chip_w + 6, y_offset + 2), text_str, fill=(40, 40, 40, 255), font=font)

            # Move offset
            bbox = draw.textbbox((0, 0), text_str, font=font)
            text_w = bbox[2] - bbox[0]
            x_offset += chip_w + text_w + 24

        return legend_img


__all__ = ["PALETTE", "BADGE_SYMBOLS", "get_color_for_annotation", "LegendRenderer"]
