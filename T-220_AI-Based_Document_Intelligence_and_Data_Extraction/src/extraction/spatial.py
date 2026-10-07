"""Spatial proximity, bounding box alignment, and label-value pairing heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import ExtractionSpatialConfig
from src.core.models import BoundingBox, DocumentPage, OCRTextRegion


@dataclass
class SpatialMatch:
    """Represents a spatial or layout-associated label and value pair."""
    label_text: str
    value_text: str
    page_number: int
    label_bbox: Optional[BoundingBox] = None
    value_bbox: Optional[BoundingBox] = None
    mean_ocr_confidence: Optional[float] = None
    spatial_score: float = 1.0
    match_type: str = "inline_colon"  # inline_colon, same_line_adjacent, vertically_below
    source_line: str = ""


def merge_bounding_boxes(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    """Merge a collection of bounding boxes into a single bounding envelope."""
    if not boxes:
        return None
    xmins = [b.xmin for b in boxes]
    ymins = [b.ymin for b in boxes]
    xmaxs = [b.xmax for b in boxes]
    ymaxs = [b.ymax for b in boxes]
    return BoundingBox(
        xmin=min(xmins),
        ymin=min(ymins),
        xmax=max(xmaxs),
        ymax=max(ymaxs),
        is_normalized=boxes[0].is_normalized,
    )


def extract_inline_label_value(
    line_text: str,
    label_keywords: List[str],
) -> Optional[Tuple[str, str]]:
    """Check if a text line contains a known label followed by a separator and value.

    Returns:
        Tuple of (matched_label, extracted_value_text) or None.
    """
    # Sort labels by descending length so multi-word labels (e.g. 'sales tax') match before substrings (e.g. 'tax')
    sorted_labels = sorted(label_keywords, key=lambda l: len(l.strip()), reverse=True)
    for label in sorted_labels:
        label_lower = label.lower().strip()
        # Clean trailing colons and whitespace from label keyword
        clean_lbl = label_lower.rstrip(":\t ")
        if not clean_lbl:
            continue

        tokens = clean_lbl.split()
        lbl_regex = r"\s+".join(re.escape(t) for t in tokens)

        # Avoid false matches where a single-word label is preceded by an adjective/qualifier on the same line
        # e.g. 'Due Date' matching 'date', 'Email Address' matching 'address', or 'Tax ID' matching 'tax'
        if len(tokens) == 1:
            if clean_lbl == "date" and re.search(r"(?i)\b(?:due|pay|payment|expiry|delivery|ship|shipping|order|created)\s+date\b", line_text):
                continue
            if clean_lbl == "address" and re.search(r"(?i)\b(?:email|e-mail|web|ip|mac|url)\s+address\b", line_text):
                continue
            if clean_lbl in ("tax", "vat", "gst"):
                if re.search(r"(?i)\b(?:tax\s+id|vat\s+id|gstin|vat\s+no|tax\s+no|reg\s+no|registration)\b", line_text):
                    continue
                if re.search(rf"(?i)\b{clean_lbl}-[A-Za-z0-9]", line_text):
                    continue

        if clean_lbl.endswith("#"):
            pattern = rf"(?i)\b{lbl_regex}\s*[:\-.]*\s*(.+)$"
        else:
            pattern = rf"(?i)\b{lbl_regex}\b(?:\s*\([^)]*\)|\s*@\s*[\d.]+\s*%)?\s*[:#\-.]*\s*(.+)$"

        match = re.search(pattern, line_text)
        if match:
            val = match.group(1).strip()
            if val:
                return label, val
    return None


def find_label_value_matches(
    page: DocumentPage,
    label_keywords: List[str],
    config: Optional[ExtractionSpatialConfig] = None,
) -> List[SpatialMatch]:
    """Search a DocumentPage for label-value associations via inline text and 2D spatial layout.

    Args:
        page: DocumentPage with OCR regions and raw text.
        label_keywords: List of label variations to search for (e.g. ['invoice number', 'inv #']).
        config: Spatial proximity thresholds.

    Returns:
        List of SpatialMatch candidate pairs.
    """
    cfg = config or ExtractionSpatialConfig()
    matches: List[SpatialMatch] = []
    seen_values: set = set()

    # =========================================================================
    # Strategy 1: Inline line-based parsing (most common & highest precision)
    # =========================================================================
    lines = page.raw_text.split("\n") if page.raw_text else []
    for line in lines:
        if not line.strip():
            continue
        inline_pair = extract_inline_label_value(line, label_keywords)
        if inline_pair:
            lbl, val = inline_pair
            val_clean = val.strip()
            if val_clean and val_clean not in seen_values:
                seen_values.add(val_clean)

                # Locate spatial coordinates from OCR regions for this value
                val_box, val_conf = locate_value_in_regions(val_clean, page.ocr_text_regions)

                matches.append(
                    SpatialMatch(
                        label_text=lbl,
                        value_text=val_clean,
                        page_number=page.page_number,
                        value_bbox=val_box,
                        mean_ocr_confidence=val_conf,
                        spatial_score=1.0,
                        match_type="inline_colon",
                        source_line=line,
                    )
                )

    # =========================================================================
    # Strategy 2: 2D Spatial Proximity on OCR Regions (for disjoint bounding boxes)
    # =========================================================================
    regions = page.ocr_text_regions
    for idx, reg in enumerate(regions):
        reg_text_lower = reg.text.lower().strip()
        matched_lbl = None
        for lbl in label_keywords:
            clean_lbl = lbl.lower().rstrip(": #.-")
            if reg_text_lower == clean_lbl or reg_text_lower.startswith(clean_lbl + ":"):
                matched_lbl = lbl
                break

        if not matched_lbl or not reg.bounding_box:
            continue

        lbl_box = reg.bounding_box

        # Search for horizontal neighbor to the right
        best_right_candidate: Optional[OCRTextRegion] = None
        min_h_gap = float("inf")

        for other_idx, other_reg in enumerate(regions):
            if other_idx == idx or not other_reg.bounding_box or not other_reg.text.strip():
                continue

            other_box = other_reg.bounding_box
            # Check same horizontal line
            v_overlap = max(0.0, min(lbl_box.ymax, other_box.ymax) - max(lbl_box.ymin, other_box.ymin))
            is_same_line = v_overlap > (lbl_box.height * 0.4) or abs(lbl_box.ymin - other_box.ymin) <= cfg.same_line_tolerance_px

            if is_same_line and other_box.xmin >= (lbl_box.xmax - 5.0):
                h_gap = other_box.xmin - lbl_box.xmax
                if 0 <= h_gap <= cfg.max_horizontal_gap_px and h_gap < min_h_gap:
                    min_h_gap = h_gap
                    best_right_candidate = other_reg

        if best_right_candidate and best_right_candidate.text.strip() not in seen_values:
            val_clean = best_right_candidate.text.strip().lstrip(": ")
            seen_values.add(val_clean)
            matches.append(
                SpatialMatch(
                    label_text=matched_lbl,
                    value_text=val_clean,
                    page_number=page.page_number,
                    label_bbox=lbl_box,
                    value_bbox=best_right_candidate.bounding_box,
                    mean_ocr_confidence=best_right_candidate.confidence,
                    spatial_score=0.90,
                    match_type="same_line_adjacent",
                    source_line=f"{reg.text} {best_right_candidate.text}",
                )
            )

        # Search for vertically aligned neighbor below
        best_below_candidate: Optional[OCRTextRegion] = None
        min_v_gap = float("inf")

        for other_idx, other_reg in enumerate(regions):
            if other_idx == idx or not other_reg.bounding_box or not other_reg.text.strip():
                continue

            other_box = other_reg.bounding_box
            h_align = abs(lbl_box.xmin - other_box.xmin)
            if h_align <= cfg.align_tolerance_px and other_box.ymin >= (lbl_box.ymax - 2.0):
                v_gap = other_box.ymin - lbl_box.ymax
                if 0 <= v_gap <= cfg.max_vertical_gap_px and v_gap < min_v_gap:
                    min_v_gap = v_gap
                    best_below_candidate = other_reg

        if best_below_candidate and best_below_candidate.text.strip() not in seen_values:
            val_clean = best_below_candidate.text.strip()
            seen_values.add(val_clean)
            matches.append(
                SpatialMatch(
                    label_text=matched_lbl,
                    value_text=val_clean,
                    page_number=page.page_number,
                    label_bbox=lbl_box,
                    value_bbox=best_below_candidate.bounding_box,
                    mean_ocr_confidence=best_below_candidate.confidence,
                    spatial_score=0.85,
                    match_type="vertically_below",
                    source_line=f"{reg.text}\n{best_below_candidate.text}",
                )
            )

    return matches


def locate_value_in_regions(
    val_text: str,
    regions: List[OCRTextRegion],
) -> Tuple[Optional[BoundingBox], Optional[float]]:
    """Search OCR regions to locate spatial bounding box and confidence for an extracted value string."""
    if not val_text or not regions:
        return None, None

    val_lower = val_text.lower().strip()
    val_words = val_lower.split()

    matching_boxes: List[BoundingBox] = []
    matching_confs: List[float] = []

    for reg in regions:
        reg_text_lower = reg.text.lower().strip()
        if val_lower in reg_text_lower or any(w in reg_text_lower for w in val_words):
            if reg.bounding_box:
                matching_boxes.append(reg.bounding_box)
            if reg.confidence is not None:
                matching_confs.append(reg.confidence)

    if not matching_boxes:
        return None, None

    merged_box = merge_bounding_boxes(matching_boxes)
    mean_conf = sum(matching_confs) / len(matching_confs) if matching_confs else None
    return merged_box, mean_conf
