
"""Header candidate detection, keyword synonym matching, and spatial header localization."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.ocr.models import OCRLine, OCRWord
from src.tables.models import TableCell, TableHeader, TableValueType

logger = get_logger("tables.header_detector")

CANONICAL_HEADER_SYNONYMS: Dict[str, Set[str]] = {
    "item_code": {
        "code", "item #", "item no", "item no.", "item code", "sku",
        "part #", "part no", "product code", "reference", "item id",
        "sl no", "sl. no.", "sr no", "sr. no.", "s.no", "s.no.",
        "sn", "s/n", "pos", "line", "line #", "line no", "serial no",
        "hsn", "hsn code", "hsn/sac", "sac",
    },
    "description": {
        "description", "item description", "product description",
        "details", "particulars", "service", "services", "goods",
        "name", "item name", "product", "products", "desc",
        "specification", "item / description", "items", "activity",
        "item details",
    },
    "quantity": {
        "quantity", "qty", "qty.", "qnty", "units", "unit", "count",
        "hours", "hrs", "pieces", "pcs", "pcs.", "volume", "units sold",
    },
    "unit_price": {
        "unit price", "unit cost", "unit rate", "price", "rate", "cost",
        "price/unit", "price / unit", "cost/unit", "each", "unit amt",
        "price per unit", "unit_price", "unit price (inr)",
    },
    "amount": {
        "amount", "total", "line total", "ext price", "extended price",
        "net amount", "subtotal", "value", "total price", "item total",
        "line amount", "ext amount", "total amt", "amount due",
        "charges", "gross amount", "line amount (inr)",
    },
    "tax": {
        "tax", "vat", "gst", "tax rate", "tax %", "tax amount", "cgst",
        "sgst", "igst", "sales tax", "tax amt",
    },
    "discount": {
        "discount", "disc", "disc.", "disc %", "disc amount",
        "rebate", "allowance",
    },
    "unit_of_measure": {
        "uom", "u/m", "measure", "unit of measure",
    },
}


def _normalize_header(text: str) -> str:
    """Normalize OCR header text before matching."""
    text = (text or "").lower().strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"[^\w\s#./%+-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def match_canonical_header(text: str) -> Optional[str]:
    """Map a header label or OCR phrase to a canonical column category."""
    cleaned = _normalize_header(text)
    if not cleaned:
        return None

    for canonical, synonyms in CANONICAL_HEADER_SYNONYMS.items():
        normalized_synonyms = {_normalize_header(item) for item in synonyms}
        if cleaned in normalized_synonyms:
            return canonical

    # More specific phrases must be tested before generic words such as
    # "price", "total", "unit", or "code".
    patterns = [
        ("item_code", [
            r"\bhsn(?:\s*/\s*sac)?(?:\s+code)?\b",
            r"\bsac\b",
            r"\bsku\b",
            r"\bitem\s*(?:#|no\.?|number|code|id)\b",
            r"\bpart\s*(?:#|no\.?|number|code)\b",
            r"\bproduct\s+code\b",
            r"\bserial\s+no\b",
            r"\bsl\.?\s*no\.?\b",
            r"\bsr\.?\s*no\.?\b",
        ]),
        ("description", [
            r"\bitem\s+details\b",
            r"\bitem\s+description\b",
            r"\bproduct\s+description\b",
            r"\bdescription\b",
            r"\bparticulars\b",
            r"\bdetails\b",
            r"\bproduct\b",
            r"\bitem\s+name\b",
            r"\bitem\b",
            r"\bdesc\.?\b",
        ]),
        ("quantity", [
            r"\bquantity\b",
            r"\bqty\.?\b",
            r"\bqnty\b",
            r"\bpieces\b",
            r"\bpcs\.?\b",
            r"\bunits\b",
            r"\bcount\b",
        ]),
        ("unit_price", [
            r"\bunit\s+price\b",
            r"\bprice\s*/\s*unit\b",
            r"\bprice\s+per\s+unit\b",
            r"\bunit\s+rate\b",
            r"\bunit\s+cost\b",
            r"\brate\b",
            r"\bprice\b",
            r"\bcost\b",
            r"\beach\b",
        ]),
        ("amount", [
            r"\bextended\s+price\b",
            r"\bext\.?\s*(?:price|amount)\b",
            r"\bline\s+(?:total|amount)\b",
            r"\bnet\s+amount\b",
            r"\bitem\s+total\b",
            r"\btotal\s+price\b",
            r"\bamount\b",
            r"\btotal\b",
            r"\bvalue\b",
        ]),
        ("tax", [r"\b(?:tax|vat|gst|cgst|sgst|igst)\b"]),
        ("discount", [r"\bdiscount\b", r"\bdisc\.?\b"]),
        ("unit_of_measure", [r"\buom\b", r"\bunit\s+of\s+measure\b"]),
    ]

    for canonical, expressions in patterns:
        if any(re.search(expression, cleaned) for expression in expressions):
            return canonical

    return None


class HeaderDetector:
    """Detect and localize table headers from OCR words or lines."""

    def __init__(
        self,
        min_header_columns: int = 2,
        line_height_tolerance_px: float = 12.0,
        min_confidence_score: float = 0.40,
    ) -> None:
        self.min_header_columns = min_header_columns
        self.line_height_tolerance_px = line_height_tolerance_px
        self.min_confidence_score = min_confidence_score

    def detect_header_candidates(
        self,
        words: List[OCRWord],
        lines: Optional[List[OCRLine]] = None,
        page_number: int = 1,
    ) -> List[TableHeader]:
        """Find horizontal lines containing multiple recognizable column headers."""
        if not words:
            return []

        candidate_lines = (
            lines if lines else self._cluster_words_into_lines(words, page_number)
        )
        candidates: List[TableHeader] = []

        for row_idx, line in enumerate(candidate_lines):
            line_words = sorted(line.words, key=lambda word: word.bounding_box.xmin)
            if len(line_words) < self.min_header_columns:
                continue

            cells, canonical_set, score = self._evaluate_header_line(
                line_words, page_number
            )

            if (
                len(canonical_set) < self.min_header_columns
                or score < self.min_confidence_score
            ):
                continue

            bbox = BoundingBox(
                xmin=min(word.bounding_box.xmin for word in line_words),
                ymin=min(word.bounding_box.ymin for word in line_words),
                xmax=max(word.bounding_box.xmax for word in line_words),
                ymax=max(word.bounding_box.ymax for word in line_words),
            )

            candidates.append(
                TableHeader(
                    row_index=row_idx,
                    cells=cells,
                    bounding_box=bbox,
                    confidence=min(1.0, score),
                    page_number=page_number,
                    column_names=[cell.text for cell in cells],
                )
            )

            logger.debug(
                "Detected header on page %s: %r; canonical fields=%s; score=%.2f",
                page_number,
                line.text,
                sorted(canonical_set),
                score,
            )

        # Prefer the strongest candidate if multiple text lines look like headers.
        candidates.sort(
            key=lambda header: (
                -header.confidence,
                header.bounding_box.ymin if header.bounding_box else 0,
            )
        )
        return candidates

    def _cluster_words_into_lines(
        self,
        words: List[OCRWord],
        page_number: int,
    ) -> List[OCRLine]:
        """Group OCR words into lines using vertical center proximity."""
        if not words:
            return []

        sorted_words = sorted(
            words,
            key=lambda word: (
                (word.bounding_box.ymin + word.bounding_box.ymax) / 2.0,
                word.bounding_box.xmin,
            ),
        )
        groups: List[List[OCRWord]] = []

        for word in sorted_words:
            word_mid_y = (
                word.bounding_box.ymin + word.bounding_box.ymax
            ) / 2.0
            tolerance = max(
                self.line_height_tolerance_px,
                word.bounding_box.height * 0.6,
            )
            matching_group = None

            for group in groups:
                group_mid_y = sum(
                    (item.bounding_box.ymin + item.bounding_box.ymax) / 2.0
                    for item in group
                ) / len(group)

                if abs(word_mid_y - group_mid_y) <= tolerance:
                    matching_group = group
                    break

            if matching_group is None:
                groups.append([word])
            else:
                matching_group.append(word)

        result: List[OCRLine] = []

        for line_idx, group in enumerate(
            sorted(groups, key=lambda items: min(w.bounding_box.ymin for w in items))
        ):
            group.sort(key=lambda word: word.bounding_box.xmin)
            text = " ".join(word.text for word in group)

            result.append(
                OCRLine(
                    text=text,
                    words=group,
                    bounding_box=BoundingBox(
                        xmin=min(word.bounding_box.xmin for word in group),
                        ymin=min(word.bounding_box.ymin for word in group),
                        xmax=max(word.bounding_box.xmax for word in group),
                        ymax=max(word.bounding_box.ymax for word in group),
                    ),
                    confidence=sum(word.confidence for word in group) / len(group),
                    page_number=page_number,
                    line_num=line_idx,
                )
            )

        return result

    def _evaluate_header_line(
        self,
        words: List[OCRWord],
        page_number: int,
    ) -> Tuple[List[TableCell], Set[str], float]:
        """Group adjacent words into header cells and score recognized labels."""
        if not words:
            return [], set(), 0.0

        words = sorted(words, key=lambda word: word.bounding_box.xmin)
        clusters: List[List[OCRWord]] = []
        current_cluster: List[OCRWord] = [words[0]]

        # Header words can be split by OCR ("Unit" + "Price"), so group nearby
        # words first and split when there is a strong column gap.
        for current in words[1:]:
            previous = current_cluster[-1]
            gap = current.bounding_box.xmin - previous.bounding_box.xmax
            char_width = max(
                1.0,
                previous.bounding_box.width / max(1, len(previous.text)),
            )
            split_threshold = max(22.0, char_width * 3.5)

            previous_phrase = " ".join(item.text for item in current_cluster)
            current_phrase = current.text.strip()

            if gap > split_threshold:
                clusters.append(current_cluster)
                current_cluster = [current]
            elif (
                match_canonical_header(previous_phrase)
                and match_canonical_header(current_phrase)
                and gap > char_width * 1.2
            ):
                clusters.append(current_cluster)
                current_cluster = [current]
            else:
                current_cluster.append(current)

        clusters.append(current_cluster)

        cells: List[TableCell] = []
        canonical_set: Set[str] = set()
        confidences: List[float] = []

        for col_idx, cluster in enumerate(clusters):
            cell_text = " ".join(word.text for word in cluster).strip()
            canonical = match_canonical_header(cell_text)
            confidence = sum(word.confidence for word in cluster) / len(cluster)
            confidences.append(confidence)

            if canonical:
                canonical_set.add(canonical)

            cells.append(
                TableCell(
                    row_index=0,
                    col_index=col_idx,
                    text=cell_text,
                    normalized_value=canonical,
                    bounding_box=BoundingBox(
                        xmin=min(word.bounding_box.xmin for word in cluster),
                        ymin=min(word.bounding_box.ymin for word in cluster),
                        xmax=max(word.bounding_box.xmax for word in cluster),
                        ymax=max(word.bounding_box.ymax for word in cluster),
                    ),
                    confidence=confidence,
                    page_number=page_number,
                    value_type=TableValueType.TEXT,
                    is_header=True,
                    words=cluster,
                    metadata={"canonical_category": canonical} if canonical else {},
                )
            )

        # Use unique recognized categories; repeated "total" labels do not count
        # as separate columns.
        count = len(canonical_set)
        if count >= 4:
            base_score = 0.95
        elif count == 3:
            base_score = 0.85
        elif count == 2:
            acceptable_pairs = (
                {"description", "amount"},
                {"quantity", "unit_price"},
                {"item_code", "amount"},
                {"description", "quantity"},
                {"description", "unit_price"},
            )
            base_score = (
                0.80
                if any(pair.issubset(canonical_set) for pair in acceptable_pairs)
                else 0.0
            )
        else:
            base_score = 0.0

        mean_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        score = base_score * 0.8 + mean_confidence * 0.2

        return cells, canonical_set, score


__all__ = [
    "CANONICAL_HEADER_SYNONYMS",
    "match_canonical_header",
    "HeaderDetector",
]
