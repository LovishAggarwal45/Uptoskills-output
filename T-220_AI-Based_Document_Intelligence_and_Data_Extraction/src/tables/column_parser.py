
"""Column boundary parsing, X-clustering, type inference, and horizontal alignment."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.core.logging import get_logger
from src.ocr.models import OCRWord
from src.tables.header_detector import match_canonical_header
from src.tables.models import TableColumn, TableHeader, TableValueType

logger = get_logger("tables.column_parser")

CURRENCY_PATTERN = re.compile(
    r"^[$€£¥₹]?\s*\(?-?\d+(?:,\d{3})*(?:\.\d{1,2})?\)?\s*[$€£¥₹]?$"
)
INTEGER_PATTERN = re.compile(r"^-?\d+$")
DECIMAL_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
DATE_PATTERN = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$"
    r"|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"
)


class ColumnParser:
    """Parse column boundaries, horizontal spans, and inferred column types."""

    def __init__(
        self,
        min_column_gap_px: float = 15.0,
        alignment_tolerance_px: float = 10.0,
    ) -> None:
        self.min_column_gap_px = min_column_gap_px
        self.alignment_tolerance_px = alignment_tolerance_px

    def parse_columns_from_header(
        self,
        header: TableHeader,
        data_words: List[OCRWord],
        page_width: Optional[float] = None,
    ) -> List[TableColumn]:
        """Build stable column boundaries from detected header positions."""
        if not header.cells:
            return []

        cells = sorted(
            header.cells,
            key=lambda cell: (
                cell.bounding_box.xmin if cell.bounding_box else 0.0
            ),
        )

        # Keep the recognized header cells as the column definitions.
        # Unrecognized header cells are retained because they may represent
        # HSN/SAC or other invoice-specific columns.
        centers = []
        for cell in cells:
            if cell.bounding_box is not None:
                centers.append(
                    (cell.bounding_box.xmin + cell.bounding_box.xmax) / 2.0
                )
            else:
                centers.append(float(len(centers)))

        count = len(cells)
        if count == 0:
            return []

        separators = [
            (centers[i] + centers[i + 1]) / 2.0
            for i in range(count - 1)
        ]

        buckets: List[List[OCRWord]] = [[] for _ in range(count)]

        for word in data_words:
            box = word.bounding_box
            midpoint = (box.xmin + box.xmax) / 2.0
            index = min(range(count), key=lambda i: abs(midpoint - centers[i]))
            buckets[index].append(word)

        columns: List[TableColumn] = []

        for i, cell in enumerate(cells):
            box = cell.bounding_box
            header_left = box.xmin if box else centers[i] - 5.0
            header_right = box.xmax if box else centers[i] + 5.0

            if i == 0:
                left = min(
                    header_left,
                    min(
                        (w.bounding_box.xmin for w in buckets[i]),
                        default=header_left,
                    ),
                )
                left = max(0.0, left - 3.0)
            else:
                left = separators[i - 1]

            if i == count - 1:
                right = max(
                    header_right,
                    max(
                        (w.bounding_box.xmax for w in buckets[i]),
                        default=header_right,
                    ),
                ) + 5.0
                if page_width is not None:
                    right = min(right, page_width)
            else:
                right = separators[i]

            if right <= left:
                right = left + 10.0

            canonical = cell.normalized_value
            if not isinstance(canonical, str) or not canonical:
                canonical = match_canonical_header(cell.text)

            inferred_type, alignment = self._infer_column_type_and_alignment(
                buckets[i], canonical
            )

            columns.append(
                TableColumn(
                    index=i,
                    name=cell.text.strip() or f"col_{i}",
                    canonical_field=canonical,
                    x_start=left,
                    x_end=right,
                    alignment=alignment,
                    inferred_type=inferred_type,
                    confidence=(
                        cell.confidence
                        if cell.confidence is not None
                        else 0.75
                    ),
                )
            )

        logger.info(
            "Header-based columns: %s",
            [
                {
                    "name": col.name,
                    "canonical": col.canonical_field,
                    "start": round(col.x_start, 1),
                    "end": round(col.x_end, 1),
                }
                for col in columns
            ],
        )
        return columns

    def parse_columns_from_words(
        self,
        words: List[OCRWord],
        page_width: Optional[float] = None,
    ) -> List[TableColumn]:
        """Infer headerless columns from repeated word X positions.

        This uses the distribution of word centers rather than merging all
        horizontally adjacent words into one long span. It is a fallback;
        explicit header-based boundaries are preferable whenever available.
        """
        if not words:
            return []

        sorted_words = sorted(
            words,
            key=lambda word: (
                (word.bounding_box.xmin + word.bounding_box.xmax) / 2.0
            ),
        )

        centers = [
            (word.bounding_box.xmin + word.bounding_box.xmax) / 2.0
            for word in sorted_words
        ]

        # Cluster X positions using gaps between word centers. The threshold
        # adapts to the observed spread, avoiding one column per OCR token.
        gaps = [
            centers[i + 1] - centers[i]
            for i in range(len(centers) - 1)
        ]
        positive_gaps = sorted(gap for gap in gaps if gap > 0)
        median_gap = (
            positive_gaps[len(positive_gaps) // 2]
            if positive_gaps
            else self.min_column_gap_px
        )
        threshold = max(self.min_column_gap_px * 2.0, median_gap * 2.5)

        clusters: List[List[OCRWord]] = [[sorted_words[0]]]

        for word in sorted_words[1:]:
            previous = clusters[-1][-1]
            previous_center = (
                previous.bounding_box.xmin + previous.bounding_box.xmax
            ) / 2.0
            current_center = (
                word.bounding_box.xmin + word.bounding_box.xmax
            ) / 2.0

            if current_center - previous_center > threshold:
                clusters.append([word])
            else:
                clusters[-1].append(word)

        # Avoid creating a huge number of token-based columns. The fallback
        # is intentionally conservative; table row parsing still uses geometry.
        if len(clusters) < 2:
            min_x = min(word.bounding_box.xmin for word in words)
            max_x = max(word.bounding_box.xmax for word in words)
            midpoint = (min_x + max_x) / 2.0
            spans = [(min_x, midpoint), (midpoint, max_x)]
        else:
            spans = [
                (
                    min(word.bounding_box.xmin for word in cluster),
                    max(word.bounding_box.xmax for word in cluster),
                )
                for cluster in clusters
            ]

        spans.sort(key=lambda span: span[0])
        boundaries: List[float] = []

        for i in range(len(spans) - 1):
            boundaries.append((spans[i][1] + spans[i + 1][0]) / 2.0)

        overall_left = min(word.bounding_box.xmin for word in words)
        overall_right = max(word.bounding_box.xmax for word in words)
        columns: List[TableColumn] = []

        for i, span in enumerate(spans):
            left = overall_left - 3.0 if i == 0 else boundaries[i - 1]
            right = (
                overall_right + 5.0
                if i == len(spans) - 1
                else boundaries[i]
            )

            left = max(0.0, left)
            if page_width is not None:
                right = min(right, page_width)
            if right <= left:
                right = left + 10.0

            column_words = [
                word
                for word in words
                if left
                <= (word.bounding_box.xmin + word.bounding_box.xmax) / 2.0
                < right
            ]

            inferred_type, alignment = self._infer_column_type_and_alignment(
                column_words, None
            )

            columns.append(
                TableColumn(
                    index=i,
                    name=f"col_{i}",
                    canonical_field=None,
                    x_start=left,
                    x_end=right,
                    alignment=alignment,
                    inferred_type=inferred_type,
                    confidence=0.60,
                )
            )

        logger.info(
            "Headerless column estimate produced %d columns",
            len(columns),
        )
        return columns

    def _infer_column_type_and_alignment(
        self,
        words: List[OCRWord],
        canonical: Optional[str] = None,
    ) -> Tuple[TableValueType, str]:
        """Infer a column's data type and horizontal alignment."""
        if canonical:
            if canonical in (
                "amount",
                "unit_price",
                "tax",
                "discount",
            ):
                return TableValueType.CURRENCY, "right"

            if canonical == "quantity":
                return TableValueType.INTEGER, "right"

            if canonical in (
                "description",
                "item_code",
                "unit_of_measure",
            ):
                return TableValueType.TEXT, "left"

        if not words:
            return TableValueType.TEXT, "left"

        nonempty = [
            word.text.strip().replace(" ", "")
            for word in words
            if word.text.strip()
        ]
        if not nonempty:
            return TableValueType.TEXT, "left"

        currency_count = 0
        integer_count = 0
        decimal_count = 0
        date_count = 0

        for value in nonempty:
            # Treat ordinary numeric values with decimals as monetary-like
            # only for type inference; currency symbols are not mandatory.
            if CURRENCY_PATTERN.fullmatch(value) and (
                any(symbol in value for symbol in "$€£¥₹")
                or "." in value
                or "," in value
            ):
                currency_count += 1
            elif INTEGER_PATTERN.fullmatch(value):
                integer_count += 1
            elif DECIMAL_PATTERN.fullmatch(value):
                decimal_count += 1
            elif DATE_PATTERN.fullmatch(value):
                date_count += 1

        total = len(nonempty)

        if currency_count / total >= 0.50:
            return TableValueType.CURRENCY, "right"
        if date_count / total >= 0.50:
            return TableValueType.DATE, "center"
        if decimal_count / total >= 0.50:
            return TableValueType.DECIMAL, "right"
        if integer_count / total >= 0.50:
            return TableValueType.INTEGER, "right"

        if len(words) >= 3:
            left_positions = [word.bounding_box.xmin for word in words]
            right_positions = [word.bounding_box.xmax for word in words]

            left_spread = max(left_positions) - min(left_positions)
            right_spread = max(right_positions) - min(right_positions)

            if (
                right_spread < left_spread
                and right_spread <= self.alignment_tolerance_px * 2
            ):
                return TableValueType.TEXT, "right"

            if left_spread <= self.alignment_tolerance_px * 2:
                return TableValueType.TEXT, "left"

        return TableValueType.TEXT, "left"


__all__ = [
    "ColumnParser",
    "CURRENCY_PATTERN",
    "INTEGER_PATTERN",
    "DECIMAL_PATTERN",
    "DATE_PATTERN",
]
