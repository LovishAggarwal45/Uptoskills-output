
"""Parse OCR words into table rows and assign words to their columns."""

from __future__ import annotations

import re
from statistics import median
from typing import Dict, List, Optional

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.ocr.models import OCRWord
from src.tables.models import TableCell, TableColumn, TableRow

logger = get_logger("tables.row_parser")


def is_summary_text(text: str) -> bool:
    """Identify invoice summary, tax, and totals rows."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False

    patterns = (
        r"\bsubtotal\b",
        r"\bgrand\s+total\b",
        r"\btotal\s+due\b",
        r"\bamount\s+due\b",
        r"\bbalance\s+due\b",
        r"\btotal\b",
        r"\btax\b",
        r"\bvat\b",
        r"\bgst\b",
        r"\bdiscount\b",
        r"\bshipping\b",
        r"\bhandling\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_artifact_text(text: str) -> bool:
    """Reject empty OCR fragments and punctuation-only artifacts."""
    value = (text or "").strip()
    return not value or re.search(r"[A-Za-z0-9]", value) is None


class RowParser:
    """Reconstruct table rows from OCR words and column geometry."""

    def __init__(
        self,
        line_height_tolerance_px: float = 10.0,
        max_multiline_gap_factor: float = 1.6,
    ) -> None:
        self.line_height_tolerance_px = line_height_tolerance_px
        self.max_multiline_gap_factor = max_multiline_gap_factor

    @staticmethod
    def _center_y(word: OCRWord) -> float:
        return (
            word.bounding_box.ymin + word.bounding_box.ymax
        ) / 2.0

    @staticmethod
    def _word_height(word: OCRWord) -> float:
        return max(
            1.0,
            word.bounding_box.ymax - word.bounding_box.ymin,
        )

    @staticmethod
    def _find_description_column_index(
        columns: List[TableColumn],
    ) -> int:
        """Find the description column by canonical field or header."""
        for index, column in enumerate(columns):
            canonical = (column.canonical_field or "").strip().lower()
            name = (column.name or "").strip().lower()

            if canonical == "description" or name in {
                "description",
                "item description",
                "product description",
            }:
                return index

        return min(1, len(columns) - 1)

    @staticmethod
    def _find_item_code_column_index(
        columns: List[TableColumn],
    ) -> Optional[int]:
        """Find the serial-number column."""
        for index, column in enumerate(columns):
            canonical = (column.canonical_field or "").strip().lower()
            name = (column.name or "").strip().lower()

            if canonical in {
                "item_code",
                "item_number",
                "serial_number",
            }:
                return index

            if name in {
                "sr. no.",
                "sr no",
                "s.no.",
                "item no.",
                "item number",
            }:
                return index

        return None

    @staticmethod
    def _find_hsn_column_index(
        columns: List[TableColumn],
    ) -> Optional[int]:
        """Find HSN/SAC or a similar product-code column."""
        for index, column in enumerate(columns):
            name = (column.name or "").strip().lower()
            canonical = (column.canonical_field or "").strip().lower()

            if canonical in {
                "hsn",
                "hsn_code",
                "sac",
                "hsn_sac",
            }:
                return index

            if any(term in name for term in ("hsn", "sac", "product code")):
                return index

        return None

    @staticmethod
    def _find_quantity_column_index(
        columns: List[TableColumn],
    ) -> Optional[int]:
        """Find the quantity column."""
        for index, column in enumerate(columns):
            name = (column.name or "").strip().lower()
            canonical = (column.canonical_field or "").strip().lower()

            if canonical == "quantity" or name in {
                "quantity",
                "qty",
                "q'ty",
            }:
                return index

        return None

    def _group_words_into_lines(
        self,
        words: List[OCRWord],
    ) -> List[List[OCRWord]]:
        """Group OCR words by vertical center."""
        ordered = sorted(
            words,
            key=lambda word: (
                self._center_y(word),
                word.bounding_box.xmin,
            ),
        )
        lines: List[List[OCRWord]] = []

        for word in ordered:
            center = self._center_y(word)
            best_line = None
            best_distance = float("inf")

            for line in lines:
                line_center = median(
                    self._center_y(item) for item in line
                )
                distance = abs(center - line_center)

                if (
                    distance <= self.line_height_tolerance_px
                    and distance < best_distance
                ):
                    best_line = line
                    best_distance = distance

            if best_line is None:
                lines.append([word])
            else:
                best_line.append(word)

        for line in lines:
            line.sort(key=lambda word: word.bounding_box.xmin)

        return sorted(
            lines,
            key=lambda line: median(
                self._center_y(word) for word in line
            ),
        )

    @staticmethod
    def _is_serial_number(value: str) -> bool:
        """Recognize a standalone row serial number."""
        return re.fullmatch(r"\d+[.)]?", value.strip()) is not None

    @staticmethod
    def _is_hsn_code(value: str) -> bool:
        """Recognize common numeric HSN/SAC codes."""
        return re.fullmatch(r"\d{6,8}", value.strip()) is not None

    def _assign_words_to_columns(
        self,
        words: List[OCRWord],
        columns: List[TableColumn],
    ) -> Dict[int, List[OCRWord]]:
        """Assign OCR words using header semantics and column geometry."""
        result: Dict[int, List[OCRWord]] = {
            index: [] for index in range(len(columns))
        }

        if not columns:
            return result

        description_index = self._find_description_column_index(columns)
        item_code_index = self._find_item_code_column_index(columns)
        hsn_index = self._find_hsn_column_index(columns)
        quantity_index = self._find_quantity_column_index(columns)

        bounds = [
            (
                min(column.x_start, column.x_end),
                max(column.x_start, column.x_end),
            )
            for column in columns
        ]

        logger.info("=== ROW WORD ASSIGNMENT START ===")
        logger.info(
            "Detected indices | serial=%s description=%s HSN=%s quantity=%s",
            item_code_index,
            description_index,
            hsn_index,
            quantity_index,
        )
        logger.info(
            "Column boundaries | %s",
            {
                index: {
                    "name": column.name,
                    "canonical": column.canonical_field,
                    "start": bounds[index][0],
                    "end": bounds[index][1],
                }
                for index, column in enumerate(columns)
            },
        )

        for word in words:
            value = word.text.strip()

            if is_artifact_text(value):
                continue

            left = word.bounding_box.xmin
            right = word.bounding_box.xmax
            midpoint = (left + right) / 2.0

            logger.info(
                "WORD CHECK | text=%r | xmin=%.2f | xmax=%.2f | "
                "ymin=%.2f | ymax=%.2f | midpoint=%.2f",
                value,
                left,
                right,
                word.bounding_box.ymin,
                word.bounding_box.ymax,
                midpoint,
            )

            selected_index = None
            reason = ""

            # Serial numbers must be near the left side of the table.
            if (
                item_code_index is not None
                and re.fullmatch(r"\d{1,3}", value)
                and midpoint <= bounds[item_code_index][1] + 30
            ):
                selected_index = item_code_index
                reason = "serial number"

            # HSN/SAC codes are assigned based on their header.
            elif hsn_index is not None and self._is_hsn_code(value):
                selected_index = hsn_index
                reason = "HSN/SAC code"

            # Description words can start a little before the column boundary.
            else:
                description_start, description_end = bounds[description_index]
                description_width = description_end - description_start

                if (
                    description_index != item_code_index
                    and re.search(r"[A-Za-z]", value)
                    and left < description_start
                    and right >= description_start - min(
                        125.0, max(0.0, description_width)
                    )
                ):
                    selected_index = description_index
                    reason = "description boundary recovery"

            # Short numeric values are quantities only when their coordinates
            # are near the quantity column.
            if (
                selected_index is None
                and quantity_index is not None
                and re.fullmatch(r"\d{1,4}", value)
            ):
                q_start, q_end = bounds[quantity_index]

                if q_start - 40 <= midpoint <= q_end + 40:
                    selected_index = quantity_index
                    reason = "quantity coordinate"

            # Use geometric boundaries when no semantic rule applies.
            if selected_index is None:
                candidates = []

                for index, (start, end) in enumerate(bounds):
                    if start <= midpoint <= end:
                        distance = 0.0
                    else:
                        distance = min(
                            abs(midpoint - start),
                            abs(midpoint - end),
                        )

                    overlap = max(
                        0.0,
                        min(right, end) - max(left, start),
                    )
                    candidates.append((distance, -overlap, index))

                _, _, selected_index = min(candidates)
                reason = "geometric fallback"

            result[selected_index].append(word)

            logger.info(
                "WORD ASSIGNED | text=%r | column_index=%d | "
                "column_name=%r | reason=%s",
                value,
                selected_index,
                columns[selected_index].name,
                reason,
            )

        for assigned in result.values():
            assigned.sort(
                key=lambda word: (
                    word.bounding_box.ymin,
                    word.bounding_box.xmin,
                )
            )

        logger.info(
            "FINAL COLUMN ASSIGNMENT | %s",
            {
                columns[index].name: [
                    word.text for word in assigned
                ]
                for index, assigned in result.items()
            },
        )
        logger.info("=== ROW WORD ASSIGNMENT END ===")

        return result

    @staticmethod
    def _make_cell(
        row_index: int,
        column_index: int,
        words: List[OCRWord],
        page_number: int,
    ) -> TableCell:
        """Create a table cell from its OCR words."""
        ordered = sorted(
            words,
            key=lambda word: word.bounding_box.xmin,
        )
        text = " ".join(
            word.text.strip()
            for word in ordered
            if word.text.strip()
        )

        boxes = [word.bounding_box for word in ordered]
        cell_box = None

        if boxes:
            cell_box = BoundingBox(
                xmin=min(box.xmin for box in boxes),
                ymin=min(box.ymin for box in boxes),
                xmax=max(box.xmax for box in boxes),
                ymax=max(box.ymax for box in boxes),
            )

        return TableCell(
            row_index=row_index,
            col_index=column_index,
            text=text,
            bounding_box=cell_box,
            page_number=page_number,
            words=ordered,
        )

    @staticmethod
    def _merge_row_bounding_box(
        current: Optional[BoundingBox],
        additional_words: List[OCRWord],
    ) -> Optional[BoundingBox]:
        """Return a bounding box covering existing and additional words."""
        if not additional_words:
            return current

        boxes = [word.bounding_box for word in additional_words]

        if current is not None:
            xmin = min(current.xmin, *(box.xmin for box in boxes))
            ymin = min(current.ymin, *(box.ymin for box in boxes))
            xmax = max(current.xmax, *(box.xmax for box in boxes))
            ymax = max(current.ymax, *(box.ymax for box in boxes))
        else:
            xmin = min(box.xmin for box in boxes)
            ymin = min(box.ymin for box in boxes)
            xmax = max(box.xmax for box in boxes)
            ymax = max(box.ymax for box in boxes)

        return BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    def parse_rows(
        self,
        words: List[OCRWord],
        columns: List[TableColumn],
        page_number: int = 1,
        header_ymax: Optional[float] = None,
    ) -> List[TableRow]:
        """Group OCR words into rows and merge wrapped descriptions."""
        if not words or not columns:
            return []

        content_words = [
            word
            for word in words
            if word.text.strip()
            and not is_artifact_text(word.text)
            and (
                header_ymax is None
                or word.bounding_box.ymin >= header_ymax
            )
        ]

        if not content_words:
            return []

        lines = self._group_words_into_lines(content_words)
        description_index = self._find_description_column_index(columns)
        item_code_index = self._find_item_code_column_index(columns)

        rows: List[TableRow] = []

        for line in lines:
            column_map = self._assign_words_to_columns(line, columns)
            continuation = False

            if rows:
                description_words = column_map.get(description_index, [])
                has_description = bool(description_words)

                has_serial = (
                    item_code_index is not None
                    and bool(column_map.get(item_code_index))
                )

                has_other_values = any(
                    index != description_index and bool(cell_words)
                    for index, cell_words in column_map.items()
                )

                if has_description and not has_serial and not has_other_values:
                    previous_row = rows[-1]
                    previous_box = previous_row.bounding_box
                    current_top = min(
                        word.bounding_box.ymin for word in line
                    )

                    if previous_box is not None:
                        gap = max(
                            0.0,
                            current_top - previous_box.ymax,
                        )
                        typical_height = median(
                            self._word_height(word) for word in line
                        )
                        allowed_gap = max(
                            self.line_height_tolerance_px,
                            typical_height * self.max_multiline_gap_factor,
                        )

                        previous_text = " ".join(
                            cell.text
                            for cell in previous_row.cells
                            if cell.text
                        )
                        previous_description = (
                            previous_row.cells[description_index]
                        )

                        if (
                            gap <= allowed_gap
                            and not is_summary_text(previous_text)
                            and previous_description.text
                        ):
                            combined_words = (
                                previous_description.words + description_words
                            )
                            combined_words.sort(
                                key=lambda word: (
                                    word.bounding_box.ymin,
                                    word.bounding_box.xmin,
                                )
                            )

                            previous_description.words = combined_words
                            previous_description.text = " ".join(
                                word.text.strip()
                                for word in combined_words
                            )
                            previous_description.bounding_box = (
                                self._merge_row_bounding_box(
                                    previous_description.bounding_box,
                                    description_words,
                                )
                            )
                            previous_row.bounding_box = (
                                self._merge_row_bounding_box(
                                    previous_row.bounding_box,
                                    line,
                                )
                            )
                            continuation = True

            if continuation:
                continue

            row_index = len(rows)

            cells = [
                self._make_cell(
                    row_index=row_index,
                    column_index=index,
                    words=column_map.get(index, []),
                    page_number=page_number,
                )
                for index in range(len(columns))
            ]

            row_box = self._merge_row_bounding_box(None, line)

            rows.append(
                TableRow(
                    row_index=row_index,
                    cells=cells,
                    bounding_box=row_box,
                    page_number=page_number,
                    is_summary=is_summary_text(
                        " ".join(cell.text for cell in cells)
                    ),
                )
            )

        return rows


__all__ = [
    "RowParser",
    "is_summary_text",
    "is_artifact_text",
]
