"""2D grid structure assembly, coordinate validation, and table model reconstruction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.ocr.models import OCRWord
from src.tables.base import BaseTableReconstructor
from src.tables.cell_parser import CellParser
from src.tables.column_parser import ColumnParser
from src.tables.header_detector import HeaderDetector
from src.tables.models import (
    Table,
    TableCell,
    TableColumn,
    TableHeader,
    TableRegion,
    TableRow,
    TableValueType,
)
from src.tables.row_parser import RowParser

logger = get_logger("tables.structure")


class TableStructureReconstructor(BaseTableReconstructor):
    """Reconstructs 2D grid matrix of columns, rows, and cells from OCR words in a table region."""

    def __init__(
        self,
        column_parser: Optional[ColumnParser] = None,
        row_parser: Optional[RowParser] = None,
        header_detector: Optional[HeaderDetector] = None,
    ) -> None:
        self.column_parser = column_parser or ColumnParser()
        self.row_parser = row_parser or RowParser()
        self.header_detector = header_detector or HeaderDetector()

    def reconstruct_table(
        self,
        region: TableRegion,
        words: List[OCRWord],
        table_id: str,
        page_number: int = 1,
    ) -> Table:
        """Reconstruct 2D grid structure from detected region and word tokens.

        Args:
            region: Detected TableRegion candidate.
            words: All OCR words on the page (filtered to region bounds).
            table_id: Unique identifier for the table.
            page_number: 1-indexed page number.

        Returns:
            Fully structured Table instance.
        """
        # Filter words within the region bounding box
        region_words = [
            w for w in words
            if self._word_inside_region(w, region.bounding_box)
        ]

        # 1. Establish Header and Columns
        header_ymax: Optional[float] = None
        if region.header and region.header.cells:
            header = region.header
            header_ymax = header.bounding_box.ymax if header.bounding_box else None
            # Filter words below the header for column/row parsing
            data_words = [w for w in region_words if header_ymax is None or w.bounding_box.ymin >= (header_ymax - 2.0)]
            columns = self.column_parser.parse_columns_from_header(
                header=header,
                data_words=data_words,
                page_width=region.bounding_box.xmax,
            )
        else:
            # Fallback for headerless table
            columns = self.column_parser.parse_columns_from_words(
                words=region_words,
                page_width=region.bounding_box.xmax,
            )
            header = None

        if not columns:
            # Fallback single column if parsing failed
            xmin = region.bounding_box.xmin
            xmax = region.bounding_box.xmax
            columns = [
                TableColumn(
                    index=0,
                    name="Content",
                    canonical_field="description",
                    x_start=xmin,
                    x_end=xmax,
                    alignment="left",
                    inferred_type=TableValueType.TEXT,
                )
            ]

        # 2. Parse Rows
        rows = self.row_parser.parse_rows(
            words=region_words,
            columns=columns,
            page_number=page_number,
            header_ymax=header_ymax,
        )

        # 3. Enrich Cell Values with Type Normalization
        for row in rows:
            for col_idx, cell in enumerate(row.cells):
                col = columns[col_idx] if col_idx < len(columns) else None
                CellParser.populate_cell_values(cell, column=col)

        # 4. Compute overall Table Bounding Box
        all_boxes: List[BoundingBox] = []
        if header and header.bounding_box:
            all_boxes.append(header.bounding_box)
        for r in rows:
            if r.bounding_box:
                all_boxes.append(r.bounding_box)

        if all_boxes:
            t_xmin = min(b.xmin for b in all_boxes)
            t_ymin = min(b.ymin for b in all_boxes)
            t_xmax = max(b.xmax for b in all_boxes)
            t_ymax = max(b.ymax for b in all_boxes)
            table_bbox = BoundingBox(xmin=t_xmin, ymin=t_ymin, xmax=t_xmax, ymax=t_ymax)
        else:
            table_bbox = region.bounding_box

        headers_list = [c.name for c in columns]

        logger.info(
            f"Reconstructed Table '{table_id}' on Page {page_number}: "
            f"{len(columns)} columns, {len(rows)} rows."
        )

        return Table(
            table_id=table_id,
            page_number=page_number,
            headers=headers_list,
            columns=columns,
            rows=rows,
            line_items=[],  # Populated subsequently by LineItemExtractor
            bounding_box=table_bbox,
            confidence=region.confidence_score,
            metadata={
                "has_explicit_header": header is not None,
                "region_signals": region.detection_signals,
            },
        )

    def _word_inside_region(self, word: OCRWord, region_bbox: BoundingBox, margin_px: float = 8.0) -> bool:
        """Check if an OCR word falls within the table bounding region."""
        w_mid_x = (word.bounding_box.xmin + word.bounding_box.xmax) / 2.0
        w_mid_y = (word.bounding_box.ymin + word.bounding_box.ymax) / 2.0

        return (
            (region_bbox.xmin - margin_px) <= w_mid_x <= (region_bbox.xmax + margin_px)
            and (region_bbox.ymin - margin_px) <= w_mid_y <= (region_bbox.ymax + margin_px)
        )


__all__ = [
    "TableStructureReconstructor",
]
