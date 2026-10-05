"""Multi-page table continuation detection, geometry matching, and logical table consolidation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.tables.base import BaseTableContinuationDetector
from src.tables.models import Table, TableColumn, TableRow

logger = get_logger("tables.continuation")


class TableContinuationDetector(BaseTableContinuationDetector):
    """Detects when tables continue across page breaks and links them into logical tables."""

    def __init__(
        self,
        min_column_overlap_ratio: float = 0.65,
        max_column_count_diff: int = 0,
    ) -> None:
        self.min_column_overlap_ratio = min_column_overlap_ratio
        self.max_column_count_diff = max_column_count_diff

    def link_continuations(
        self,
        tables: List[Table],
    ) -> List[Table]:
        """Identify multi-page continuous tables and link their continuation metadata.

        Args:
            tables: List of all reconstructed Table instances sorted by page number.

        Returns:
            List of updated Table instances with is_continuation and continuation_of_table_id set.
        """
        if len(tables) < 2:
            return tables

        sorted_tables = sorted(tables, key=lambda t: (t.page_number, t.bounding_box.ymin if t.bounding_box else 0.0))

        for i in range(len(sorted_tables) - 1):
            t_curr = sorted_tables[i]
            t_next = sorted_tables[i + 1]

            # Check if next table is on the immediately following page
            if t_next.page_number == t_curr.page_number + 1:
                if self._are_tables_continuous(t_curr, t_next):
                    t_next.is_continuation = True
                    t_next.continuation_of_table_id = t_curr.table_id
                    logger.info(
                        f"Detected Multi-Page Table Continuation: "
                        f"Table '{t_next.table_id}' (Page {t_next.page_number}) continues Table '{t_curr.table_id}' (Page {t_curr.page_number})."
                    )

        return sorted_tables

    def merge_continued_tables(self, tables: List[Table]) -> List[Table]:
        """Optionally merge continuous multi-page table fragments into unified logical tables."""
        linked_tables = self.link_continuations(tables)
        if not linked_tables:
            return []

        merged_results: List[Table] = []
        table_by_id = {t.table_id: t for t in linked_tables}
        visited = set()

        for t in linked_tables:
            if t.table_id in visited:
                continue

            if not t.is_continuation:
                # Find all chain descendants
                chain = [t]
                curr_id = t.table_id
                visited.add(curr_id)

                for next_t in linked_tables:
                    if next_t.is_continuation and next_t.continuation_of_table_id == curr_id:
                        chain.append(next_t)
                        curr_id = next_t.table_id
                        visited.add(next_t.table_id)

                if len(chain) == 1:
                    merged_results.append(t)
                else:
                    # Merge chain into consolidated table
                    merged = self._merge_table_chain(chain)
                    merged_results.append(merged)
            else:
                visited.add(t.table_id)

        return merged_results

    def _are_tables_continuous(self, t1: Table, t2: Table) -> bool:
        """Check if two tables across consecutive pages represent the same continuous table."""
        # Check column counts
        if abs(t1.column_count - t2.column_count) > self.max_column_count_diff:
            return False

        if not t1.columns or not t2.columns:
            return False

        # Check horizontal column span alignment overlap
        overlaps: List[float] = []
        for c1, c2 in zip(t1.columns, t2.columns):
            # Compute 1D interval overlap
            start = max(c1.x_start, c2.x_start)
            end = min(c1.x_end, c2.x_end)
            intersection = max(0.0, end - start)
            union = max(c1.x_end, c2.x_end) - min(c1.x_start, c2.x_start)

            if union > 0:
                overlaps.append(intersection / union)
            else:
                overlaps.append(0.0)

        mean_overlap = sum(overlaps) / len(overlaps)
        if mean_overlap >= self.min_column_overlap_ratio:
            return True

        # Check header similarity if t2 also has headers
        if t1.headers and t2.headers:
            h1 = [h.strip().lower() for h in t1.headers]
            h2 = [h.strip().lower() for h in t2.headers]
            if h1 == h2:
                return True

        return False

    def _merge_table_chain(self, chain: List[Table]) -> Table:
        """Merge a sequence of continuous table pages into a single logical Table."""
        primary = chain[0]

        all_rows: List[TableRow] = list(primary.rows)
        all_line_items = list(primary.line_items)

        for cont_table in chain[1:]:
            all_rows.extend(cont_table.rows)
            all_line_items.extend(cont_table.line_items)

        # Re-index row indices consecutively
        for idx, row in enumerate(all_rows):
            row.row_index = idx
            for cell in row.cells:
                cell.row_index = idx

        for idx, item in enumerate(all_line_items):
            item.row_index = idx

        avg_conf = sum(t.confidence for t in chain) / len(chain)

        return Table(
            table_id=primary.table_id,
            page_number=primary.page_number,
            headers=list(primary.headers),
            columns=list(primary.columns),
            rows=all_rows,
            line_items=all_line_items,
            bounding_box=primary.bounding_box,
            confidence=avg_conf,
            confidence_source=primary.confidence_source,
            validation_status=primary.validation_status,
            validation_result=primary.validation_result,
            is_continuation=False,
            continuation_of_table_id=None,
            metadata={
                "is_multipage_merged": True,
                "merged_page_count": len(chain),
                "source_page_numbers": [t.page_number for t in chain],
                "source_table_ids": [t.table_id for t in chain],
            },
        )


__all__ = [
    "TableContinuationDetector",
]
