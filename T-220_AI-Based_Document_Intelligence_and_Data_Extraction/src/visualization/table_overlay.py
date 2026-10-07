"""Specialized overlay builder for 2D table structures, rows, cells, and line items."""

from __future__ import annotations

from typing import List, Optional

from src.core.types import SeverityLevel, ValidationStatus
from src.visualization.legend import PALETTE, BADGE_SYMBOLS, get_color_for_annotation
from src.visualization.models import AnnotationType, EvidenceRegion, OverlayAnnotation


class TableOverlayBuilder:
    """Constructs visual overlay annotations for detected tables, rows, cells, and line items."""

    @classmethod
    def build_table_annotation(
        cls,
        region: EvidenceRegion,
        show_confidence: bool = True,
        line_width: int = 2,
    ) -> Optional[OverlayAnnotation]:
        """Convert a table EvidenceRegion into a renderable OverlayAnnotation."""
        if not region.is_spatial or not region.bbox:
            return None

        badge = BADGE_SYMBOLS.get(region.region_type, "[TBL]")
        label_parts = [badge]

        if region.table_id:
            label_parts.append(region.table_id)
        if region.column_name:
            label_parts.append(f"Col: {region.column_name}")
        if region.row_index is not None:
            label_parts.append(f"Row {region.row_index + 1}")

        if show_confidence and region.extraction_confidence is not None:
            label_parts.append(f"({region.extraction_confidence:.2f})")

        label_text = " ".join(label_parts)
        tooltip = f"Table Region: {region.table_id or 'table'}"
        if region.source_text:
            tooltip += f" | Text: '{region.source_text}'"
        if region.review_reasons:
            tooltip += f" | Issues: {', '.join(region.review_reasons)}"

        # Check if this line item or cell has an arithmetic or validation failure
        has_math_error = (
            region.validation_status == ValidationStatus.INVALID
            or any("arithmetic" in r.lower() or "calculation" in r.lower() or "mismatch" in r.lower() for r in region.review_reasons)
        )

        if has_math_error:
            border_rgba = PALETTE["table_math_err"]
            fill_rgba = PALETTE["table_math_err_fill"]
            severity = SeverityLevel.CRITICAL
            label_text = f"[MATH-ERR] {label_text}"
        else:
            border_rgba, fill_rgba = get_color_for_annotation(
                annotation_type=region.region_type,
                validation_status=region.validation_status,
                confidence_band=region.confidence_band,
            )
            severity = SeverityLevel.INFO

        return OverlayAnnotation(
            annotation_id=f"ann_{region.evidence_id}",
            page_number=region.page_number,
            annotation_type=region.region_type,
            bounding_box=region.bbox,
            label=label_text,
            tooltip=tooltip,
            severity=severity,
            confidence_band=region.confidence_band,
            confidence_score=region.extraction_confidence,
            source_evidence_id=region.evidence_id,
            table_id=region.table_id,
            row_index=region.row_index,
            column_name=region.column_name,
            review_reasons=list(region.review_reasons),
            color_rgba=border_rgba,
            fill_rgba=fill_rgba,
            line_width=line_width,
            badge_text=badge,
            is_visible=True,
            rendering_metadata={
                "table_id": region.table_id,
                "has_math_error": has_math_error,
                "source_text": region.source_text,
            },
        )

    @classmethod
    def build_all(
        cls,
        regions: List[EvidenceRegion],
        show_confidence: bool = True,
        line_width: int = 2,
    ) -> List[OverlayAnnotation]:
        """Build annotations for all table, row, cell, and line-item evidence regions."""
        table_types = (
            AnnotationType.TABLE_BOUNDS,
            AnnotationType.TABLE_HEADER,
            AnnotationType.TABLE_ROW,
            AnnotationType.TABLE_CELL,
            AnnotationType.TABLE_LINE_ITEM,
        )
        annotations: List[OverlayAnnotation] = []
        for reg in regions:
            if reg.region_type in table_types:
                ann = cls.build_table_annotation(
                    reg,
                    show_confidence=show_confidence,
                    line_width=line_width,
                )
                if ann:
                    annotations.append(ann)
        return annotations


__all__ = ["TableOverlayBuilder"]
