"""Specialized overlay builder for human review targets and validation issue highlights."""

from __future__ import annotations

from typing import List, Optional

from src.core.types import SeverityLevel
from src.visualization.legend import PALETTE, BADGE_SYMBOLS, get_color_for_annotation
from src.visualization.models import AnnotationType, EvidenceRegion, OverlayAnnotation


class ReviewOverlayBuilder:
    """Constructs high-visibility visual overlay annotations for review targets and validation issues."""

    @classmethod
    def build_review_annotation(
        cls,
        region: EvidenceRegion,
        show_reasons: bool = True,
        line_width: int = 3,
    ) -> Optional[OverlayAnnotation]:
        """Convert a review target or validation issue EvidenceRegion into an OverlayAnnotation.

        Args:
            region: EvidenceRegion representing a human review target or validation issue.
            show_reasons: If True, include the primary trigger reason in the visible label.
            line_width: High-visibility border thickness in pixels (default 3px).

        Returns:
            OverlayAnnotation or None if region is non-spatial.
        """
        if not region.is_spatial or not region.bbox:
            return None

        badge = BADGE_SYMBOLS.get(region.region_type, "[REVIEW]")
        target_name = region.field_name or region.table_id or "Target"

        if region.review_reasons and show_reasons:
            primary_reason = region.review_reasons[0]
            # Truncate reason if too long for single line badge
            if len(primary_reason) > 35:
                primary_reason = primary_reason[:32] + "..."
            label_text = f"{badge} {target_name}: {primary_reason}"
        else:
            label_text = f"{badge} {target_name}"

        tooltip = f"Review Alert: {target_name}"
        if region.review_reasons:
            tooltip += f" | Reasons: {'; '.join(region.review_reasons)}"

        is_critical = (
            region.region_type in (AnnotationType.VALIDATION_CRITICAL, AnnotationType.REVIEW_CRITICAL)
            or any("critical" in r.lower() or "arithmetic" in r.lower() for r in region.review_reasons)
        )

        if is_critical:
            border_rgba = PALETTE["critical"]
            fill_rgba = PALETTE["critical_fill"]
            severity = SeverityLevel.CRITICAL
        elif region.region_type in (AnnotationType.VALIDATION_WARNING, AnnotationType.CONFIDENCE_LOW):
            border_rgba = PALETTE["warning"]
            fill_rgba = PALETTE["warning_fill"]
            severity = SeverityLevel.WARNING
        else:
            border_rgba = PALETTE["error"]
            fill_rgba = PALETTE["error_fill"]
            severity = SeverityLevel.ERROR

        return OverlayAnnotation(
            annotation_id=f"ann_rev_{region.evidence_id}",
            page_number=region.page_number,
            annotation_type=region.region_type,
            bounding_box=region.bbox,
            label=label_text,
            tooltip=tooltip,
            severity=severity,
            confidence_band=region.confidence_band,
            confidence_score=region.aggregated_confidence or region.extraction_confidence,
            source_evidence_id=region.evidence_id,
            field_name=region.field_name,
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
                "is_critical": is_critical,
                "review_reasons": region.review_reasons,
            },
        )

    @classmethod
    def build_all(
        cls,
        regions: List[EvidenceRegion],
        show_reasons: bool = True,
        line_width: int = 3,
    ) -> List[OverlayAnnotation]:
        """Build annotations for all review target and validation issue evidence regions."""
        review_types = (
            AnnotationType.REVIEW_TARGET,
            AnnotationType.REVIEW_CRITICAL,
            AnnotationType.VALIDATION_CRITICAL,
            AnnotationType.VALIDATION_ERROR,
            AnnotationType.VALIDATION_WARNING,
        )
        annotations: List[OverlayAnnotation] = []
        for reg in regions:
            if reg.region_type in review_types:
                ann = cls.build_review_annotation(
                    reg,
                    show_reasons=show_reasons,
                    line_width=line_width,
                )
                if ann:
                    annotations.append(ann)
        return annotations


__all__ = ["ReviewOverlayBuilder"]
