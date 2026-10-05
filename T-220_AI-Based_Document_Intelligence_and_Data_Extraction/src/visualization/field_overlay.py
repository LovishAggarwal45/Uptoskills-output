"""Specialized overlay builder for extracted fields and generic entities."""

from __future__ import annotations

from typing import List, Optional

from src.visualization.legend import get_color_for_annotation, BADGE_SYMBOLS
from src.visualization.models import AnnotationType, EvidenceRegion, OverlayAnnotation


class FieldOverlayBuilder:
    """Constructs visual overlay annotations for extracted key-value fields and named entities."""

    @classmethod
    def build_field_annotation(
        cls,
        region: EvidenceRegion,
        show_confidence: bool = True,
        show_normalized_value: bool = True,
        line_width: int = 2,
    ) -> Optional[OverlayAnnotation]:
        """Convert a field/entity EvidenceRegion into a renderable OverlayAnnotation.

        Args:
            region: EvidenceRegion instance representing an extracted field or entity.
            show_confidence: If True, include confidence score in the visible label.
            show_normalized_value: If True, include normalized value in the tooltip.
            line_width: Border thickness in pixels.

        Returns:
            OverlayAnnotation or None if the region is not spatial.
        """
        if not region.is_spatial or not region.bbox:
            return None

        badge = BADGE_SYMBOLS.get(region.region_type, "[FLD]")
        field_display = region.field_name or "field"

        # Construct label text
        if show_confidence and region.extraction_confidence is not None:
            label_text = f"{badge} {field_display} ({region.extraction_confidence:.2f})"
        else:
            label_text = f"{badge} {field_display}"

        # Construct tooltip
        norm_val_str = str(region.normalized_value) if region.normalized_value is not None else str(region.source_text or "")
        tooltip_text = f"{field_display}: {norm_val_str}"
        if region.review_reasons:
            tooltip_text += f" | Flags: {', '.join(region.review_reasons)}"

        # Determine severity and colors
        border_rgba, fill_rgba = get_color_for_annotation(
            annotation_type=region.region_type,
            validation_status=region.validation_status,
            confidence_band=region.confidence_band,
        )

        return OverlayAnnotation(
            annotation_id=f"ann_{region.evidence_id}",
            page_number=region.page_number,
            annotation_type=region.region_type,
            bounding_box=region.bbox,
            label=label_text,
            tooltip=tooltip_text,
            confidence_band=region.confidence_band,
            confidence_score=region.extraction_confidence or region.aggregated_confidence,
            source_evidence_id=region.evidence_id,
            field_name=region.field_name,
            review_reasons=list(region.review_reasons),
            color_rgba=border_rgba,
            fill_rgba=fill_rgba,
            line_width=line_width,
            badge_text=badge,
            is_visible=True,
            rendering_metadata={
                "source_text": region.source_text,
                "normalized_value": region.normalized_value,
                "source_method": region.source_method,
            },
        )

    @classmethod
    def build_all(
        cls,
        regions: List[EvidenceRegion],
        show_confidence: bool = True,
        line_width: int = 2,
    ) -> List[OverlayAnnotation]:
        """Build annotations for all field and entity evidence regions."""
        annotations: List[OverlayAnnotation] = []
        for reg in regions:
            if reg.region_type in (AnnotationType.EXTRACTED_FIELD, AnnotationType.EXTRACTED_ENTITY):
                ann = cls.build_field_annotation(
                    reg,
                    show_confidence=show_confidence,
                    line_width=line_width,
                )
                if ann:
                    annotations.append(ann)
        return annotations


__all__ = ["FieldOverlayBuilder"]
