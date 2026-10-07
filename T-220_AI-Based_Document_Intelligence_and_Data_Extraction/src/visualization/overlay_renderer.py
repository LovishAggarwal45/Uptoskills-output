"""Deterministic image overlay renderer applying spatial annotations, alpha blending, and badges."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from src.core.config import VisualizationConfig
from src.core.logging import get_logger
from src.core.models import BoundingBox, DocumentPage
from src.visualization.coordinates import CoordinateTransformer
from src.visualization.exceptions import (
    OverlayRenderingError,
    SourcePageUnavailableError,
)
from src.visualization.legend import LegendRenderer, PALETTE
from src.visualization.models import (
    AnnotationType,
    OverlayAnnotation,
    PageVisualization,
    VisualCoordinateSystem,
)

logger = get_logger("visualization.renderer")


class OverlayRenderer:
    """Renders visual overlays on document page images using Pillow with alpha transparency."""

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or VisualizationConfig()

    def render_page(
        self,
        document_id: str,
        page: DocumentPage,
        annotations: List[OverlayAnnotation],
        output_dir: Union[str, Path] = "outputs/visualizations",
        source_image_override: Optional[Union[str, Path]] = None,
    ) -> PageVisualization:
        """Render all active visual annotations on a single document page.

        Args:
            document_id: Unique document identifier.
            page: DocumentPage to render.
            annotations: List of OverlayAnnotation objects for this page.
            output_dir: Base output directory.
            source_image_override: Optional explicit image path.

        Returns:
            PageVisualization aggregate with overlay artifact path and metadata.
        """
        start_time = time.perf_counter()
        target_dir = Path(output_dir) / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        overlay_filename = f"page_{page.page_number:03d}_overlay.{self.config.output.format}"
        overlay_path = target_dir / overlay_filename

        # 1. Resolve and open source image
        image_path = self._resolve_source_image(page, source_image_override)
        if not image_path.exists():
            raise SourcePageUnavailableError(
                f"Cannot render overlay: source image not found at '{image_path}'",
                details={"document_id": document_id, "page_number": page.page_number, "path": str(image_path)},
            )

        try:
            with Image.open(image_path) as src_img:
                base_img = src_img.convert("RGBA")
        except Exception as e:
            raise OverlayRenderingError(
                f"Failed to open source image for rendering: {e}",
                details={"image_path": str(image_path), "error": str(e)},
            ) from e

        img_w, img_h = base_img.size

        # 2. Filter annotations by enabled layers
        active_annotations, enabled_layers = self._filter_active_annotations(annotations)

        # 3. Create transparent overlay layer for alpha blending
        overlay_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay_layer)

        # Sort annotations so that critical/review targets are rendered on top
        sorted_annotations = self._sort_annotations_by_priority(active_annotations)

        # 4. Render Bounding Boxes & Fills
        labels_drawn = 0
        max_labels = self.config.rendering.max_labels_per_page
        font = ImageFont.load_default()

        for ann in sorted_annotations:
            if not ann.bounding_box:
                continue

            try:
                pixel_box = CoordinateTransformer.to_pixel_box(
                    bbox=ann.bounding_box,
                    image_width=img_w,
                    image_height=img_h,
                    clip_to_page=self.config.coordinates.clip_to_page,
                )
            except Exception as e:
                logger.warning(f"Skipping malformed bounding box in annotation {ann.annotation_id}: {e}")
                continue

            coords = [pixel_box.xmin, pixel_box.ymin, pixel_box.xmax, pixel_box.ymax]

            # A. Draw semi-transparent fill
            if ann.fill_rgba:
                fill_color = self._apply_opacity(ann.fill_rgba, self.config.rendering.opacity)
                draw_overlay.rectangle(coords, fill=fill_color)

            # B. Draw solid border
            border_width = ann.line_width or self.config.rendering.line_width
            draw_overlay.rectangle(coords, outline=ann.color_rgba, width=border_width)

            # C. Draw Label & Badge Pill (if enabled)
            if self.config.rendering.show_labels and ann.label and labels_drawn < max_labels:
                self._draw_label_pill(
                    draw=draw_overlay,
                    pixel_box=pixel_box,
                    label=ann.label,
                    border_color=ann.color_rgba,
                    font=font,
                )
                labels_drawn += 1

        # 5. Composite overlay layer onto base image
        composite_img = Image.alpha_composite(base_img, overlay_layer)

        # 6. Append Legend Banner (if enabled)
        if self.config.rendering.draw_legend:
            legend_img = LegendRenderer.render_legend_banner(canvas_width=img_w)
            final_img = Image.new("RGBA", (img_w, img_h + legend_img.height), (255, 255, 255, 255))
            final_img.paste(composite_img, (0, 0))
            final_img.paste(legend_img, (0, img_h))
        else:
            final_img = composite_img

        # 7. Save output artifact
        final_rgb = final_img.convert("RGB")
        final_rgb.save(overlay_path, format=self.config.output.format.upper())

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Rendered overlay for page {page.page_number} ({len(sorted_annotations)} annotations, "
            f"{labels_drawn} labels) -> {overlay_path.name} in {elapsed:.3f}s"
        )

        return PageVisualization(
            document_id=document_id,
            page_number=page.page_number,
            source_image_path=image_path,
            image_width=img_w,
            image_height=img_h,
            coordinate_system=VisualCoordinateSystem.ABSOLUTE_PIXEL,
            evidence_regions=[],
            annotations=active_annotations,
            overlay_image_path=overlay_path,
            rendered_layers=enabled_layers,
            rendering_status="rendered",
            warnings=[],
            metadata={
                "duration_seconds": round(elapsed, 4),
                "labels_drawn": labels_drawn,
                "legend_attached": self.config.rendering.draw_legend,
            },
        )

    def _resolve_source_image(
        self,
        page: DocumentPage,
        override_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Find the best available image path for the page."""
        if override_path and Path(override_path).exists():
            return Path(override_path)

        if page.image_path:
            p = Path(page.image_path)
            if p.exists():
                return p

            # Sibling processed check
            processed_sibling = p.parent / f"page_{page.page_number:03d}_processed.png"
            if processed_sibling.exists():
                return processed_sibling

            # Sibling original check
            orig_sibling = p.parent / f"page_{page.page_number:03d}_original.png"
            if orig_sibling.exists():
                return orig_sibling

        # Check default staging location in data/processed
        default_candidate = Path("data/processed") / f"page_{page.page_number:03d}_processed.png"
        if default_candidate.exists():
            return default_candidate

        return Path(page.image_path or f"page_{page.page_number}.png")

    def _filter_active_annotations(
        self,
        annotations: List[OverlayAnnotation],
    ) -> Tuple[List[OverlayAnnotation], List[str]]:
        """Filter annotations according to configuration layer toggles."""
        layers = self.config.layers
        active: List[OverlayAnnotation] = []
        enabled_layer_names: List[str] = []

        if layers.ocr:
            enabled_layer_names.append("ocr")
        if layers.fields:
            enabled_layer_names.append("fields")
        if layers.entities:
            enabled_layer_names.append("entities")
        if layers.tables:
            enabled_layer_names.append("tables")
        if layers.validation:
            enabled_layer_names.append("validation")
        if layers.confidence:
            enabled_layer_names.append("confidence")
        if layers.review:
            enabled_layer_names.append("review")

        for ann in annotations:
            t = ann.annotation_type
            if t in (AnnotationType.OCR_WORD, AnnotationType.OCR_LINE, AnnotationType.OCR_BLOCK) and not layers.ocr:
                continue
            if t == AnnotationType.EXTRACTED_FIELD and not layers.fields:
                continue
            if t == AnnotationType.EXTRACTED_ENTITY and not layers.entities:
                continue
            if t in (AnnotationType.TABLE_BOUNDS, AnnotationType.TABLE_HEADER, AnnotationType.TABLE_ROW, AnnotationType.TABLE_CELL, AnnotationType.TABLE_LINE_ITEM) and not layers.tables:
                continue
            if t in (AnnotationType.VALIDATION_PASS, AnnotationType.VALIDATION_WARNING, AnnotationType.VALIDATION_ERROR, AnnotationType.VALIDATION_CRITICAL) and not layers.validation:
                continue
            if t in (AnnotationType.REVIEW_TARGET, AnnotationType.REVIEW_CRITICAL) and not layers.review:
                continue

            active.append(ann)

        return active, enabled_layer_names

    @staticmethod
    def _sort_annotations_by_priority(
        annotations: List[OverlayAnnotation],
    ) -> List[OverlayAnnotation]:
        """Sort annotations so that reviews and validation errors are drawn on top of ordinary text."""
        priority_map = {
            AnnotationType.OCR_WORD: 1,
            AnnotationType.OCR_LINE: 2,
            AnnotationType.OCR_BLOCK: 3,
            AnnotationType.EXTRACTED_ENTITY: 4,
            AnnotationType.TABLE_BOUNDS: 5,
            AnnotationType.TABLE_CELL: 6,
            AnnotationType.EXTRACTED_FIELD: 7,
            AnnotationType.VALIDATION_WARNING: 8,
            AnnotationType.TABLE_LINE_ITEM: 9,
            AnnotationType.VALIDATION_ERROR: 10,
            AnnotationType.REVIEW_TARGET: 11,
            AnnotationType.VALIDATION_CRITICAL: 12,
            AnnotationType.REVIEW_CRITICAL: 13,
        }
        return sorted(annotations, key=lambda a: priority_map.get(a.annotation_type, 5))

    @staticmethod
    def _apply_opacity(
        color: Tuple[int, int, int, int],
        opacity_factor: float,
    ) -> Tuple[int, int, int, int]:
        """Scale the alpha channel of a color tuple."""
        r, g, b, a = color
        new_a = int(max(0, min(255, a * opacity_factor)))
        return (r, g, b, new_a)

    @staticmethod
    def _draw_label_pill(
        draw: ImageDraw.ImageDraw,
        pixel_box: BoundingBox,
        label: str,
        border_color: Tuple[int, int, int, int],
        font: Any,
    ) -> None:
        """Draw a high-contrast pill badge with text label."""
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pill_pad_x = 4
        pill_pad_y = 2

        # Position label above bounding box if space permits, else inside top-left
        pill_x0 = pixel_box.xmin
        pill_y0 = pixel_box.ymin - text_h - (pill_pad_y * 2) - 2
        if pill_y0 < 0:
            pill_y0 = pixel_box.ymin + 2

        pill_x1 = pill_x0 + text_w + (pill_pad_x * 2)
        pill_y1 = pill_y0 + text_h + (pill_pad_y * 2)

        # Draw pill background
        draw.rectangle([pill_x0, pill_y0, pill_x1, pill_y1], fill=(255, 255, 255, 230), outline=border_color, width=1)
        # Draw label text
        draw.text((pill_x0 + pill_pad_x, pill_y0 + pill_pad_y), label, fill=(10, 10, 10, 255), font=font)


__all__ = ["OverlayRenderer"]
