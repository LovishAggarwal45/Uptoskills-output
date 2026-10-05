"""Abstract base contracts and master visualizer coordinator for visual explainability."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.config import VisualizationConfig
from src.core.logging import get_logger
from src.core.models import Document, ExtractedField
from src.confidence.models import ReviewRoutingResult
from src.extraction.models import ExtractedEntity
from src.ocr.models import DocumentOCRResult
from src.tables.models import Table
from src.validation.models import ValidationReport
from src.visualization.evidence_mapper import EvidenceMapper
from src.visualization.models import (
    OverlayAnnotation,
    PageVisualization,
    VisualizationManifest,
    VisualizationResult,
)
from src.visualization.overlay_renderer import OverlayRenderer
from src.visualization.serializer import VisualizationSerializer

logger = get_logger("visualization.coordinator")


class BaseEvidenceMapper(ABC):
    """Abstract contract for extracting and assembling evidence from pipeline artifacts."""

    @abstractmethod
    def map_all(
        self,
        document: Document,
        fields: Optional[Dict[str, ExtractedField]] = None,
        entities: Optional[List[ExtractedEntity]] = None,
        tables: Optional[List[Table]] = None,
        validation_report: Optional[ValidationReport] = None,
        routing_result: Optional[ReviewRoutingResult] = None,
        ocr_result: Optional[DocumentOCRResult] = None,
    ) -> Tuple[VisualizationManifest, Dict[int, List[OverlayAnnotation]]]:
        """Extract evidence and compile annotations."""
        pass


class BaseOverlayRenderer(ABC):
    """Abstract contract for rendering visual annotations on document page images."""

    @abstractmethod
    def render_page(
        self,
        document_id: str,
        page: Any,
        annotations: List[OverlayAnnotation],
        output_dir: Union[str, Path],
    ) -> PageVisualization:
        """Render annotations on page image."""
        pass


class BaseVisualizer(ABC):
    """High-level abstract interface for document visualization."""

    @abstractmethod
    def visualize_document(
        self,
        document: Document,
        fields: Optional[Dict[str, ExtractedField]] = None,
        entities: Optional[List[ExtractedEntity]] = None,
        tables: Optional[List[Table]] = None,
        validation_report: Optional[ValidationReport] = None,
        routing_result: Optional[ReviewRoutingResult] = None,
        ocr_result: Optional[DocumentOCRResult] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> VisualizationResult:
        """Execute end-to-end evidence mapping and visual overlay generation."""
        pass


class DocumentVisualizer(BaseVisualizer):
    """Master coordinator orchestrating evidence mapping, overlay rendering, and artifact persistence."""

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or VisualizationConfig()
        self.renderer = OverlayRenderer(config=self.config)

    def visualize_document(
        self,
        document: Document,
        fields: Optional[Dict[str, ExtractedField]] = None,
        entities: Optional[List[ExtractedEntity]] = None,
        tables: Optional[List[Table]] = None,
        validation_report: Optional[ValidationReport] = None,
        routing_result: Optional[ReviewRoutingResult] = None,
        ocr_result: Optional[DocumentOCRResult] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> VisualizationResult:
        """Run complete evidence extraction, overlay rendering, and JSON manifest generation.

        Args:
            document: Processed Document aggregate.
            fields: Optional map of extracted fields from Phase 5.
            entities: Optional list of extracted entities.
            tables: Optional list of tables from Phase 6.
            validation_report: Optional validation report from Phase 7.
            routing_result: Optional confidence scoring & review routing result from Phase 8.
            ocr_result: Optional document OCR result from Phase 3.
            output_dir: Destination folder for overlays and manifests (defaults to config).

        Returns:
            VisualizationResult containing all page overlay paths and manifest.
        """
        start_time = time.perf_counter()
        target_output_dir = Path(output_dir or self.config.output.directory)
        logger.info(f"Initiating visual explainability for document '{document.id}' ({len(document.pages)} pages)")

        # 1. Map Evidence and Annotations
        mapper = EvidenceMapper(document_id=document.id)
        manifest, annotations_by_page = mapper.map_all(
            document=document,
            fields=fields,
            entities=entities,
            tables=tables,
            validation_report=validation_report,
            routing_result=routing_result,
            ocr_result=ocr_result,
        )

        # 2. Render Page Overlays
        page_visualizations: List[PageVisualization] = []
        artifact_paths: Dict[str, Path] = {}
        annotation_counts: Dict[str, int] = {}
        warnings: List[str] = []
        errors: List[str] = []

        for page in sorted(document.pages, key=lambda p: p.page_number):
            anns = annotations_by_page.get(page.page_number, [])
            annotation_counts[f"page_{page.page_number}"] = len(anns)

            try:
                page_vis = self.renderer.render_page(
                    document_id=document.id,
                    page=page,
                    annotations=anns,
                    output_dir=target_output_dir,
                )
                page_visualizations.append(page_vis)
                if page_vis.overlay_image_path:
                    artifact_paths[f"page_{page.page_number}_overlay"] = page_vis.overlay_image_path
            except Exception as e:
                err_msg = f"Failed to render overlay for page {page.page_number}: {e}"
                logger.error(err_msg, exc_info=True)
                errors.append(err_msg)

                failed_vis = PageVisualization(
                    document_id=document.id,
                    page_number=page.page_number,
                    source_image_path=Path(page.image_path) if page.image_path else None,
                    rendering_status="failed",
                    warnings=[str(e)],
                )
                page_visualizations.append(failed_vis)

        # 3. Save Evidence Manifest & Summary JSON
        if self.config.output.save_manifest:
            try:
                manifest_path = VisualizationSerializer.save_manifest(
                    manifest=manifest,
                    output_dir=target_output_dir,
                )
                artifact_paths["evidence_manifest"] = manifest_path
            except Exception as e:
                logger.warning(f"Could not save evidence manifest: {e}")
                warnings.append(f"Manifest save error: {e}")

        elapsed = time.perf_counter() - start_time

        result = VisualizationResult(
            document_id=document.id,
            page_visualizations=page_visualizations,
            manifest=manifest,
            artifact_paths=artifact_paths,
            annotation_counts=annotation_counts,
            warnings=warnings,
            errors=errors,
            execution_time_seconds=round(elapsed, 4),
            configuration_summary={
                "enabled_layers": [k for k, v in self.config.layers.__dict__.items() if v],
                "format": self.config.output.format,
                "opacity": self.config.rendering.opacity,
            },
            metadata={
                "total_spatial_evidence": len(manifest.evidence_regions),
                "total_unlocated_evidence": len(manifest.unlocated_evidence),
            },
        )

        if self.config.output.save_summary:
            try:
                summary_path = VisualizationSerializer.save_summary(
                    result=result,
                    output_dir=target_output_dir,
                )
                artifact_paths["visualization_summary"] = summary_path
            except Exception as e:
                logger.warning(f"Could not save visualization summary: {e}")

        logger.info(
            f"Visual explainability complete for '{document.id}': {len(page_visualizations)} pages, "
            f"{len(artifact_paths)} artifacts in {elapsed:.3f}s"
        )

        return result


__all__ = [
    "BaseEvidenceMapper",
    "BaseOverlayRenderer",
    "BaseVisualizer",
    "DocumentVisualizer",
]
