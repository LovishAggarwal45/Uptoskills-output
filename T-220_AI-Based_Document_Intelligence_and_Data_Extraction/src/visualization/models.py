"""Domain data models and serialization structures for Visual Explainability & Evidence Overlays."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.models import BoundingBox, Provenance
from src.core.types import SeverityLevel, ValidationStatus
from src.confidence.models import ConfidenceBand, ReviewPriority, ReviewStatus, ReviewTargetType


class AnnotationType(str, Enum):
    """Categorization of visual evidence and overlay annotations."""
    # OCR Layers
    OCR_WORD = "ocr_word"
    OCR_LINE = "ocr_line"
    OCR_BLOCK = "ocr_block"

    # Field & Entity Layers
    EXTRACTED_FIELD = "extracted_field"
    EXTRACTED_ENTITY = "extracted_entity"

    # Table Intelligence Layers
    TABLE_BOUNDS = "table_bounds"
    TABLE_HEADER = "table_header"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    TABLE_LINE_ITEM = "table_line_item"

    # Validation Layers
    VALIDATION_PASS = "validation_pass"
    VALIDATION_WARNING = "validation_warning"
    VALIDATION_ERROR = "validation_error"
    VALIDATION_CRITICAL = "validation_critical"

    # Confidence Bands
    CONFIDENCE_HIGH = "confidence_high"
    CONFIDENCE_MEDIUM = "confidence_medium"
    CONFIDENCE_LOW = "confidence_low"
    CONFIDENCE_VERY_LOW = "confidence_very_low"

    # Review Routing Layers
    REVIEW_TARGET = "review_target"
    REVIEW_CRITICAL = "review_critical"
    UNLOCATED_ISSUE = "unlocated_issue"


class VisualCoordinateSystem(str, Enum):
    """Coordinate reference frame used for visual elements."""
    ABSOLUTE_PIXEL = "absolute_pixel"
    NORMALIZED_FRACTION = "normalized_fraction"
    PROCESSED_IMAGE_PIXEL = "processed_image_pixel"
    ORIGINAL_IMAGE_PIXEL = "original_image_pixel"


@dataclass
class EvidenceRegion:
    """Standardized granular evidence region linking extracted information to its source location."""
    evidence_id: str
    document_id: str
    page_number: int
    region_type: AnnotationType
    bbox: Optional[BoundingBox] = None
    source_text: Optional[str] = None
    normalized_value: Optional[Any] = None
    field_name: Optional[str] = None
    table_id: Optional[str] = None
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    ocr_confidence: Optional[float] = None
    extraction_confidence: Optional[float] = None
    aggregated_confidence: Optional[float] = None
    confidence_band: Optional[ConfidenceBand] = None
    validation_status: Optional[ValidationStatus] = None
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    source_method: Optional[str] = None
    provenance: Optional[Provenance] = None
    is_spatial: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"EvidenceRegion page_number {self.page_number} must be >= 1")
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"EvidenceRegion ocr_confidence {self.ocr_confidence} must be in [0.0, 1.0]")
        if self.extraction_confidence is not None and not (0.0 <= self.extraction_confidence <= 1.0):
            raise ValueError(f"EvidenceRegion extraction_confidence {self.extraction_confidence} must be in [0.0, 1.0]")
        if self.aggregated_confidence is not None and not (0.0 <= self.aggregated_confidence <= 1.0):
            raise ValueError(f"EvidenceRegion aggregated_confidence {self.aggregated_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize EvidenceRegion to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "region_type": self.region_type.value,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "source_text": self.source_text,
            "normalized_value": (
                self.normalized_value.isoformat()
                if isinstance(self.normalized_value, datetime)
                else self.normalized_value
            ),
            "field_name": self.field_name,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "column_name": self.column_name,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "extraction_confidence": round(self.extraction_confidence, 4) if self.extraction_confidence is not None else None,
            "aggregated_confidence": round(self.aggregated_confidence, 4) if self.aggregated_confidence is not None else None,
            "confidence_band": self.confidence_band.value if self.confidence_band else None,
            "validation_status": self.validation_status.value if self.validation_status else None,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "source_method": self.source_method,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "is_spatial": self.is_spatial,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceRegion:
        """Construct EvidenceRegion from dictionary."""
        bbox_data = data.get("bbox")
        prov_data = data.get("provenance")
        band_raw = data.get("confidence_band")
        val_status_raw = data.get("validation_status")

        return cls(
            evidence_id=data["evidence_id"],
            document_id=data["document_id"],
            page_number=int(data.get("page_number", 1)),
            region_type=AnnotationType(data["region_type"]),
            bbox=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            source_text=data.get("source_text"),
            normalized_value=data.get("normalized_value"),
            field_name=data.get("field_name"),
            table_id=data.get("table_id"),
            row_index=int(data["row_index"]) if data.get("row_index") is not None else None,
            column_name=data.get("column_name"),
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            extraction_confidence=float(data["extraction_confidence"]) if data.get("extraction_confidence") is not None else None,
            aggregated_confidence=float(data["aggregated_confidence"]) if data.get("aggregated_confidence") is not None else None,
            confidence_band=ConfidenceBand(band_raw) if band_raw else None,
            validation_status=ValidationStatus(val_status_raw) if val_status_raw else None,
            review_required=bool(data.get("review_required", False)),
            review_reasons=list(data.get("review_reasons", [])),
            source_method=data.get("source_method"),
            provenance=Provenance.from_dict(prov_data) if prov_data else None,
            is_spatial=bool(data.get("is_spatial", True)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OverlayAnnotation:
    """Visual annotation ready for rendering onto a document page image."""
    annotation_id: str
    page_number: int
    annotation_type: AnnotationType
    bounding_box: Optional[BoundingBox] = None
    label: str = ""
    tooltip: str = ""
    severity: Optional[SeverityLevel] = None
    confidence_band: Optional[ConfidenceBand] = None
    confidence_score: Optional[float] = None
    source_evidence_id: Optional[str] = None
    field_name: Optional[str] = None
    table_id: Optional[str] = None
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    review_reasons: List[str] = field(default_factory=list)
    color_rgba: Tuple[int, int, int, int] = (70, 130, 180, 255)  # Steel Blue default
    fill_rgba: Optional[Tuple[int, int, int, int]] = (70, 130, 180, 40)
    line_width: int = 2
    badge_text: Optional[str] = None
    is_visible: bool = True
    rendering_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize OverlayAnnotation to dictionary."""
        return {
            "annotation_id": self.annotation_id,
            "page_number": self.page_number,
            "annotation_type": self.annotation_type.value,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "label": self.label,
            "tooltip": self.tooltip,
            "severity": self.severity.value if self.severity else None,
            "confidence_band": self.confidence_band.value if self.confidence_band else None,
            "confidence_score": round(self.confidence_score, 4) if self.confidence_score is not None else None,
            "source_evidence_id": self.source_evidence_id,
            "field_name": self.field_name,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "column_name": self.column_name,
            "review_reasons": self.review_reasons,
            "color_rgba": list(self.color_rgba),
            "fill_rgba": list(self.fill_rgba) if self.fill_rgba else None,
            "line_width": self.line_width,
            "badge_text": self.badge_text,
            "is_visible": self.is_visible,
            "rendering_metadata": self.rendering_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OverlayAnnotation:
        """Construct OverlayAnnotation from dictionary."""
        bbox_data = data.get("bounding_box")
        sev_raw = data.get("severity")
        band_raw = data.get("confidence_band")
        color_data = data.get("color_rgba", [70, 130, 180, 255])
        fill_data = data.get("fill_rgba")

        return cls(
            annotation_id=data["annotation_id"],
            page_number=int(data.get("page_number", 1)),
            annotation_type=AnnotationType(data["annotation_type"]),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            label=data.get("label", ""),
            tooltip=data.get("tooltip", ""),
            severity=SeverityLevel(sev_raw) if sev_raw else None,
            confidence_band=ConfidenceBand(band_raw) if band_raw else None,
            confidence_score=float(data["confidence_score"]) if data.get("confidence_score") is not None else None,
            source_evidence_id=data.get("source_evidence_id"),
            field_name=data.get("field_name"),
            table_id=data.get("table_id"),
            row_index=int(data["row_index"]) if data.get("row_index") is not None else None,
            column_name=data.get("column_name"),
            review_reasons=list(data.get("review_reasons", [])),
            color_rgba=tuple(color_data),  # type: ignore
            fill_rgba=tuple(fill_data) if fill_data else None,  # type: ignore
            line_width=int(data.get("line_width", 2)),
            badge_text=data.get("badge_text"),
            is_visible=bool(data.get("is_visible", True)),
            rendering_metadata=data.get("rendering_metadata", {}),
        )


@dataclass
class PageVisualization:
    """Complete visualization record and artifact mapping for a single document page."""
    document_id: str
    page_number: int
    source_image_path: Optional[Path] = None
    image_width: int = 0
    image_height: int = 0
    coordinate_system: VisualCoordinateSystem = VisualCoordinateSystem.ABSOLUTE_PIXEL
    evidence_regions: List[EvidenceRegion] = field(default_factory=list)
    annotations: List[OverlayAnnotation] = field(default_factory=list)
    overlay_image_path: Optional[Path] = None
    rendered_layers: List[str] = field(default_factory=list)
    rendering_status: str = "pending"
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PageVisualization to dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "source_image_path": str(self.source_image_path) if self.source_image_path else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_system": self.coordinate_system.value,
            "evidence_region_count": len(self.evidence_regions),
            "annotation_count": len(self.annotations),
            "evidence_regions": [er.to_dict() for er in self.evidence_regions],
            "annotations": [an.to_dict() for an in self.annotations],
            "overlay_image_path": str(self.overlay_image_path) if self.overlay_image_path else None,
            "rendered_layers": self.rendered_layers,
            "rendering_status": self.rendering_status,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageVisualization:
        """Construct PageVisualization from dictionary."""
        src_path = data.get("source_image_path")
        overlay_path = data.get("overlay_image_path")
        coord_sys_raw = data.get("coordinate_system", VisualCoordinateSystem.ABSOLUTE_PIXEL.value)

        return cls(
            document_id=data["document_id"],
            page_number=int(data.get("page_number", 1)),
            source_image_path=Path(src_path) if src_path else None,
            image_width=int(data.get("image_width", 0)),
            image_height=int(data.get("image_height", 0)),
            coordinate_system=VisualCoordinateSystem(coord_sys_raw),
            evidence_regions=[EvidenceRegion.from_dict(er) for er in data.get("evidence_regions", [])],
            annotations=[OverlayAnnotation.from_dict(an) for an in data.get("annotations", [])],
            overlay_image_path=Path(overlay_path) if overlay_path else None,
            rendered_layers=list(data.get("rendered_layers", [])),
            rendering_status=data.get("rendering_status", "pending"),
            warnings=list(data.get("warnings", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisualizationManifest:
    """Document-wide structured manifest cataloging all spatial and non-spatial evidence."""
    document_id: str
    total_pages: int
    evidence_regions: List[EvidenceRegion] = field(default_factory=list)
    unlocated_evidence: List[EvidenceRegion] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize VisualizationManifest to dictionary."""
        return {
            "document_id": self.document_id,
            "total_pages": self.total_pages,
            "total_spatial_evidence": len(self.evidence_regions),
            "total_unlocated_evidence": len(self.unlocated_evidence),
            "created_at": self.created_at.isoformat(),
            "evidence_regions": [er.to_dict() for er in self.evidence_regions],
            "unlocated_evidence": [ue.to_dict() for ue in self.unlocated_evidence],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize VisualizationManifest to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisualizationManifest:
        """Construct VisualizationManifest from dictionary."""
        ts_str = data.get("created_at")
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
        return cls(
            document_id=data["document_id"],
            total_pages=int(data.get("total_pages", 1)),
            evidence_regions=[EvidenceRegion.from_dict(er) for er in data.get("evidence_regions", [])],
            unlocated_evidence=[EvidenceRegion.from_dict(ue) for ue in data.get("unlocated_evidence", [])],
            created_at=ts,
            metadata=data.get("metadata", {}),
        )


@dataclass
class VisualizationResult:
    """Unified result aggregate containing all page overlays, manifests, and artifact paths."""
    document_id: str
    page_visualizations: List[PageVisualization] = field(default_factory=list)
    manifest: Optional[VisualizationManifest] = None
    artifact_paths: Dict[str, Path] = field(default_factory=dict)
    annotation_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    configuration_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        """Total pages visualized."""
        return len(self.page_visualizations)

    @property
    def has_errors(self) -> bool:
        """True if any page failed rendering."""
        return len(self.errors) > 0 or any(p.rendering_status == "failed" for p in self.page_visualizations)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize VisualizationResult to dictionary."""
        return {
            "document_id": self.document_id,
            "total_pages": self.total_pages,
            "page_visualizations": [pv.to_dict() for pv in self.page_visualizations],
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "artifact_paths": {k: str(v) for k, v in self.artifact_paths.items()},
            "annotation_counts": self.annotation_counts,
            "warnings": self.warnings,
            "errors": self.errors,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "configuration_summary": self.configuration_summary,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize VisualizationResult to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisualizationResult:
        """Construct VisualizationResult from dictionary."""
        manifest_data = data.get("manifest")
        art_paths = {k: Path(v) for k, v in data.get("artifact_paths", {}).items()}

        return cls(
            document_id=data["document_id"],
            page_visualizations=[PageVisualization.from_dict(pv) for pv in data.get("page_visualizations", [])],
            manifest=VisualizationManifest.from_dict(manifest_data) if manifest_data else None,
            artifact_paths=art_paths,
            annotation_counts=dict(data.get("annotation_counts", {})),
            warnings=list(data.get("warnings", [])),
            errors=list(data.get("errors", [])),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
            configuration_summary=dict(data.get("configuration_summary", {})),
            metadata=dict(data.get("metadata", {})),
        )


__all__ = [
    "AnnotationType",
    "VisualCoordinateSystem",
    "EvidenceRegion",
    "OverlayAnnotation",
    "PageVisualization",
    "VisualizationManifest",
    "VisualizationResult",
]
