"""DocuMind AI — Visual Explainability & Evidence Overlays Subsystem."""

from src.visualization.base import (
    BaseEvidenceMapper,
    BaseOverlayRenderer,
    BaseVisualizer,
    DocumentVisualizer,
)
from src.visualization.coordinates import CoordinateTransformer
from src.visualization.evidence_mapper import EvidenceMapper
from src.visualization.exceptions import (
    CoordinateMappingError,
    EvidenceMappingError,
    InvalidBoundingBoxError,
    OverlayRenderingError,
    SourcePageUnavailableError,
    VisualizationConfigurationError,
    VisualizationError,
)
from src.visualization.field_overlay import FieldOverlayBuilder
from src.visualization.legend import (
    BADGE_SYMBOLS,
    PALETTE,
    LegendRenderer,
    get_color_for_annotation,
)
from src.visualization.models import (
    AnnotationType,
    EvidenceRegion,
    OverlayAnnotation,
    PageVisualization,
    VisualCoordinateSystem,
    VisualizationManifest,
    VisualizationResult,
)
from src.visualization.overlay_renderer import OverlayRenderer
from src.visualization.review_overlay import ReviewOverlayBuilder
from src.visualization.serializer import VisualizationSerializer
from src.visualization.table_overlay import TableOverlayBuilder

__all__ = [
    # Base and Coordinator
    "BaseEvidenceMapper",
    "BaseOverlayRenderer",
    "BaseVisualizer",
    "DocumentVisualizer",
    # Coordinates and Mapper
    "CoordinateTransformer",
    "EvidenceMapper",
    # Exceptions
    "VisualizationError",
    "VisualizationConfigurationError",
    "CoordinateMappingError",
    "InvalidBoundingBoxError",
    "SourcePageUnavailableError",
    "OverlayRenderingError",
    "EvidenceMappingError",
    # Overlay Builders
    "FieldOverlayBuilder",
    "TableOverlayBuilder",
    "ReviewOverlayBuilder",
    # Legend
    "LegendRenderer",
    "PALETTE",
    "BADGE_SYMBOLS",
    "get_color_for_annotation",
    # Models
    "AnnotationType",
    "VisualCoordinateSystem",
    "EvidenceRegion",
    "OverlayAnnotation",
    "PageVisualization",
    "VisualizationManifest",
    "VisualizationResult",
    # Renderer and Serializer
    "OverlayRenderer",
    "VisualizationSerializer",
]
