"""Domain exceptions for the Visual Explainability & Evidence Overlays subsystem."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.exceptions import DocuMindError


class VisualizationError(DocuMindError):
    """Base exception for all visual evidence and overlay failures."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details)


class VisualizationConfigurationError(VisualizationError):
    """Raised when visualization settings or layer definitions are invalid."""
    pass


class CoordinateMappingError(VisualizationError):
    """Raised when converting between pixel, normalized, or resized coordinate systems fails."""
    pass


class InvalidBoundingBoxError(CoordinateMappingError):
    """Raised when bounding box geometry is inverted, non-finite, or malformed."""
    pass


class SourcePageUnavailableError(VisualizationError):
    """Raised when the source page image file cannot be located or opened for rendering."""
    pass


class OverlayRenderingError(VisualizationError):
    """Raised when drawing or saving visual overlay artifacts fails."""
    pass


class EvidenceMappingError(VisualizationError):
    """Raised when assembling or indexing evidence provenance fails."""
    pass


__all__ = [
    "VisualizationError",
    "VisualizationConfigurationError",
    "CoordinateMappingError",
    "InvalidBoundingBoxError",
    "SourcePageUnavailableError",
    "OverlayRenderingError",
    "EvidenceMappingError",
]
