"""Image preprocessing contract and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.config import PreprocessingConfig
from src.core.models import DocumentPage


@dataclass
class PreprocessingResult:
    """Outcome of applying modular image transformations to a document page."""
    page_number: int
    processed_image_path: Optional[Union[str, Path]] = None
    applied_operations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    deskew_angle: Optional[float] = None
    is_modified: bool = False


class BaseImagePreprocessor(ABC):
    """Abstract interface for modular page image enhancement and normalization."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    @abstractmethod
    def preprocess_page(
        self,
        page: DocumentPage,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PreprocessingResult:
        """Apply configured image operations (grayscale, deskew, denoise, contrast) to a page.

        Args:
            page: DocumentPage instance with an associated source image.
            output_dir: Optional directory to store preprocessed intermediate images.

        Returns:
            PreprocessingResult containing the processed image location and telemetry.
        """
        pass
