"""Document classification abstraction interface."""

from abc import ABC, abstractmethod
from typing import Optional

from src.classification.models import ClassificationResult
from src.core.config import ClassificationConfig
from src.core.models import Document


class BaseDocumentClassifier(ABC):
    """Abstract interface for classifying document category."""

    def __init__(self, config: Optional[ClassificationConfig] = None) -> None:
        self.config = config or ClassificationConfig()

    @abstractmethod
    def classify(self, document: Document) -> ClassificationResult:
        """Classify document category based on ingested text, layout, and structure.

        Args:
            document: Document aggregate with OCR results and pages populated.

        Returns:
            ClassificationResult containing the classified type, confidence, evidence, and scores.
        """
        pass
