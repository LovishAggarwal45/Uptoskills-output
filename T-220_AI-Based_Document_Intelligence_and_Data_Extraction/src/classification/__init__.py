"""DocuMind AI Document Classification Subsystem."""

from src.classification.base import BaseDocumentClassifier
from src.classification.classifier import RuleBasedDocumentClassifier
from src.classification.exceptions import (
    ClassificationConfigurationError,
    ClassificationError,
    InvalidClassificationInput,
)
from src.classification.models import (
    ClassificationCandidate,
    ClassificationEvidence,
    ClassificationResult,
    SignalType,
)
from src.classification.rules import (
    RuleDefinition,
    get_default_classification_rules,
    locate_text_provenance,
)

__all__ = [
    "BaseDocumentClassifier",
    "RuleBasedDocumentClassifier",
    "ClassificationResult",
    "ClassificationEvidence",
    "ClassificationCandidate",
    "SignalType",
    "RuleDefinition",
    "get_default_classification_rules",
    "locate_text_provenance",
    "ClassificationError",
    "InvalidClassificationInput",
    "ClassificationConfigurationError",
]
