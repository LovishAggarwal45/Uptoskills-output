"""DocuMind AI Multi-Tier Confidence Scoring & Human Review Routing Subsystem."""

from src.confidence.aggregator import (
    aggregate_weighted_signals,
    compute_confidence_band,
    map_validation_status_to_signal,
)
from src.confidence.base import (
    BaseConfidenceEngine,
    ConfidenceEngine,
    ConfidenceEvaluation,
)
from src.confidence.document_confidence import DocumentConfidenceCalculator
from src.confidence.exceptions import (
    ConfidenceAggregationError,
    ConfidenceConfigurationError,
    ConfidenceError,
    ReviewRoutingError,
)
from src.confidence.field_confidence import FieldConfidenceCalculator
from src.confidence.models import (
    ConfidenceBand,
    ConfidenceSignals,
    DocumentConfidence,
    FieldConfidence,
    LineItemConfidence,
    ReviewIssue,
    ReviewPriority,
    ReviewQueueItem,
    ReviewReason,
    ReviewRoutingResult,
    ReviewStatus,
    ReviewTarget,
    ReviewTargetType,
    TableConfidence,
)
from src.confidence.priority import PriorityEvaluator
from src.confidence.review_router import ReviewRouter
from src.confidence.review_rules import (
    BaseReviewRule,
    CandidateConflictReviewRule,
    LowFieldConfidenceReviewRule,
    LowOCRConfidenceReviewRule,
    MissingRequiredFieldReviewRule,
    ReviewRuleCatalog,
    TableArithmeticReviewRule,
    UnclassifiedDocumentReviewRule,
    ValidationReportReviewRule,
)
from src.confidence.signals import (
    extract_document_signals,
    extract_field_signals,
    extract_table_signals,
)
from src.confidence.table_confidence import TableConfidenceCalculator

__all__ = [
    # Base and Orchestrators
    "BaseConfidenceEngine",
    "ConfidenceEngine",
    "ConfidenceEvaluation",
    "ReviewRouter",
    "FieldConfidenceCalculator",
    "TableConfidenceCalculator",
    "DocumentConfidenceCalculator",
    "PriorityEvaluator",
    "ReviewRuleCatalog",
    # Models and Enums
    "ConfidenceBand",
    "ConfidenceSignals",
    "FieldConfidence",
    "LineItemConfidence",
    "TableConfidence",
    "DocumentConfidence",
    "ReviewPriority",
    "ReviewStatus",
    "ReviewTargetType",
    "ReviewTarget",
    "ReviewIssue",
    "ReviewReason",
    "ReviewQueueItem",
    "ReviewRoutingResult",
    # Rules
    "BaseReviewRule",
    "LowFieldConfidenceReviewRule",
    "LowOCRConfidenceReviewRule",
    "MissingRequiredFieldReviewRule",
    "CandidateConflictReviewRule",
    "TableArithmeticReviewRule",
    "ValidationReportReviewRule",
    "UnclassifiedDocumentReviewRule",
    # Signal Extractors & Helpers
    "extract_field_signals",
    "extract_table_signals",
    "extract_document_signals",
    "aggregate_weighted_signals",
    "compute_confidence_band",
    "map_validation_status_to_signal",
    # Exceptions
    "ConfidenceError",
    "ConfidenceConfigurationError",
    "ConfidenceAggregationError",
    "ReviewRoutingError",
]
