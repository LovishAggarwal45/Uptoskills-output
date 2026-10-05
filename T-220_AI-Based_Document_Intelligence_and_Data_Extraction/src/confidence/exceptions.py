"""Confidence scoring and review routing subsystem exceptions."""

from src.core.exceptions import DocuMindError


class ConfidenceError(DocuMindError):
    """Base exception for all confidence calculation and review routing errors."""
    pass


class ConfidenceConfigurationError(ConfidenceError):
    """Raised when confidence weights, thresholds, or priorities are invalid or malformed."""
    pass


class ConfidenceAggregationError(ConfidenceError):
    """Raised when numerical aggregation or signal synthesis encounters an unrecoverable state."""
    pass


class ReviewRoutingError(ConfidenceError):
    """Raised when review rules evaluation or review queue dispatch fails."""
    pass
