"""Deterministic review priority evaluation and urgency calculation."""

from __future__ import annotations

from typing import List, Optional

from src.core.config import ReviewPriorityConfig
from src.core.types import SeverityLevel
from src.confidence.models import DocumentConfidence, ReviewIssue, ReviewPriority


class PriorityEvaluator:
    """Deterministically computes operational queue priority based on issue severities and risk signals."""

    def __init__(self, config: Optional[ReviewPriorityConfig] = None) -> None:
        self.config = config or ReviewPriorityConfig()

    def evaluate(
        self,
        issues: List[ReviewIssue],
        document_confidence: DocumentConfidence,
    ) -> ReviewPriority:
        """Determine human review priority.

        Deterministic Rules:
        1. URGENT:
           - Any issue with CRITICAL severity (e.g. arithmetic discrepancy, corrupted totals).
           - Table arithmetic or grand total validation failure.
        2. HIGH:
           - Any issue with HIGH/ERROR severity.
           - Mandatory required field missing.
           - Document overall confidence < 0.60.
           - 3 or more fields flagged for review.
        3. MEDIUM:
           - Issues with WARNING/MEDIUM severity.
           - Conflicting extraction candidates.
           - Document overall confidence < 0.70.
        4. LOW:
           - Minor warnings or single borderline field confidence dipping slightly below threshold.
        """
        if not issues and not document_confidence.review_required:
            return ReviewPriority.LOW

        # 1. Check for Critical Severity or Arithmetic Mismatches -> URGENT
        for issue in issues:
            if issue.severity == SeverityLevel.CRITICAL:
                return ReviewPriority.URGENT
            if "arithmetic" in issue.source_signal or "table_validation" in issue.source_signal:
                return ReviewPriority.URGENT

        # 2. Check for High Severity, Missing Required Fields, or Low Overall Score -> HIGH
        has_high_severity = any(
            i.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL) or
            i.source_signal == "required_field_missing"
            for i in issues
        )
        if has_high_severity:
            return ReviewPriority.HIGH

        if document_confidence.overall_confidence < 0.60:
            return ReviewPriority.HIGH

        field_review_count = sum(1 for fc in document_confidence.field_confidences.values() if fc.review_required)
        if field_review_count >= 3:
            return ReviewPriority.HIGH

        # 3. Check for Medium Severity or Moderate Score -> MEDIUM
        has_medium_severity = any(
            i.severity == SeverityLevel.WARNING or
            i.source_signal in ("conflict_count", "ocr_confidence", "aggregated_confidence")
            for i in issues
        )
        if has_medium_severity:
            return ReviewPriority.MEDIUM

        if document_confidence.overall_confidence < 0.70:
            return ReviewPriority.MEDIUM

        # 4. Fallback -> LOW
        return ReviewPriority.LOW
