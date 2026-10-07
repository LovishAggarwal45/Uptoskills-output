"""Unit tests for priority evaluator and deterministic urgency calculation."""

import unittest

from src.core.config import ReviewPriorityConfig
from src.core.types import SeverityLevel
from src.confidence.models import (
    DocumentConfidence,
    FieldConfidence,
    ReviewIssue,
    ReviewPriority,
    ReviewTarget,
    ReviewTargetType,
)
from src.confidence.priority import PriorityEvaluator


class TestPriority(unittest.TestCase):
    """Test suite for deterministic review priority evaluation."""

    def setUp(self) -> None:
        self.evaluator = PriorityEvaluator(config=ReviewPriorityConfig())

    def test_critical_severity_assigns_urgent(self) -> None:
        """Verify any CRITICAL severity issue assigns URGENT priority."""
        target = ReviewTarget(target_type=ReviewTargetType.DOCUMENT, reason="Math failure", severity=SeverityLevel.CRITICAL)
        issue = ReviewIssue(
            issue_id="ISSUE_MATH",
            rule_name="Math Check",
            severity=SeverityLevel.CRITICAL,
            message="Total arithmetic mismatch",
            affected_targets=[target],
            source_signal="line_item_arithmetic",
        )

        doc_conf = DocumentConfidence(document_id="doc_1", overall_confidence=0.85, review_required=True)
        priority = self.evaluator.evaluate([issue], doc_conf)
        self.assertEqual(priority, ReviewPriority.URGENT)

    def test_missing_required_field_assigns_high(self) -> None:
        """Verify missing required field assigns HIGH priority."""
        target = ReviewTarget(target_type=ReviewTargetType.FIELD, field_name="total", reason="Missing", severity=SeverityLevel.ERROR)
        issue = ReviewIssue(
            issue_id="ISSUE_REQ",
            rule_name="Required Check",
            severity=SeverityLevel.ERROR,
            message="Required field 'total' is missing",
            affected_targets=[target],
            source_signal="required_field_missing",
        )

        doc_conf = DocumentConfidence(document_id="doc_1", overall_confidence=0.75, review_required=True)
        priority = self.evaluator.evaluate([issue], doc_conf)
        self.assertEqual(priority, ReviewPriority.HIGH)

    def test_low_document_confidence_assigns_high(self) -> None:
        """Verify overall document score < 0.60 assigns HIGH priority."""
        doc_conf = DocumentConfidence(document_id="doc_1", overall_confidence=0.52, review_required=True)
        priority = self.evaluator.evaluate([], doc_conf)
        self.assertEqual(priority, ReviewPriority.HIGH)

    def test_moderate_warnings_assigns_medium(self) -> None:
        """Verify moderate confidence and candidate conflicts assign MEDIUM priority."""
        target = ReviewTarget(target_type=ReviewTargetType.FIELD, field_name="date", reason="Conflict", severity=SeverityLevel.WARNING)
        issue = ReviewIssue(
            issue_id="ISSUE_CONF",
            rule_name="Conflict Check",
            severity=SeverityLevel.WARNING,
            message="2 candidates found",
            affected_targets=[target],
            source_signal="conflict_count",
        )

        doc_conf = DocumentConfidence(document_id="doc_1", overall_confidence=0.78, review_required=True)
        priority = self.evaluator.evaluate([issue], doc_conf)
        self.assertEqual(priority, ReviewPriority.MEDIUM)

    def test_clean_document_assigns_low(self) -> None:
        """Verify document with no review required assigns LOW priority."""
        doc_conf = DocumentConfidence(document_id="doc_1", overall_confidence=0.95, review_required=False)
        priority = self.evaluator.evaluate([], doc_conf)
        self.assertEqual(priority, ReviewPriority.LOW)


if __name__ == "__main__":
    unittest.main()
