"""Validation score calculation and explainable consistency metrics.

Note:
Validation score measures rule compliance and mathematical/logical consistency.
It is strictly distinct from OCR confidence (character recognition probability),
extraction confidence (spatial/pattern matching certainty), and table confidence.
"""

from typing import Any, Dict, List

from src.core.types import SeverityLevel, ValidationStatus
from src.validation.models import ValidationIssue


class ValidationConfidenceCalculator:
    """Computes an explainable, deterministic validation consistency score in [0.0, 1.0]."""

    def __init__(
        self,
        critical_penalty: float = 0.35,
        error_penalty: float = 0.15,
        warning_penalty: float = 0.05,
    ) -> None:
        self.critical_penalty = critical_penalty
        self.error_penalty = error_penalty
        self.warning_penalty = warning_penalty

    def calculate_score(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Calculate deterministic validation score and explainable factor breakdown.

        Args:
            issues: All evaluated validation issues.

        Returns:
            Dictionary with final score and detailed penalty breakdown.
        """
        if not issues:
            return {
                "score": 1.0,
                "base_score": 1.0,
                "rules_evaluated": 0,
                "rules_passed": 0,
                "rules_failed": 0,
                "rules_warning": 0,
                "critical_penalties": 0.0,
                "error_penalties": 0.0,
                "warning_penalties": 0.0,
                "explanation": "No validation rules evaluated; default score 1.0.",
            }

        total_rules = len(issues)
        passed_rules = sum(1 for i in issues if i.status == ValidationStatus.VALID)
        failed_rules = sum(1 for i in issues if i.status == ValidationStatus.INVALID)
        warning_rules = sum(1 for i in issues if i.status == ValidationStatus.WARNING)

        base_ratio = passed_rules / total_rules

        # Calculate severity-weighted penalties
        crit_count = sum(1 for i in issues if i.severity == SeverityLevel.CRITICAL and i.status == ValidationStatus.INVALID)
        err_count = sum(1 for i in issues if i.severity == SeverityLevel.ERROR and i.status == ValidationStatus.INVALID)
        warn_count = sum(1 for i in issues if i.severity == SeverityLevel.WARNING or i.status == ValidationStatus.WARNING)

        crit_deduction = crit_count * self.critical_penalty
        err_deduction = err_count * self.error_penalty
        warn_deduction = warn_count * self.warning_penalty
        total_deductions = crit_deduction + err_deduction + warn_deduction

        raw_score = base_ratio - total_deductions
        final_score = max(0.0, min(1.0, raw_score))

        explanation = (
            f"Evaluated {total_rules} rules: {passed_rules} passed ({base_ratio:.1%}), "
            f"{failed_rules} failed ({err_count} errors, {crit_count} critical), "
            f"{warning_rules} warnings. Total penalties: {total_deductions:.2f}."
        )

        return {
            "score": round(final_score, 4),
            "base_score": round(base_ratio, 4),
            "rules_evaluated": total_rules,
            "rules_passed": passed_rules,
            "rules_failed": failed_rules,
            "rules_warning": warning_rules,
            "critical_penalties": round(crit_deduction, 4),
            "error_penalties": round(err_deduction, 4),
            "warning_penalties": round(warn_deduction, 4),
            "explanation": explanation,
        }


__all__ = ["ValidationConfidenceCalculator"]
