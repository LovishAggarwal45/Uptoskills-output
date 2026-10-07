"""Unit tests for validation score calculation and explainable confidence metrics."""

import unittest

from src.core.types import SeverityLevel, ValidationStatus
from src.validation.confidence import ValidationConfidenceCalculator
from src.validation.models import ValidationCategory, ValidationIssue


class TestValidationConfidence(unittest.TestCase):
    """Test suite for ValidationConfidenceCalculator."""

    def setUp(self) -> None:
        self.calc = ValidationConfidenceCalculator()

    def test_empty_issues_returns_default_one(self) -> None:
        res = self.calc.calculate_score([])
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["rules_evaluated"], 0)

    def test_all_passed_returns_perfect_score(self) -> None:
        issues = [
            ValidationIssue(
                rule_id=f"VAL_TEST_{i}",
                rule_name=f"Test Rule {i}",
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message="OK",
            )
            for i in range(5)
        ]
        res = self.calc.calculate_score(issues)
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["rules_passed"], 5)
        self.assertEqual(res["rules_failed"], 0)

    def test_errors_apply_weighted_penalties(self) -> None:
        issues = [
            ValidationIssue(rule_id="VAL_1", rule_name="R1", status=ValidationStatus.VALID, severity=SeverityLevel.INFO, message="OK"),
            ValidationIssue(rule_id="VAL_2", rule_name="R2", status=ValidationStatus.VALID, severity=SeverityLevel.INFO, message="OK"),
            ValidationIssue(rule_id="VAL_3", rule_name="R3", status=ValidationStatus.INVALID, severity=SeverityLevel.ERROR, message="Err"),
            ValidationIssue(rule_id="VAL_4", rule_name="R4", status=ValidationStatus.WARNING, severity=SeverityLevel.WARNING, message="Warn"),
        ]
        res = self.calc.calculate_score(issues)
        self.assertEqual(res["rules_evaluated"], 4)
        self.assertEqual(res["rules_passed"], 2)
        self.assertEqual(res["rules_failed"], 1)
        self.assertEqual(res["rules_warning"], 1)
        self.assertLess(res["score"], 0.50)
        self.assertIn("Total penalties", res["explanation"])


if __name__ == "__main__":
    unittest.main()
