"""Unit tests for candidate conflict detection and ambiguity identification."""

import unittest

from src.core.models import ExtractedField
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.validation.conflict_detector import CandidateConflictDetector
from src.validation.models import ValidationCategory


class TestConflictDetection(unittest.TestCase):
    """Test suite for CandidateConflictDetector."""

    def setUp(self) -> None:
        self.detector = CandidateConflictDetector()

    def test_single_candidate_no_conflict(self) -> None:
        field = ExtractedField(
            name="invoice_number",
            value="INV-100",
            field_type=FieldType.IDENTIFIER,
            candidates=[{"value": "INV-100", "confidence": 0.95}],
        )
        issues = self.detector.evaluate({"invoice_number": field}, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 0)

    def test_repeated_identical_candidates_valid(self) -> None:
        field = ExtractedField(
            name="invoice_number",
            value="INV-100",
            field_type=FieldType.IDENTIFIER,
            candidates=[
                {"value": "INV-100", "confidence": 0.95, "page_number": 1},
                {"value": "INV-100", "confidence": 0.92, "page_number": 2},
            ],
        )
        issues = self.detector.evaluate({"invoice_number": field}, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)
        self.assertEqual(issues[0].category, ValidationCategory.CONFLICT)

    def test_distinct_competing_candidates_flagged(self) -> None:
        field = ExtractedField(
            name="invoice_number",
            value="INV-100",
            field_type=FieldType.IDENTIFIER,
            candidates=[
                {"value": "INV-100", "confidence": 0.95},
                {"value": "INV-999", "confidence": 0.88},
            ],
        )
        issues = self.detector.evaluate({"invoice_number": field}, [], DocumentType.INVOICE)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].status, ValidationStatus.INVALID)
        self.assertEqual(issues[0].severity, SeverityLevel.WARNING)
        self.assertIn("competing candidate values", issues[0].message)


if __name__ == "__main__":
    unittest.main()
