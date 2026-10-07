"""Unit tests for field-level confidence calculation and review triggers."""

import unittest

from src.core.config import FieldConfidenceConfig
from src.core.models import BoundingBox, ExtractedField, Provenance
from src.core.types import ExtractionMethod, FieldType, SeverityLevel, ValidationStatus
from src.confidence.field_confidence import FieldConfidenceCalculator
from src.confidence.models import ConfidenceBand
from src.validation.models import ValidationIssue, ValidationReport


class TestFieldConfidence(unittest.TestCase):
    """Test suite for field-level confidence scoring, weighting, and review triggers."""

    def setUp(self) -> None:
        self.config = FieldConfidenceConfig(
            extraction_weight=0.50,
            ocr_weight=0.30,
            validation_weight=0.20,
            conflict_penalty=0.15,
            high_threshold=0.85,
            medium_threshold=0.65,
            low_threshold=0.40,
            min_confidence_threshold=0.60,
        )
        self.calc = FieldConfidenceCalculator(config=self.config)

    def test_strong_field_high_confidence(self) -> None:
        """Verify high confidence when extraction, OCR, and validation all pass."""
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance("doc_1", 1, "INV-2026-001", bbox, 0.98, ExtractionMethod.REGEX_PATTERN)

        field = ExtractedField(
            name="invoice_number",
            value="INV-2026-001",
            normalized_value="INV-2026-001",
            field_type=FieldType.IDENTIFIER,
            provenance=prov,
            extraction_confidence=0.96,
            validation_status=ValidationStatus.VALID,
        )

        fc = self.calc.calculate(field)

        # Expected: 0.96*0.5 + 0.98*0.3 + 1.0*0.2 = 0.48 + 0.294 + 0.20 = 0.974
        self.assertAlmostEqual(fc.aggregated_confidence, 0.974, places=2)
        self.assertEqual(fc.confidence_band, ConfidenceBand.HIGH)
        self.assertFalse(fc.review_required)
        self.assertEqual(len(fc.review_reasons), 0)

    def test_missing_ocr_signal_renormalizes_weights(self) -> None:
        """Verify that missing OCR confidence dynamically renormalizes across extraction and validation."""
        field = ExtractedField(
            name="customer_notes",
            value="Please expedite shipping",
            normalized_value="Please expedite shipping",
            field_type=FieldType.STRING,
            provenance=None,
            extraction_confidence=0.80,
            validation_status=ValidationStatus.VALID,
        )

        fc = self.calc.calculate(field)

        # Extraction weight 0.50 + Validation weight 0.20 = 0.70 total active
        # Expected: (0.80*0.50 + 1.0*0.20) / 0.70 = (0.40 + 0.20) / 0.70 = 0.60 / 0.70 ≈ 0.8571
        self.assertAlmostEqual(fc.aggregated_confidence, 0.8571, places=2)
        self.assertEqual(fc.confidence_band, ConfidenceBand.HIGH)
        self.assertFalse(fc.review_required)

    def test_invalid_validation_status_triggers_review(self) -> None:
        """Verify that validation status INVALID applies a strong penalty and triggers review."""
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance("doc_1", 1, "99/99/2026", bbox, 0.95, ExtractionMethod.REGEX_PATTERN)

        field = ExtractedField(
            name="invoice_date",
            value="99/99/2026",
            normalized_value=None,
            field_type=FieldType.DATE,
            provenance=prov,
            extraction_confidence=0.90,
            validation_status=ValidationStatus.INVALID,
            validation_messages=["Date format does not match recognized patterns"],
        )

        fc = self.calc.calculate(field)

        # Validation signal = 0.0 -> score = (0.90*0.5 + 0.95*0.3 + 0.0*0.2) = 0.45 + 0.285 + 0.0 = 0.735
        self.assertTrue(fc.review_required)
        self.assertTrue(any("Validation failed" in r for r in fc.review_reasons))

    def test_conflict_penalty_applied(self) -> None:
        """Verify candidate conflict penalty reduces aggregated confidence."""
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=40.0)
        prov = Provenance("doc_1", 1, "$450.00", bbox, 0.90, ExtractionMethod.KEY_VALUE_HEURISTIC)

        field = ExtractedField(
            name="total",
            value="$450.00",
            normalized_value=450.00,
            field_type=FieldType.CURRENCY,
            provenance=prov,
            extraction_confidence=0.85,
            validation_status=ValidationStatus.VALID,
            candidates=[
                {"value": "$450.00", "normalized_value": 450.00},
                {"value": "$495.00", "normalized_value": 495.00},
            ],
        )

        fc = self.calc.calculate(field)

        # Base score = 0.85*0.5 + 0.90*0.3 + 1.0*0.2 = 0.425 + 0.27 + 0.20 = 0.895
        # With penalty 0.15 -> 0.895 - 0.15 = 0.745
        self.assertAlmostEqual(fc.aggregated_confidence, 0.745, places=2)
        self.assertTrue(fc.review_required)
        self.assertTrue(any("conflicting" in r.lower() for r in fc.review_reasons))

    def test_missing_required_field_triggers_review(self) -> None:
        """Verify missing required field triggers review."""
        field = ExtractedField(
            name="vendor_name",
            value="",
            normalized_value=None,
            is_required=True,
            extraction_confidence=0.0,
            validation_status=ValidationStatus.INVALID,
        )

        fc = self.calc.calculate(field)
        self.assertTrue(fc.review_required)
        self.assertTrue(any("missing" in r.lower() for r in fc.review_reasons))


if __name__ == "__main__":
    unittest.main()
