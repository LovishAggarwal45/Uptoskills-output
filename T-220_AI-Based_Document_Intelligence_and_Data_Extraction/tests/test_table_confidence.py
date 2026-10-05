"""Unit tests for table and line-item confidence scoring and validation checks."""

import unittest

from src.core.config import TableConfidenceConfig
from src.core.models import BoundingBox, Provenance
from src.core.types import ExtractionMethod, ValidationStatus
from src.confidence.models import ConfidenceBand
from src.confidence.table_confidence import TableConfidenceCalculator
from src.tables.models import LineItem, Table, TableValidationResult


class TestTableConfidence(unittest.TestCase):
    """Test suite for line-item and table confidence aggregation."""

    def setUp(self) -> None:
        self.config = TableConfidenceConfig(
            structure_weight=0.30,
            row_weight=0.30,
            cell_weight=0.20,
            validation_weight=0.20,
            high_threshold=0.85,
            medium_threshold=0.65,
            low_threshold=0.40,
            min_line_item_threshold=0.60,
        )
        self.calc = TableConfidenceCalculator(config=self.config)

    def test_line_item_valid_arithmetic(self) -> None:
        """Verify line item with valid math receives high confidence score."""
        bbox = BoundingBox(xmin=50.0, ymin=200.0, xmax=500.0, ymax=220.0)
        prov = Provenance("doc_1", 1, "2 x $100.00 = $200.00", bbox, 0.98, ExtractionMethod.TABLE_STRUCTURE_PARSER)

        li = LineItem(
            row_index=0,
            description="Widget Model A",
            quantity=2.0,
            unit_price=100.0,
            amount=200.0,
            confidence=0.95,
            provenance=prov,
            is_valid_arithmetic=True,
        )

        lic = self.calc.calculate_line_item(li)
        self.assertGreater(lic.aggregated_confidence, 0.90)
        self.assertEqual(lic.confidence_band, ConfidenceBand.HIGH)
        self.assertFalse(lic.review_required)

    def test_line_item_arithmetic_mismatch_triggers_review(self) -> None:
        """Verify line item with arithmetic mismatch triggers review with clear reason."""
        bbox = BoundingBox(xmin=50.0, ymin=200.0, xmax=500.0, ymax=220.0)
        prov = Provenance("doc_1", 1, "2 x $100.00 = $300.00", bbox, 0.95, ExtractionMethod.TABLE_STRUCTURE_PARSER)

        li = LineItem(
            row_index=1,
            description="Widget Model B",
            quantity=2.0,
            unit_price=100.0,
            amount=300.0,  # Math error! 2 * 100 != 300
            confidence=0.90,
            provenance=prov,
            is_valid_arithmetic=False,
        )

        lic = self.calc.calculate_line_item(li)
        self.assertTrue(lic.review_required)
        self.assertTrue(any("arithmetic mismatch" in r.lower() for r in lic.review_reasons))

    def test_table_subtotal_mismatch_triggers_review(self) -> None:
        """Verify table validation failure triggers table-level review."""
        val_res = TableValidationResult(
            is_valid=False,
            row_checks_passed=2,
            row_checks_failed=0,
            subtotal_valid=False,
            calculated_subtotal=300.0,
            reported_subtotal=450.0,
            messages=["Reported subtotal $450.0 != sum of line items $300.0"],
        )

        table = Table(
            table_id="table_inv_01",
            page_number=1,
            headers=["Description", "Qty", "Price", "Amount"],
            confidence=0.92,
            confidence_breakdown={"structure_confidence": 0.90, "row_confidence": 0.95, "cell_confidence": 0.92},
            validation_status=ValidationStatus.INVALID,
            validation_result=val_res,
        )

        tc = self.calc.calculate_table(table)
        self.assertTrue(tc.review_required)
        self.assertTrue(any("subtotal mismatch" in r.lower() for r in tc.review_reasons))


if __name__ == "__main__":
    unittest.main()
