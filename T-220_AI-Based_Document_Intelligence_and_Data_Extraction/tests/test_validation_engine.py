"""Unit tests for validation engine contracts and deterministic rule evaluation."""

import unittest
from typing import Any, Dict, List, Optional

from src.core.models import (
    ExtractedField,
    ExtractedTable,
    ReviewFlag,
    ValidationResult,
)
from src.core.types import (
    ConfidenceSource,
    DocumentType,
    FieldType,
    ReviewTriggerType,
    SeverityLevel,
    ValidationStatus,
)
from src.validation.base import BaseValidationEngine, BaseValidationRule


class InvoiceArithmeticSumRule(BaseValidationRule):
    """Deterministic rule verifying: subtotal + tax == total (within tolerance)."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance

    @property
    def rule_id(self) -> str:
        return "RULE_INVOICE_ARITHMETIC_SUM"

    @property
    def rule_name(self) -> str:
        return "Invoice Subtotal + Tax Arithmetic Consistency"

    @property
    def target_fields(self) -> List[str]:
        return ["subtotal", "tax", "total"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationResult]:
        if document_type != DocumentType.INVOICE and document_type != DocumentType.UNKNOWN:
            return []

        subtotal_f = fields.get("subtotal")
        tax_f = fields.get("tax")
        total_f = fields.get("total")

        if not (subtotal_f and tax_f and total_f):
            return []

        try:
            subtotal_val = float(subtotal_f.normalized_value or subtotal_f.value)
            tax_val = float(tax_f.normalized_value or tax_f.value)
            total_val = float(total_f.normalized_value or total_f.value)
        except (ValueError, TypeError):
            return [
                ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message="Failed to parse numerical values for arithmetic validation.",
                    affected_fields=self.target_fields,
                )
            ]

        expected_total = subtotal_val + tax_val
        diff = abs(expected_total - total_val)

        if diff <= self.tolerance:
            return [
                ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.VALID,
                    severity=SeverityLevel.INFO,
                    message=f"Arithmetic check passed: {subtotal_val:.2f} + {tax_val:.2f} == {total_val:.2f}",
                    affected_fields=self.target_fields,
                    expected_value=expected_total,
                    actual_value=total_val,
                )
            ]
        else:
            return [
                ValidationResult(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Arithmetic mismatch: subtotal ({subtotal_val:.2f}) + tax ({tax_val:.2f}) = {expected_total:.2f}, but total is {total_val:.2f} (diff: {diff:.2f})",
                    affected_fields=self.target_fields,
                    expected_value=expected_total,
                    actual_value=total_val,
                )
            ]


class SimpleValidationEngine(BaseValidationEngine):
    """Concrete validation engine for executing registered rules."""

    def validate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
    ) -> List[ValidationResult]:
        all_results: List[ValidationResult] = []
        for rule in self.rules:
            results = rule.evaluate(fields, tables, document_type)
            for res in results:
                all_results.append(res)
                # Update affected field statuses if result is invalid/warning
                for field_name in res.affected_fields:
                    if field_name in fields:
                        if res.status == ValidationStatus.INVALID:
                            fields[field_name].validation_status = ValidationStatus.INVALID
                            fields[field_name].validation_messages.append(res.message)
                        elif res.status == ValidationStatus.VALID and fields[field_name].validation_status == ValidationStatus.UNVALIDATED:
                            fields[field_name].validation_status = ValidationStatus.VALID
        return all_results


class TestValidationEngine(unittest.TestCase):
    """Test suite for deterministic validation execution."""

    def setUp(self) -> None:
        self.engine = SimpleValidationEngine()
        self.engine.register_rule(InvoiceArithmeticSumRule(tolerance=0.05))

    def test_arithmetic_validation_valid(self) -> None:
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=100.0, normalized_value=100.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=10.0, normalized_value=10.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=110.0, normalized_value=110.0, field_type=FieldType.CURRENCY),
        }
        results = self.engine.validate(fields, [], DocumentType.INVOICE)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, ValidationStatus.VALID)
        self.assertEqual(results[0].severity, SeverityLevel.INFO)
        self.assertEqual(fields["total"].validation_status, ValidationStatus.VALID)

    def test_arithmetic_validation_mismatch_flags_error(self) -> None:
        fields = {
            "subtotal": ExtractedField(name="subtotal", value=100.0, normalized_value=100.0, field_type=FieldType.CURRENCY),
            "tax": ExtractedField(name="tax", value=10.0, normalized_value=10.0, field_type=FieldType.CURRENCY),
            "total": ExtractedField(name="total", value=150.0, normalized_value=150.0, field_type=FieldType.CURRENCY),
        }
        results = self.engine.validate(fields, [], DocumentType.INVOICE)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, ValidationStatus.INVALID)
        self.assertEqual(results[0].severity, SeverityLevel.ERROR)
        self.assertEqual(fields["total"].validation_status, ValidationStatus.INVALID)
        self.assertIn("Arithmetic mismatch", fields["total"].validation_messages[0])


if __name__ == "__main__":
    unittest.main()
