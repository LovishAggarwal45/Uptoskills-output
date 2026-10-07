"""Unit tests for Phase 7 validation models and serialization."""

import unittest
from datetime import datetime, timezone

from src.core.models import BoundingBox, ExtractedField
from src.core.types import (
    DocumentType,
    ExtractionMethod,
    FieldType,
    SeverityLevel,
    ValidationStatus,
)
from src.validation.models import (
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


class TestValidationModels(unittest.TestCase):
    """Test suite for ValidationIssue and ValidationReport domain models."""

    def test_validation_issue_creation_and_provenance(self) -> None:
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=50.0)
        issue = ValidationIssue(
            rule_id="VAL_REQ_INVOICE_NUMBER",
            rule_name="Required Field: invoice_number",
            status=ValidationStatus.VALID,
            severity=SeverityLevel.INFO,
            message="Field invoice_number is present and valid.",
            category=ValidationCategory.REQUIRED_FIELD,
            affected_fields=["invoice_number"],
            expected_value="Present non-empty value",
            actual_value="INV-2023-001",
            bounding_box=bbox,
            page_number=1,
            raw_text="Invoice #: INV-2023-001",
            ocr_confidence=0.98,
            extraction_method=ExtractionMethod.KEY_VALUE_HEURISTIC,
            metadata={"source": "header"},
        )

        self.assertEqual(issue.rule_id, "VAL_REQ_INVOICE_NUMBER")
        self.assertEqual(issue.status, ValidationStatus.VALID)
        self.assertEqual(issue.category, ValidationCategory.REQUIRED_FIELD)
        self.assertEqual(issue.page_number, 1)
        self.assertEqual(issue.ocr_confidence, 0.98)
        self.assertEqual(issue.bounding_box.xmin, 10.0)

    def test_validation_issue_dict_and_core_conversion(self) -> None:
        issue = ValidationIssue(
            rule_id="VAL_ARITH_001",
            rule_name="Total Math Check",
            status=ValidationStatus.INVALID,
            severity=SeverityLevel.ERROR,
            message="Mismatch in total sum.",
            category=ValidationCategory.ARITHMETIC,
            affected_fields=["total", "subtotal"],
            expected_value=110.0,
            actual_value=150.0,
        )

        d = issue.to_dict()
        self.assertEqual(d["rule_id"], "VAL_ARITH_001")
        self.assertEqual(d["category"], "arithmetic")
        self.assertEqual(d["status"], "invalid")
        self.assertEqual(d["severity"], "error")

        # Roundtrip from_dict
        restored = ValidationIssue.from_dict(d)
        self.assertEqual(restored.rule_id, issue.rule_id)
        self.assertEqual(restored.category, ValidationCategory.ARITHMETIC)
        self.assertEqual(restored.status, ValidationStatus.INVALID)

        # Core result conversion
        core_res = issue.to_core_result()
        self.assertEqual(core_res.rule_id, "VAL_ARITH_001")
        self.assertEqual(core_res.status, ValidationStatus.INVALID)

    def test_validation_report_properties_and_serialization(self) -> None:
        issue1 = ValidationIssue(
            rule_id="VAL_REQ_001",
            rule_name="Required Total",
            status=ValidationStatus.VALID,
            severity=SeverityLevel.INFO,
            message="Total is present",
            category=ValidationCategory.REQUIRED_FIELD,
            affected_fields=["total"],
        )
        issue2 = ValidationIssue(
            rule_id="VAL_FMT_001",
            rule_name="Date Format",
            status=ValidationStatus.INVALID,
            severity=SeverityLevel.ERROR,
            message="Invalid date format",
            category=ValidationCategory.FORMAT,
            affected_fields=["invoice_date"],
        )
        issue3 = ValidationIssue(
            rule_id="VAL_CONF_001",
            rule_name="Ambiguity Check",
            status=ValidationStatus.WARNING,
            severity=SeverityLevel.WARNING,
            message="Multiple possible invoice numbers",
            category=ValidationCategory.CONFLICT,
            affected_fields=["invoice_number"],
        )

        report = ValidationReport(
            document_id="doc_test_123",
            document_type=DocumentType.INVOICE,
            overall_status=ValidationStatus.INVALID,
            validation_score=0.65,
            rules_evaluated=3,
            rules_passed=1,
            rules_failed=1,
            rules_warning=1,
            issues=[issue1, issue2, issue3],
            field_statuses={"total": ValidationStatus.VALID, "invoice_date": ValidationStatus.INVALID},
            execution_time_seconds=0.012,
        )

        self.assertFalse(report.is_valid)
        self.assertEqual(report.error_count, 1)
        self.assertEqual(report.warning_count, 1)
        self.assertEqual(len(report.get_issues_by_severity(SeverityLevel.ERROR)), 1)
        self.assertEqual(len(report.get_issues_by_category(ValidationCategory.FORMAT)), 1)
        self.assertEqual(len(report.get_issues_for_field("total")), 1)

        # Serialization to dict & json
        d = report.to_dict()
        self.assertEqual(d["document_id"], "doc_test_123")
        self.assertEqual(d["document_type"], "invoice")
        self.assertEqual(len(d["issues"]), 3)

        json_str = report.to_json()
        self.assertIn("doc_test_123", json_str)
        self.assertIn("VAL_FMT_001", json_str)

        # Roundtrip from_dict
        restored = ValidationReport.from_dict(d)
        self.assertEqual(restored.document_id, "doc_test_123")
        self.assertEqual(restored.document_type, DocumentType.INVOICE)
        self.assertEqual(len(restored.issues), 3)


if __name__ == "__main__":
    unittest.main()
