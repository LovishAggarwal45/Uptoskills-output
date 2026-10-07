"""Unit tests for RuleRegistry and validation rule catalog."""

import unittest
from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType
from src.validation.base import BaseValidationRule
from src.validation.models import ValidationIssue
from src.validation.rules import RuleRegistry


class CustomDummyRule(BaseValidationRule):
    @property
    def rule_id(self) -> str:
        return "VAL_CUSTOM_DUMMY"

    @property
    def rule_name(self) -> str:
        return "Custom Dummy Validation"

    @property
    def target_fields(self) -> List[str]:
        return ["dummy"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        return []


class TestValidationRules(unittest.TestCase):
    """Test suite for RuleRegistry and default rule suites."""

    def test_rule_registry_listing_and_lookup(self) -> None:
        rules = RuleRegistry.list_available_rules()
        self.assertIn("VAL_REQ_001", rules)
        self.assertIn("VAL_FMT_001", rules)
        self.assertIn("VAL_ARITH_SUBTOTAL_TAX_001", rules)

        req_cls = RuleRegistry.get_rule_cls("VAL_REQ_001")
        self.assertIsNotNone(req_cls)

    def test_custom_rule_registration(self) -> None:
        RuleRegistry.register("VAL_CUSTOM_DUMMY", CustomDummyRule)
        self.assertIn("VAL_CUSTOM_DUMMY", RuleRegistry.list_available_rules())
        self.assertEqual(RuleRegistry.get_rule_cls("VAL_CUSTOM_DUMMY"), CustomDummyRule)

    def test_build_default_suite(self) -> None:
        cfg = ValidationConfig(subtotal_tax_tolerance=0.01)
        suite = RuleRegistry.build_default_suite(cfg)
        self.assertGreaterEqual(len(suite), 8)
        rule_ids = [r.rule_id for r in suite]
        self.assertIn("VAL_REQ_001", rule_ids)
        self.assertIn("VAL_FMT_001", rule_ids)
        self.assertIn("VAL_XFLD_DATE_001", rule_ids)


if __name__ == "__main__":
    unittest.main()
