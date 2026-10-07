"""Rule registry and rule catalog management for DocuMind AI validation."""

from typing import Any, Dict, List, Optional, Type

from src.core.config import ValidationConfig
from src.core.types import DocumentType
from src.validation.arithmetic_validators import (
    LineItemArithmeticRule,
    LineItemsSubtotalSumRule,
    SubtotalTaxTotalRule,
)
from src.validation.base import BaseValidationRule
from src.validation.conflict_detector import CandidateConflictDetector
from src.validation.cross_field_validators import (
    DateOfBirthSanityRule,
    InvoiceDueDateRule,
    SubtotalTotalRelationshipRule,
)
from src.validation.format_validators import DocumentFormatValidationRule
from src.validation.required_field_validator import RequiredFieldValidator
from src.validation.table_validators import TableIntegrityValidationRule


class RuleRegistry:
    """Catalog and factory for all available validation rules."""

    _registry: Dict[str, Type[BaseValidationRule]] = {
        "VAL_REQ_001": RequiredFieldValidator,
        "VAL_FMT_001": DocumentFormatValidationRule,
        "VAL_XFLD_DATE_001": InvoiceDueDateRule,
        "VAL_XFLD_SUBTOTAL_TOTAL_001": SubtotalTotalRelationshipRule,
        "VAL_XFLD_DOB_001": DateOfBirthSanityRule,
        "VAL_ARITH_SUBTOTAL_TAX_001": SubtotalTaxTotalRule,
        "VAL_ARITH_LINE_ITEM_001": LineItemArithmeticRule,
        "VAL_ARITH_LINES_SUBTOTAL_001": LineItemsSubtotalSumRule,
        "VAL_CONF_CANDIDATES_001": CandidateConflictDetector,
        "VAL_TBL_INTEGRITY_001": TableIntegrityValidationRule,
    }

    @classmethod
    def register(cls, rule_id: str, rule_cls: Type[BaseValidationRule]) -> None:
        """Register a new or custom rule class."""
        cls._registry[rule_id] = rule_cls

    @classmethod
    def get_rule_cls(cls, rule_id: str) -> Optional[Type[BaseValidationRule]]:
        """Lookup rule class by ID."""
        return cls._registry.get(rule_id)

    @classmethod
    def list_available_rules(cls) -> List[str]:
        """Return all registered rule IDs."""
        return sorted(list(cls._registry.keys()))

    @classmethod
    def build_default_suite(cls, config: Optional[ValidationConfig] = None) -> List[BaseValidationRule]:
        """Instantiate the standard default rule suite based on configuration."""
        cfg = config or ValidationConfig()
        return [
            RequiredFieldValidator(cfg),
            DocumentFormatValidationRule(cfg),
            InvoiceDueDateRule(cfg),
            SubtotalTotalRelationshipRule(getattr(cfg, "subtotal_tax_tolerance", 0.05)),
            DateOfBirthSanityRule(cfg),
            SubtotalTaxTotalRule(getattr(cfg, "subtotal_tax_tolerance", 0.05)),
            LineItemArithmeticRule(getattr(cfg, "line_item_amount_tolerance", 0.02)),
            LineItemsSubtotalSumRule(getattr(cfg, "table_subtotal_tolerance", 0.05)),
            CandidateConflictDetector(),
            TableIntegrityValidationRule(cfg),
        ]


__all__ = ["RuleRegistry"]
