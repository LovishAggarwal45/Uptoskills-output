"""Specialized validation suite for invoices."""

from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType
from src.validation.arithmetic_validators import (
    LineItemArithmeticRule,
    LineItemsSubtotalSumRule,
    SubtotalTaxTotalRule,
)
from src.validation.base import BaseDocumentValidator, BaseValidationRule
from src.validation.conflict_detector import CandidateConflictDetector
from src.validation.cross_field_validators import (
    InvoiceDueDateRule,
    SubtotalTotalRelationshipRule,
)
from src.validation.format_validators import DocumentFormatValidationRule
from src.validation.models import ValidationIssue
from src.validation.required_field_validator import RequiredFieldValidator
from src.validation.table_validators import TableIntegrityValidationRule


class InvoiceValidator(BaseDocumentValidator):
    """Orchestrates comprehensive validation checks tailored for invoice documents."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.rules: List[BaseValidationRule] = [
            RequiredFieldValidator(self.config),
            DocumentFormatValidationRule(self.config),
            InvoiceDueDateRule(self.config),
            SubtotalTotalRelationshipRule(getattr(self.config, "subtotal_tax_tolerance", 0.05)),
            SubtotalTaxTotalRule(getattr(self.config, "subtotal_tax_tolerance", 0.05)),
            LineItemArithmeticRule(getattr(self.config, "line_item_amount_tolerance", 0.02)),
            LineItemsSubtotalSumRule(getattr(self.config, "table_subtotal_tolerance", 0.05)),
            CandidateConflictDetector(),
            TableIntegrityValidationRule(self.config),
        ]

    def validate_document(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.INVOICE,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for rule in self.rules:
            rule_issues = rule.evaluate(fields, tables, document_type, context)
            issues.extend(rule_issues)
        return issues


__all__ = ["InvoiceValidator"]
