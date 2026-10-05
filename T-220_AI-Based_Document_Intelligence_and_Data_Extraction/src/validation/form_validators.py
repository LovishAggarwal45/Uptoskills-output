"""Specialized validation suite for forms and applications."""

from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType
from src.validation.base import BaseDocumentValidator, BaseValidationRule
from src.validation.conflict_detector import CandidateConflictDetector
from src.validation.cross_field_validators import DateOfBirthSanityRule
from src.validation.format_validators import DocumentFormatValidationRule
from src.validation.models import ValidationIssue
from src.validation.required_field_validator import RequiredFieldValidator


class FormValidator(BaseDocumentValidator):
    """Orchestrates validation checks specific to forms and structured application documents."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.rules: List[BaseValidationRule] = [
            RequiredFieldValidator(self.config),
            DocumentFormatValidationRule(self.config),
            DateOfBirthSanityRule(self.config),
            CandidateConflictDetector(),
        ]

    def validate_document(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.FORM,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        for rule in self.rules:
            rule_issues = rule.evaluate(fields, tables, document_type, context)
            issues.extend(rule_issues)
        return issues


__all__ = ["FormValidator"]
