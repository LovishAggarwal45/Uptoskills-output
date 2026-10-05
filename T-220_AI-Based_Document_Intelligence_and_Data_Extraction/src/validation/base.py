"""Validation framework contracts, rule interfaces, and engine bases."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable, ValidationResult as CoreValidationResult
from src.core.types import DocumentType, SeverityLevel
from src.validation.models import ValidationCategory, ValidationIssue


class BaseValidationRule(ABC):
    """Abstract interface for a single deterministic or heuristic validation rule."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique machine-readable identifier for the rule (e.g. 'VAL_REQ_001')."""
        pass

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Human-readable name of the validation rule."""
        pass

    @property
    @abstractmethod
    def target_fields(self) -> List[str]:
        """Field names evaluated by this rule."""
        pass

    @property
    def category(self) -> ValidationCategory:
        """Categorization of this validation rule."""
        return ValidationCategory.CONSISTENCY

    @property
    def default_severity(self) -> SeverityLevel:
        """Default severity level when this rule fails."""
        return SeverityLevel.ERROR

    @property
    def is_enabled(self) -> bool:
        """Whether this rule is active."""
        return True

    @abstractmethod
    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Execute validation check against extracted fields and tables.

        Args:
            fields: Map of field name to ExtractedField instance.
            tables: List of extracted tables.
            document_type: Semantic document type.
            context: Additional runtime context or configuration.

        Returns:
            List of ValidationIssue items (can be empty if check passes or is not applicable).
        """
        pass


class BaseFieldValidator(ABC):
    """Abstract interface for specialized single-field value or format validators."""

    @abstractmethod
    def validate_field(
        self,
        field: ExtractedField,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate a single extracted field."""
        pass


class BaseDocumentValidator(ABC):
    """Abstract interface for document-type specific validation suites."""

    @abstractmethod
    def validate_document(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Run document-level validation checks."""
        pass


class BaseValidationEngine(ABC):
    """Abstract interface for orchestrating rule suites and updating field validation statuses."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.rules: List[BaseValidationRule] = []

    def register_rule(self, rule: BaseValidationRule) -> None:
        """Register a validation rule into the active suite."""
        self.rules.append(rule)

    def register_rules(self, rules: List[BaseValidationRule]) -> None:
        """Register multiple validation rules into the active suite."""
        for rule in rules:
            self.register_rule(rule)

    @abstractmethod
    def validate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CoreValidationResult]:
        """Run all registered rules against extracted fields and update field validation statuses.

        Args:
            fields: Extracted fields map.
            tables: Extracted tables.
            document_type: Classified document category.
            context: Optional execution context.

        Returns:
            Comprehensive list of all generated ValidationResults (or ValidationIssues).
        """
        pass


__all__ = [
    "BaseValidationRule",
    "BaseFieldValidator",
    "BaseDocumentValidator",
    "BaseValidationEngine",
]
