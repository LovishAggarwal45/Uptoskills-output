"""DocuMind AI validation and consistency subsystem."""

from src.validation.arithmetic_validators import (
    LineItemArithmeticRule,
    LineItemsSubtotalSumRule,
    SubtotalTaxTotalRule,
)
from src.validation.base import (
    BaseDocumentValidator,
    BaseFieldValidator,
    BaseValidationEngine,
    BaseValidationRule,
)
from src.validation.confidence import ValidationConfidenceCalculator
from src.validation.conflict_detector import CandidateConflictDetector
from src.validation.cross_field_validators import (
    DateOfBirthSanityRule,
    InvoiceDueDateRule,
    SubtotalTotalRelationshipRule,
)
from src.validation.exceptions import (
    ValidationConfigurationError,
    ValidationDataError,
    ValidationError,
    ValidationRuleError,
)
from src.validation.format_validators import (
    CurrencyFormatValidator,
    DateFormatValidator,
    DocumentFormatValidationRule,
    EmailFormatValidator,
    IdentifierFormatValidator,
    PhoneFormatValidator,
)
from src.validation.form_validators import FormValidator
from src.validation.invoice_validators import InvoiceValidator
from src.validation.models import (
    ValidationCategory,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationStatus,
)
from src.validation.receipt_validators import ReceiptValidator
from src.validation.required_field_validator import RequiredFieldValidator
from src.validation.rules import RuleRegistry
from src.validation.table_validators import TableIntegrityValidationRule
from src.validation.validator import DocumentValidationEngine

__all__ = [
    # Exceptions
    "ValidationError",
    "ValidationConfigurationError",
    "ValidationRuleError",
    "ValidationDataError",
    # Models & Enums
    "ValidationCategory",
    "ValidationSeverity",
    "ValidationStatus",
    "ValidationIssue",
    "ValidationReport",
    # Base Classes
    "BaseValidationRule",
    "BaseFieldValidator",
    "BaseDocumentValidator",
    "BaseValidationEngine",
    # Rules & Registries
    "RuleRegistry",
    "RequiredFieldValidator",
    "DateFormatValidator",
    "CurrencyFormatValidator",
    "EmailFormatValidator",
    "PhoneFormatValidator",
    "IdentifierFormatValidator",
    "DocumentFormatValidationRule",
    "InvoiceDueDateRule",
    "SubtotalTotalRelationshipRule",
    "DateOfBirthSanityRule",
    "SubtotalTaxTotalRule",
    "LineItemArithmeticRule",
    "LineItemsSubtotalSumRule",
    "CandidateConflictDetector",
    "TableIntegrityValidationRule",
    # Specialized Validators
    "InvoiceValidator",
    "ReceiptValidator",
    "FormValidator",
    # Confidence & Coordinator
    "ValidationConfidenceCalculator",
    "DocumentValidationEngine",
]
