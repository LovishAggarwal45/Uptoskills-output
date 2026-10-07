"""Format validation rules for date, currency, email, phone, and identifier fields."""

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType, FieldType, SeverityLevel, ValidationStatus
from src.validation.base import BaseFieldValidator, BaseValidationRule
from src.validation.models import ValidationCategory, ValidationIssue

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
PHONE_REGEX = re.compile(r"^\+?(\d[\d\-\.\s\(\)]{6,20}\d)$")


class DateFormatValidator(BaseFieldValidator):
    """Validates date values against parseable formats and sensible historical/future year bounds."""

    def __init__(
        self,
        formats: Optional[List[str]] = None,
        min_year: int = 1900,
        max_year: int = 2100,
    ) -> None:
        self.formats = formats or [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y.%m.%d",
            "%d.%m.%Y",
        ]
        self.min_year = min_year
        self.max_year = max_year

    def parse_date(self, value: Any) -> Optional[datetime]:
        """Attempt to parse date from string or return datetime if already parsed."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)
        if not isinstance(value, str):
            return None

        val_str = value.strip()
        if not val_str:
            return None

        # Try ISO format
        try:
            return datetime.fromisoformat(val_str)
        except ValueError:
            pass

        for fmt in self.formats:
            try:
                return datetime.strptime(val_str, fmt)
            except ValueError:
                continue

        # Regex fallback for standard YYYY-MM-DD, DD/MM/YYYY
        m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", val_str)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

        m2 = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", val_str)
        if m2:
            try:
                return datetime(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
            except ValueError:
                try:
                    return datetime(int(m2.group(3)), int(m2.group(1)), int(m2.group(2)))
                except ValueError:
                    pass

        return None

    def validate_field(
        self,
        field: ExtractedField,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        target_val = field.normalized_value if field.normalized_value is not None else field.value
        if target_val is None or (isinstance(target_val, str) and not target_val.strip()):
            return []

        parsed_dt = self.parse_date(target_val)
        rule_id = f"VAL_FMT_DATE_{field.name.upper()}"
        rule_name = f"Date Format Validation: {field.name}"

        if parsed_dt is None:
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Field '{field.name}' value '{target_val}' cannot be parsed as a valid calendar date.",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value="Valid date (e.g. YYYY-MM-DD or DD/MM/YYYY)",
                    actual_value=target_val,
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        if not (self.min_year <= parsed_dt.year <= self.max_year):
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Date '{target_val}' has anomalous year {parsed_dt.year} (must be between {self.min_year} and {self.max_year}).",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value=f"Year between {self.min_year} and {self.max_year}",
                    actual_value=str(parsed_dt.year),
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=rule_id,
                rule_name=rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Date '{target_val}' is syntactically valid ({parsed_dt.strftime('%Y-%m-%d')}).",
                category=ValidationCategory.FORMAT,
                affected_fields=[field.name],
                expected_value="Valid calendar date",
                actual_value=parsed_dt.strftime("%Y-%m-%d"),
                bounding_box=field.bounding_box,
                page_number=field.page_number,
                raw_text=field.source_text,
                ocr_confidence=field.source_ocr_confidence,
                extraction_method=field.extraction_method,
            )
        ]


class CurrencyFormatValidator(BaseFieldValidator):
    """Validates monetary / currency fields for numeric parseability and valid values."""

    def __init__(self, currency_symbols: Optional[List[str]] = None) -> None:
        self.currency_symbols = currency_symbols or ["$", "€", "£", "₹", "¥", "USD", "EUR", "GBP", "CAD", "AUD", "INR"]

    def parse_amount(self, value: Any) -> Optional[float]:
        """Convert input value to float amount."""
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        val_clean = value.strip()
        for sym in self.currency_symbols:
            val_clean = val_clean.replace(sym, "")
        val_clean = val_clean.replace(",", "").strip()

        try:
            return float(val_clean)
        except ValueError:
            return None

    def validate_field(
        self,
        field: ExtractedField,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        target_val = field.normalized_value if field.normalized_value is not None else field.value
        if target_val is None or (isinstance(target_val, str) and not target_val.strip()):
            return []

        amount = self.parse_amount(target_val)
        rule_id = f"VAL_FMT_CURR_{field.name.upper()}"
        rule_name = f"Currency Format Validation: {field.name}"

        if amount is None:
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Field '{field.name}' value '{target_val}' is not a valid numerical currency amount.",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value="Numeric decimal amount",
                    actual_value=target_val,
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        # Check for non-negative totals unless discount/credit
        if amount < 0 and "discount" not in field.name.lower() and "refund" not in field.name.lower():
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.WARNING,
                    severity=SeverityLevel.WARNING,
                    message=f"Monetary field '{field.name}' has negative amount ({amount:.2f}).",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value="Non-negative currency value",
                    actual_value=amount,
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=rule_id,
                rule_name=rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Currency field '{field.name}' has valid amount: {amount:.2f}",
                category=ValidationCategory.FORMAT,
                affected_fields=[field.name],
                expected_value="Valid currency amount",
                actual_value=round(amount, 2),
                bounding_box=field.bounding_box,
                page_number=field.page_number,
                raw_text=field.source_text,
                ocr_confidence=field.source_ocr_confidence,
                extraction_method=field.extraction_method,
            )
        ]


class EmailFormatValidator(BaseFieldValidator):
    """Validates email addresses against standard format patterns."""

    def validate_field(
        self,
        field: ExtractedField,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        target_val = str(field.normalized_value or field.value or "").strip()
        if not target_val:
            return []

        rule_id = f"VAL_FMT_EMAIL_{field.name.upper()}"
        rule_name = f"Email Format Validation: {field.name}"

        if not EMAIL_REGEX.match(target_val):
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Field '{field.name}' value '{target_val}' is not a valid email address.",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value="name@domain.com",
                    actual_value=target_val,
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=rule_id,
                rule_name=rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Email address '{target_val}' is well-formed.",
                category=ValidationCategory.FORMAT,
                affected_fields=[field.name],
                expected_value="name@domain.com",
                actual_value=target_val,
                bounding_box=field.bounding_box,
                page_number=field.page_number,
                raw_text=field.source_text,
                ocr_confidence=field.source_ocr_confidence,
                extraction_method=field.extraction_method,
            )
        ]


class PhoneFormatValidator(BaseFieldValidator):
    """Validates telephone numbers for digit count and standard separators."""

    def validate_field(
        self,
        field: ExtractedField,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        target_val = str(field.normalized_value or field.value or "").strip()
        if not target_val:
            return []

        digits_only = re.sub(r"\D", "", target_val)
        rule_id = f"VAL_FMT_PHONE_{field.name.upper()}"
        rule_name = f"Phone Number Format Validation: {field.name}"

        if len(digits_only) < 7 or len(digits_only) > 15 or not PHONE_REGEX.match(target_val):
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.WARNING,
                    message=f"Field '{field.name}' value '{target_val}' does not match standard telephone number formats (found {len(digits_only)} digits).",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value="Valid phone number (7-15 digits)",
                    actual_value=target_val,
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=rule_id,
                rule_name=rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Phone number '{target_val}' is well-formed.",
                category=ValidationCategory.FORMAT,
                affected_fields=[field.name],
                expected_value="Valid telephone format",
                actual_value=target_val,
                bounding_box=field.bounding_box,
                page_number=field.page_number,
                raw_text=field.source_text,
                ocr_confidence=field.source_ocr_confidence,
                extraction_method=field.extraction_method,
            )
        ]


class IdentifierFormatValidator(BaseFieldValidator):
    """Validates business identifiers, invoice numbers, or reference IDs."""

    def validate_field(
        self,
        field: ExtractedField,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        target_val = str(field.normalized_value or field.value or "").strip()
        if not target_val:
            return []

        rule_id = f"VAL_FMT_ID_{field.name.upper()}"
        rule_name = f"Identifier Format Validation: {field.name}"

        # Clean noise characters
        alnum_chars = [c for c in target_val if c.isalnum()]
        if len(alnum_chars) < 2:
            return [
                ValidationIssue(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Identifier '{field.name}' value '{target_val}' is too short or contains only punctuation/noise.",
                    category=ValidationCategory.FORMAT,
                    affected_fields=[field.name],
                    expected_value="Alphanumeric identifier (>= 2 chars)",
                    actual_value=target_val,
                    bounding_box=field.bounding_box,
                    page_number=field.page_number,
                    raw_text=field.source_text,
                    ocr_confidence=field.source_ocr_confidence,
                    extraction_method=field.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=rule_id,
                rule_name=rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Identifier '{field.name}' value '{target_val}' is valid.",
                category=ValidationCategory.FORMAT,
                affected_fields=[field.name],
                expected_value="Alphanumeric identifier",
                actual_value=target_val,
                bounding_box=field.bounding_box,
                page_number=field.page_number,
                raw_text=field.source_text,
                ocr_confidence=field.source_ocr_confidence,
                extraction_method=field.extraction_method,
            )
        ]


class DocumentFormatValidationRule(BaseValidationRule):
    """Applies comprehensive format validation across all extracted fields based on their semantic types."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.date_validator = DateFormatValidator(
            formats=getattr(self.config, "date_formats", None),
            min_year=getattr(self.config, "min_valid_year", 1900),
            max_year=getattr(self.config, "max_valid_year", 2100),
        )
        self.currency_validator = CurrencyFormatValidator(
            currency_symbols=getattr(self.config, "currency_symbols", None)
        )
        self.email_validator = EmailFormatValidator()
        self.phone_validator = PhoneFormatValidator()
        self.id_validator = IdentifierFormatValidator()

    @property
    def rule_id(self) -> str:
        return "VAL_FMT_001"

    @property
    def rule_name(self) -> str:
        return "Comprehensive Field Format Validation"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.FORMAT

    @property
    def target_fields(self) -> List[str]:
        return ["*"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for field_name, field_obj in fields.items():
            if field_obj.value is None:
                continue

            fname_lower = field_name.lower()
            ftype = field_obj.field_type

            # Date check
            if (
                ftype == FieldType.DATE
                or fname_lower in ("date", "dob", "due_date", "issue_date", "invoice_date", "transaction_date", "document_date", "birth_date", "delivery_date", "payment_date", "statement_date")
                or (("date" in fname_lower.split("_") or fname_lower.endswith("_date") or fname_lower.startswith("date_")) and "candidate" not in fname_lower and "mandate" not in fname_lower and "update" not in fname_lower)
            ):
                issues.extend(self.date_validator.validate_field(field_obj, context))

            # Currency check
            elif ftype == FieldType.CURRENCY or fname_lower in ("total", "subtotal", "tax", "amount", "unit_price", "discount", "tip", "shipping"):
                issues.extend(self.currency_validator.validate_field(field_obj, context))

            # Email check
            elif ftype == FieldType.EMAIL or "email" in fname_lower:
                issues.extend(self.email_validator.validate_field(field_obj, context))

            # Phone check
            elif ftype == FieldType.PHONE or "phone" in fname_lower or "tel" in fname_lower or "fax" in fname_lower:
                issues.extend(self.phone_validator.validate_field(field_obj, context))

            # Identifier check
            elif ftype == FieldType.IDENTIFIER or fname_lower in ("invoice_number", "order_number", "account_number", "tax_id", "ssn", "ein", "id_number"):
                issues.extend(self.id_validator.validate_field(field_obj, context))

        return issues


__all__ = [
    "DateFormatValidator",
    "CurrencyFormatValidator",
    "EmailFormatValidator",
    "PhoneFormatValidator",
    "IdentifierFormatValidator",
    "DocumentFormatValidationRule",
]
