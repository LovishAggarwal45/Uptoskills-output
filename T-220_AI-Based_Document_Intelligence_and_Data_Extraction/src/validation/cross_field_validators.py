"""Cross-field consistency rules verifying logical relationships between extracted fields."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.validation.base import BaseValidationRule
from src.validation.format_validators import DateFormatValidator, CurrencyFormatValidator
from src.validation.models import ValidationCategory, ValidationIssue


class InvoiceDueDateRule(BaseValidationRule):
    """Verifies that invoice due_date is chronologically on or after invoice_date."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.date_parser = DateFormatValidator(
            formats=getattr(self.config, "date_formats", None)
        )

    @property
    def rule_id(self) -> str:
        return "VAL_XFLD_DATE_001"

    @property
    def rule_name(self) -> str:
        return "Invoice Date vs Due Date Chronology"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CROSS_FIELD

    @property
    def target_fields(self) -> List[str]:
        return ["invoice_date", "due_date"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        inv_date_f = fields.get("invoice_date") or fields.get("transaction_date") or fields.get("date")
        due_date_f = fields.get("due_date") or fields.get("payment_due_date")

        if not (inv_date_f and due_date_f):
            return []

        inv_val = inv_date_f.normalized_value or inv_date_f.value
        due_val = due_date_f.normalized_value or due_date_f.value

        inv_dt = self.date_parser.parse_date(inv_val)
        due_dt = self.date_parser.parse_date(due_val)

        if not (inv_dt and due_dt):
            return []

        affected = [inv_date_f.name, due_date_f.name]

        if due_dt < inv_dt:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=(
                        f"Chronological conflict: due date '{due_dt.strftime('%Y-%m-%d')}' "
                        f"is earlier than invoice date '{inv_dt.strftime('%Y-%m-%d')}'."
                    ),
                    category=self.category,
                    affected_fields=affected,
                    expected_value=f">= {inv_dt.strftime('%Y-%m-%d')}",
                    actual_value=due_dt.strftime("%Y-%m-%d"),
                    bounding_box=due_date_f.bounding_box or inv_date_f.bounding_box,
                    page_number=due_date_f.page_number,
                    ocr_confidence=due_date_f.source_ocr_confidence,
                    extraction_method=due_date_f.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=(
                    f"Date chronology valid: invoice date ({inv_dt.strftime('%Y-%m-%d')}) "
                    f"<= due date ({due_dt.strftime('%Y-%m-%d')})."
                ),
                category=self.category,
                affected_fields=affected,
                expected_value=f">= {inv_dt.strftime('%Y-%m-%d')}",
                actual_value=due_dt.strftime("%Y-%m-%d"),
                bounding_box=due_date_f.bounding_box,
                page_number=due_date_f.page_number,
                ocr_confidence=due_date_f.source_ocr_confidence,
                extraction_method=due_date_f.extraction_method,
            )
        ]


class SubtotalTotalRelationshipRule(BaseValidationRule):
    """Verifies that subtotal does not exceed grand total when tax/fees are non-negative."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance
        self.currency_parser = CurrencyFormatValidator()

    @property
    def rule_id(self) -> str:
        return "VAL_XFLD_SUBTOTAL_TOTAL_001"

    @property
    def rule_name(self) -> str:
        return "Subtotal vs Total Magnitude Consistency"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CROSS_FIELD

    @property
    def target_fields(self) -> List[str]:
        return ["subtotal", "total"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        subtotal_f = fields.get("subtotal")
        total_f = fields.get("total")

        if not (subtotal_f and total_f):
            return []

        sub_val = self.currency_parser.parse_amount(subtotal_f.normalized_value or subtotal_f.value)
        tot_val = self.currency_parser.parse_amount(total_f.normalized_value or total_f.value)

        if sub_val is None or tot_val is None:
            return []

        # If discount exists, subtotal might legitimately exceed total
        discount_f = fields.get("discount")
        discount_val = self.currency_parser.parse_amount(
            discount_f.normalized_value or discount_f.value
        ) if discount_f else 0.0

        effective_sub = sub_val - (discount_val or 0.0)

        affected = [subtotal_f.name, total_f.name]

        if effective_sub > tot_val + self.tolerance:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=(
                        f"Logical mismatch: subtotal ({sub_val:.2f}) exceeds total ({tot_val:.2f}) "
                        f"by {(sub_val - tot_val):.2f} without sufficient discount."
                    ),
                    category=self.category,
                    affected_fields=affected,
                    expected_value=f"<= {tot_val:.2f}",
                    actual_value=sub_val,
                    bounding_box=subtotal_f.bounding_box or total_f.bounding_box,
                    page_number=total_f.page_number,
                    ocr_confidence=total_f.source_ocr_confidence,
                    extraction_method=total_f.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Subtotal ({sub_val:.2f}) <= Total ({tot_val:.2f}) is logically consistent.",
                category=self.category,
                affected_fields=affected,
                expected_value=f"<= {tot_val:.2f}",
                actual_value=sub_val,
                bounding_box=total_f.bounding_box,
                page_number=total_f.page_number,
                ocr_confidence=total_f.source_ocr_confidence,
                extraction_method=total_f.extraction_method,
            )
        ]


class DateOfBirthSanityRule(BaseValidationRule):
    """Verifies that applicant date of birth is in the past and represents a realistic human age."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.date_parser = DateFormatValidator(
            formats=getattr(self.config, "date_formats", None)
        )

    @property
    def rule_id(self) -> str:
        return "VAL_XFLD_DOB_001"

    @property
    def rule_name(self) -> str:
        return "Date of Birth Sanity Validation"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CROSS_FIELD

    @property
    def target_fields(self) -> List[str]:
        return ["date_of_birth", "dob", "birth_date"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        dob_f = fields.get("date_of_birth") or fields.get("dob") or fields.get("birth_date")
        if not dob_f:
            return []

        dob_dt = self.date_parser.parse_date(dob_f.normalized_value or dob_f.value)
        if not dob_dt:
            return []

        now = datetime.now(timezone.utc)
        dob_dt_tz = dob_dt if dob_dt.tzinfo else dob_dt.replace(tzinfo=timezone.utc)

        if dob_dt_tz > now:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=f"Date of birth '{dob_dt.strftime('%Y-%m-%d')}' cannot be in the future.",
                    category=self.category,
                    affected_fields=[dob_f.name],
                    expected_value="Past date",
                    actual_value=dob_dt.strftime("%Y-%m-%d"),
                    bounding_box=dob_f.bounding_box,
                    page_number=dob_f.page_number,
                    ocr_confidence=dob_f.source_ocr_confidence,
                    extraction_method=dob_f.extraction_method,
                )
            ]

        age_years = (now - dob_dt_tz).days / 365.25
        if age_years > 130:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.WARNING,
                    message=f"Date of birth '{dob_dt.strftime('%Y-%m-%d')}' indicates anomalous age ({age_years:.1f} years).",
                    category=self.category,
                    affected_fields=[dob_f.name],
                    expected_value="Age <= 130 years",
                    actual_value=f"{age_years:.1f} years",
                    bounding_box=dob_f.bounding_box,
                    page_number=dob_f.page_number,
                    ocr_confidence=dob_f.source_ocr_confidence,
                    extraction_method=dob_f.extraction_method,
                )
            ]

        return [
            ValidationIssue(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                status=ValidationStatus.VALID,
                severity=SeverityLevel.INFO,
                message=f"Date of birth '{dob_dt.strftime('%Y-%m-%d')}' is valid (age ~{int(age_years)} years).",
                category=self.category,
                affected_fields=[dob_f.name],
                expected_value="Valid past date",
                actual_value=dob_dt.strftime("%Y-%m-%d"),
                bounding_box=dob_f.bounding_box,
                page_number=dob_f.page_number,
                ocr_confidence=dob_f.source_ocr_confidence,
                extraction_method=dob_f.extraction_method,
            )
        ]


__all__ = [
    "InvoiceDueDateRule",
    "SubtotalTotalRelationshipRule",
    "DateOfBirthSanityRule",
]
