"""Document-type-aware required field validator."""

from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.validation.base import BaseValidationRule
from src.validation.models import ValidationCategory, ValidationIssue


class RequiredFieldValidator(BaseValidationRule):
    """Verifies presence and non-emptiness of mandatory fields for classified document types."""

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        custom_required_fields: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.required_field_mapping: Dict[DocumentType, List[str]] = {
            DocumentType.INVOICE: getattr(
                self.config, "invoice_required_fields", ["invoice_number", "invoice_date", "vendor_name", "total"]
            ),
            DocumentType.RECEIPT: getattr(
                self.config, "receipt_required_fields", ["merchant_name", "transaction_date", "total"]
            ),
            DocumentType.FORM: getattr(
                self.config, "form_required_fields", ["applicant_name", "date_of_birth"]
            ),
            DocumentType.RESUME: [],
            DocumentType.GENERAL_DOCUMENT: [],
            DocumentType.UNKNOWN: [],
        }
        if custom_required_fields:
            for doc_type_key, fields_list in custom_required_fields.items():
                try:
                    dt = DocumentType(doc_type_key) if isinstance(doc_type_key, str) else doc_type_key
                    self.required_field_mapping[dt] = fields_list
                except (ValueError, TypeError):
                    pass

    @property
    def rule_id(self) -> str:
        return "VAL_REQ_001"

    @property
    def rule_name(self) -> str:
        return "Required Field Presence Validation"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.REQUIRED_FIELD

    @property
    def target_fields(self) -> List[str]:
        all_fields = set()
        for field_list in self.required_field_mapping.values():
            all_fields.update(field_list)
        return sorted(list(all_fields))

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        required_names = self.required_field_mapping.get(document_type, [])

        for field_name in required_names:
            field_obj = fields.get(field_name)
            is_missing = False

            if field_obj is None:
                is_missing = True
            elif field_obj.value is None:
                is_missing = True
            elif isinstance(field_obj.value, str) and not field_obj.value.strip():
                is_missing = True

            rule_sub_id = f"VAL_REQ_{field_name.upper()}"
            if is_missing:
                issues.append(
                    ValidationIssue(
                        rule_id=rule_sub_id,
                        rule_name=f"Required Field: {field_name}",
                        status=ValidationStatus.INVALID,
                        severity=SeverityLevel.ERROR,
                        message=f"Mandatory field '{field_name}' is missing or empty for document type '{document_type.value}'.",
                        category=ValidationCategory.REQUIRED_FIELD,
                        affected_fields=[field_name],
                        expected_value="Present non-empty value",
                        actual_value=None if field_obj is None else field_obj.value,
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        rule_id=rule_sub_id,
                        rule_name=f"Required Field: {field_name}",
                        status=ValidationStatus.VALID,
                        severity=SeverityLevel.INFO,
                        message=f"Mandatory field '{field_name}' is present with value: '{field_obj.value}'.",
                        category=ValidationCategory.REQUIRED_FIELD,
                        affected_fields=[field_name],
                        expected_value="Present non-empty value",
                        actual_value=field_obj.value,
                        bounding_box=field_obj.bounding_box,
                        page_number=field_obj.page_number,
                        raw_text=field_obj.source_text,
                        ocr_confidence=field_obj.source_ocr_confidence,
                        extraction_method=field_obj.extraction_method,
                    )
                )

        return issues


__all__ = ["RequiredFieldValidator"]
