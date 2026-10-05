"""Typed domain models and data structures for document validation and consistency."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.core.models import (
    BoundingBox,
    ExtractedField,
    ExtractedTable,
    ValidationResult as CoreValidationResult,
)
from src.core.types import (
    DocumentType,
    ExtractionMethod,
    FieldType,
    SeverityLevel,
    ValidationStatus,
)


class ValidationCategory(str, Enum):
    """Categorization of validation checks and findings."""
    REQUIRED_FIELD = "required_field"
    FORMAT = "format"
    TYPE = "type"
    CROSS_FIELD = "cross_field"
    ARITHMETIC = "arithmetic"
    TABLE = "table"
    CONFLICT = "conflict"
    CONSISTENCY = "consistency"
    BUSINESS_RULE = "business_rule"


ValidationSeverity = SeverityLevel


@dataclass
class ValidationIssue(CoreValidationResult):
    """Detailed outcome of a single validation rule evaluation with source provenance."""
    category: ValidationCategory = ValidationCategory.CONSISTENCY
    bounding_box: Optional[BoundingBox] = None
    page_number: Optional[int] = None
    raw_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    extraction_method: Optional[ExtractionMethod] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number is not None and self.page_number < 1:
            raise ValueError(f"ValidationIssue page_number {self.page_number} must be >= 1")
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"ValidationIssue ocr_confidence {self.ocr_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ValidationIssue to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "category": self.category.value if hasattr(self.category, "value") else str(self.category),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "extraction_method": (
                self.extraction_method.value
                if self.extraction_method and hasattr(self.extraction_method, "value")
                else (str(self.extraction_method) if self.extraction_method else None)
            ),
            "metadata": self.metadata,
        })
        return base_dict

    def to_core_result(self) -> CoreValidationResult:
        """Convert to basic core ValidationResult instance."""
        return CoreValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            status=self.status,
            severity=self.severity,
            message=self.message,
            affected_fields=list(self.affected_fields),
            expected_value=self.expected_value,
            actual_value=self.actual_value,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationIssue:
        """Construct ValidationIssue from dictionary."""
        ts_str = data.get("timestamp")
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
        bbox_data = data.get("bounding_box")
        raw_cat = data.get("category", ValidationCategory.CONSISTENCY.value)
        try:
            cat = ValidationCategory(raw_cat)
        except ValueError:
            cat = ValidationCategory.CONSISTENCY

        raw_method = data.get("extraction_method")
        method = ExtractionMethod(raw_method) if raw_method else None

        return cls(
            rule_id=data["rule_id"],
            rule_name=data["rule_name"],
            status=ValidationStatus(data["status"]),
            severity=SeverityLevel(data["severity"]),
            message=data["message"],
            affected_fields=data.get("affected_fields", []),
            expected_value=data.get("expected_value"),
            actual_value=data.get("actual_value"),
            timestamp=ts,
            category=cat,
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            page_number=int(data["page_number"]) if data.get("page_number") is not None else None,
            raw_text=data.get("raw_text"),
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            extraction_method=method,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ValidationReport:
    """Comprehensive document-level validation outcome and consistency assessment."""
    document_id: str
    document_type: DocumentType = DocumentType.UNKNOWN
    overall_status: ValidationStatus = ValidationStatus.UNVALIDATED
    validation_score: float = 1.0
    rules_evaluated: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    rules_warning: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    field_statuses: Dict[str, ValidationStatus] = field(default_factory=dict)
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.validation_score <= 1.0):
            raise ValueError(f"Validation score {self.validation_score} must be in [0.0, 1.0]")

    @property
    def is_valid(self) -> bool:
        """True if overall validation passed without blocking errors."""
        return self.overall_status in (ValidationStatus.VALID, ValidationStatus.WARNING)

    @property
    def error_count(self) -> int:
        """Number of ERROR and CRITICAL severity issues."""
        return sum(
            1 for issue in self.issues
            if issue.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL) and issue.status == ValidationStatus.INVALID
        )

    @property
    def warning_count(self) -> int:
        """Number of WARNING severity issues."""
        return sum(
            1 for issue in self.issues
            if issue.severity == SeverityLevel.WARNING or issue.status == ValidationStatus.WARNING
        )

    def get_issues_by_severity(self, severity: SeverityLevel) -> List[ValidationIssue]:
        """Filter issues by specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Filter issues by specific category."""
        return [issue for issue in self.issues if issue.category == category]

    def get_issues_for_field(self, field_name: str) -> List[ValidationIssue]:
        """Retrieve all issues affecting a given field name."""
        return [issue for issue in self.issues if field_name in issue.affected_fields]

    def to_core_validation_results(self) -> List[CoreValidationResult]:
        """Export issues as standard core ValidationResult objects for legacy pipeline compatibility."""
        return [issue.to_core_result() for issue in self.issues]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ValidationReport to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value if hasattr(self.document_type, "value") else str(self.document_type),
            "overall_status": self.overall_status.value if hasattr(self.overall_status, "value") else str(self.overall_status),
            "validation_score": round(self.validation_score, 4),
            "rules_evaluated": self.rules_evaluated,
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "rules_warning": self.rules_warning,
            "issues": [issue.to_dict() for issue in self.issues],
            "field_statuses": {
                k: (v.value if hasattr(v, "value") else str(v))
                for k, v in self.field_statuses.items()
            },
            "discrepancies": self.discrepancies,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize ValidationReport to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationReport:
        """Construct ValidationReport from dictionary."""
        raw_doc_type = data.get("document_type", DocumentType.UNKNOWN.value)
        try:
            doc_type = DocumentType(raw_doc_type)
        except (ValueError, TypeError):
            doc_type = DocumentType.UNKNOWN

        raw_status = data.get("overall_status", ValidationStatus.UNVALIDATED.value)
        try:
            status = ValidationStatus(raw_status)
        except (ValueError, TypeError):
            status = ValidationStatus.UNVALIDATED

        issues_data = data.get("issues", [])
        field_statuses_raw = data.get("field_statuses", {})
        field_statuses = {}
        for k, v in field_statuses_raw.items():
            try:
                field_statuses[k] = ValidationStatus(v)
            except (ValueError, TypeError):
                field_statuses[k] = ValidationStatus.UNVALIDATED

        return cls(
            document_id=data["document_id"],
            document_type=doc_type,
            overall_status=status,
            validation_score=float(data.get("validation_score", 1.0)),
            rules_evaluated=int(data.get("rules_evaluated", 0)),
            rules_passed=int(data.get("rules_passed", 0)),
            rules_failed=int(data.get("rules_failed", 0)),
            rules_warning=int(data.get("rules_warning", 0)),
            issues=[ValidationIssue.from_dict(i) for i in issues_data],
            field_statuses=field_statuses,
            discrepancies=data.get("discrepancies", []),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "ValidationCategory",
    "ValidationSeverity",
    "ValidationStatus",
    "ValidationIssue",
    "ValidationReport",
]
