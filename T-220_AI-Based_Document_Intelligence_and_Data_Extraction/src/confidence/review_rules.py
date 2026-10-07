"""Declarative review rules and issue generation for human review routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.config import ReviewRoutingConfig
from src.core.models import Document, ExtractedField
from src.core.types import DocumentType, ReviewTriggerType, SeverityLevel, ValidationStatus
from src.confidence.models import (
    DocumentConfidence,
    FieldConfidence,
    ReviewIssue,
    ReviewTarget,
    ReviewTargetType,
    TableConfidence,
)
from src.tables.models import Table
from src.validation.models import ValidationReport


class BaseReviewRule(ABC):
    """Abstract interface for human review triggering rules."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique identifier for the review rule."""
        pass

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Human-readable descriptive name."""
        pass

    @abstractmethod
    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        """Evaluate rule logic and produce granular review issues with spatial targets."""
        pass


class LowFieldConfidenceReviewRule(BaseReviewRule):
    """Flags fields with aggregated confidence below configured threshold."""

    @property
    def rule_id(self) -> str:
        return "REV_LOW_FIELD_CONFIDENCE"

    @property
    def rule_name(self) -> str:
        return "Low Field Confidence Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        cfg = config or ReviewRoutingConfig()
        issues: List[ReviewIssue] = []

        for name, fc in field_confidences.items():
            if fc.aggregated_confidence < cfg.low_confidence_threshold:
                bbox = fc.provenance.bounding_box if fc.provenance else None
                page = fc.provenance.page_number if fc.provenance else 1
                target = ReviewTarget(
                    target_type=ReviewTargetType.FIELD,
                    page_number=page,
                    field_name=name,
                    bounding_box=bbox,
                    reason=f"Aggregated confidence ({fc.aggregated_confidence:.2f}) < threshold ({cfg.low_confidence_threshold:.2f})",
                    severity=SeverityLevel.WARNING if fc.aggregated_confidence >= 0.50 else SeverityLevel.ERROR,
                )
                issues.append(
                    ReviewIssue(
                        issue_id=f"{self.rule_id}_{name}",
                        rule_name=self.rule_name,
                        severity=target.severity,
                        message=f"Field '{name}' has low confidence score ({fc.aggregated_confidence:.2f})",
                        affected_targets=[target],
                        source_signal="aggregated_confidence",
                    )
                )
        return issues


class LowOCRConfidenceReviewRule(BaseReviewRule):
    """Flags fields where optical character recognition confidence is suspect."""

    @property
    def rule_id(self) -> str:
        return "REV_LOW_OCR_CONFIDENCE"

    @property
    def rule_name(self) -> str:
        return "Low OCR Recognition Confidence Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        cfg = config or ReviewRoutingConfig()
        issues: List[ReviewIssue] = []

        for name, fc in field_confidences.items():
            if fc.ocr_confidence is not None and fc.ocr_confidence < cfg.low_ocr_threshold:
                bbox = fc.provenance.bounding_box if fc.provenance else None
                page = fc.provenance.page_number if fc.provenance else 1
                target = ReviewTarget(
                    target_type=ReviewTargetType.FIELD,
                    page_number=page,
                    field_name=name,
                    bounding_box=bbox,
                    reason=f"OCR recognition confidence ({fc.ocr_confidence:.2f}) < threshold ({cfg.low_ocr_threshold:.2f})",
                    severity=SeverityLevel.WARNING,
                )
                issues.append(
                    ReviewIssue(
                        issue_id=f"{self.rule_id}_{name}",
                        rule_name=self.rule_name,
                        severity=SeverityLevel.WARNING,
                        message=f"Field '{name}' has low OCR confidence ({fc.ocr_confidence:.2f})",
                        affected_targets=[target],
                        source_signal="ocr_confidence",
                    )
                )
        return issues


class MissingRequiredFieldReviewRule(BaseReviewRule):
    """Flags documents missing essential required fields."""

    @property
    def rule_id(self) -> str:
        return "REV_MISSING_REQUIRED_FIELD"

    @property
    def rule_name(self) -> str:
        return "Mandatory Required Field Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for name, fc in field_confidences.items():
            is_req = fc.metadata.get("is_required", False)
            if is_req and (fc.value is None or str(fc.value).strip() == ""):
                target = ReviewTarget(
                    target_type=ReviewTargetType.FIELD,
                    page_number=1,
                    field_name=name,
                    reason=f"Mandatory field '{name}' is missing or empty",
                    severity=SeverityLevel.ERROR,
                )
                issues.append(
                    ReviewIssue(
                        issue_id=f"{self.rule_id}_{name}",
                        rule_name=self.rule_name,
                        severity=SeverityLevel.ERROR,
                        message=f"Mandatory field '{name}' was not extracted",
                        affected_targets=[target],
                        source_signal="required_field_missing",
                    )
                )
        return issues


class CandidateConflictReviewRule(BaseReviewRule):
    """Flags fields with multiple ambiguous competing candidate extractions."""

    @property
    def rule_id(self) -> str:
        return "REV_CANDIDATE_CONFLICT"

    @property
    def rule_name(self) -> str:
        return "Conflicting Extraction Candidates Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for name, fc in field_confidences.items():
            if fc.signals and fc.signals.conflict_count > 0:
                bbox = fc.provenance.bounding_box if fc.provenance else None
                page = fc.provenance.page_number if fc.provenance else 1
                target = ReviewTarget(
                    target_type=ReviewTargetType.FIELD,
                    page_number=page,
                    field_name=name,
                    bounding_box=bbox,
                    reason=f"Ambiguity: {fc.signals.conflict_count} conflicting alternative candidate values detected",
                    severity=SeverityLevel.WARNING,
                )
                issues.append(
                    ReviewIssue(
                        issue_id=f"{self.rule_id}_{name}",
                        rule_name=self.rule_name,
                        severity=SeverityLevel.WARNING,
                        message=f"Field '{name}' has conflicting candidate values",
                        affected_targets=[target],
                        source_signal="conflict_count",
                    )
                )
        return issues


class TableArithmeticReviewRule(BaseReviewRule):
    """Flags tables and line items that fail arithmetic cross-checks."""

    @property
    def rule_id(self) -> str:
        return "REV_TABLE_ARITHMETIC_FAILURE"

    @property
    def rule_name(self) -> str:
        return "Table Arithmetic & Sum Integrity Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        for tc in table_confidences:
            for lic in tc.line_item_confidences:
                if lic.is_valid_arithmetic is False:
                    bbox = lic.provenance.bounding_box if lic.provenance else None
                    page = lic.provenance.page_number if lic.provenance else tc.page_number
                    target = ReviewTarget(
                        target_type=ReviewTargetType.LINE_ITEM,
                        page_number=page,
                        table_id=tc.table_id,
                        row_index=lic.row_index,
                        bounding_box=bbox,
                        reason=f"Line item row {lic.row_index + 1} arithmetic check failed",
                        severity=SeverityLevel.CRITICAL,
                    )
                    issues.append(
                        ReviewIssue(
                            issue_id=f"{self.rule_id}_{tc.table_id}_row_{lic.row_index}",
                            rule_name=self.rule_name,
                            severity=SeverityLevel.CRITICAL,
                            message=f"Line item at row {lic.row_index + 1} has mathematical calculation error",
                            affected_targets=[target],
                            source_signal="line_item_arithmetic",
                        )
                    )

            if tc.validation_status == ValidationStatus.INVALID:
                target = ReviewTarget(
                    target_type=ReviewTargetType.TABLE,
                    page_number=tc.page_number,
                    table_id=tc.table_id,
                    bounding_box=tc.bounding_box,
                    reason=f"Table '{tc.table_id}' failed structural or sum validation: {'; '.join(tc.review_reasons)}",
                    severity=SeverityLevel.CRITICAL,
                )
                issues.append(
                    ReviewIssue(
                        issue_id=f"{self.rule_id}_{tc.table_id}",
                        rule_name=self.rule_name,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Table '{tc.table_id}' integrity validation failed",
                        affected_targets=[target],
                        source_signal="table_validation",
                    )
                )
        return issues


class ValidationReportReviewRule(BaseReviewRule):
    """Directly bridges Phase 7 validation issues into review issues with spatial targets."""

    @property
    def rule_id(self) -> str:
        return "REV_VALIDATION_REPORT_ISSUES"

    @property
    def rule_name(self) -> str:
        return "Validation Suite Findings Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []
        if not validation_report:
            return issues

        for val_issue in validation_report.issues:
            if val_issue.status in (ValidationStatus.INVALID, ValidationStatus.WARNING):
                targets: List[ReviewTarget] = []
                for f_name in val_issue.affected_fields:
                    fc = field_confidences.get(f_name)
                    bbox = val_issue.bounding_box or (fc.provenance.bounding_box if fc and fc.provenance else None)
                    page = val_issue.page_number or (fc.provenance.page_number if fc and fc.provenance else 1)
                    targets.append(
                        ReviewTarget(
                            target_type=ReviewTargetType.FIELD,
                            page_number=page,
                            field_name=f_name,
                            bounding_box=bbox,
                            reason=val_issue.message,
                            severity=val_issue.severity,
                        )
                    )

                if not targets:
                    targets.append(
                        ReviewTarget(
                            target_type=ReviewTargetType.DOCUMENT,
                            page_number=val_issue.page_number or 1,
                            bounding_box=val_issue.bounding_box,
                            reason=val_issue.message,
                            severity=val_issue.severity,
                        )
                    )

                issues.append(
                    ReviewIssue(
                        issue_id=f"{self.rule_id}_{val_issue.rule_id}",
                        rule_name=val_issue.rule_name,
                        severity=val_issue.severity,
                        message=val_issue.message,
                        affected_targets=targets,
                        source_signal="validation_report",
                    )
                )
        return issues


class UnclassifiedDocumentReviewRule(BaseReviewRule):
    """Flags documents with UNKNOWN or unconfident document classification."""

    @property
    def rule_id(self) -> str:
        return "REV_UNCLASSIFIED_DOCUMENT"

    @property
    def rule_name(self) -> str:
        return "Document Classification Confidence Check"

    def evaluate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        issues: List[ReviewIssue] = []

        if document.classified_type == DocumentType.UNKNOWN:
            target = ReviewTarget(
                target_type=ReviewTargetType.DOCUMENT,
                page_number=1,
                reason="Document could not be deterministically classified into a known category",
                severity=SeverityLevel.ERROR,
            )
            issues.append(
                ReviewIssue(
                    issue_id=f"{self.rule_id}_unknown",
                    rule_name=self.rule_name,
                    severity=SeverityLevel.ERROR,
                    message="Document type is UNKNOWN and requires manual classification",
                    affected_targets=[target],
                    source_signal="classification_type",
                )
            )
        return issues


class ReviewRuleCatalog:
    """Catalog holding default and custom human review evaluation rules."""

    def __init__(self, custom_rules: Optional[List[BaseReviewRule]] = None) -> None:
        self._rules: List[BaseReviewRule] = custom_rules if custom_rules is not None else self._build_default_rules()

    @staticmethod
    def _build_default_rules() -> List[BaseReviewRule]:
        return [
            LowFieldConfidenceReviewRule(),
            LowOCRConfidenceReviewRule(),
            MissingRequiredFieldReviewRule(),
            CandidateConflictReviewRule(),
            TableArithmeticReviewRule(),
            ValidationReportReviewRule(),
            UnclassifiedDocumentReviewRule(),
        ]

    @property
    def rules(self) -> List[BaseReviewRule]:
        """List of active review rules."""
        return list(self._rules)

    def evaluate_all(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        document_confidence: DocumentConfidence,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[ReviewRoutingConfig] = None,
    ) -> List[ReviewIssue]:
        """Execute all rules and collect deduplicated review issues."""
        all_issues: List[ReviewIssue] = []
        seen_issue_ids = set()

        for rule in self._rules:
            issues = rule.evaluate(
                document=document,
                field_confidences=field_confidences,
                table_confidences=table_confidences,
                document_confidence=document_confidence,
                validation_report=validation_report,
                config=config,
            )
            for issue in issues:
                if issue.issue_id not in seen_issue_ids:
                    seen_issue_ids.add(issue.issue_id)
                    all_issues.append(issue)

        return all_issues
