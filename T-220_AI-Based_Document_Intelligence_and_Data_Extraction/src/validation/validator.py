"""Main document validation and consistency engine coordinator."""

import time
from typing import Any, Dict, List, Optional, Union

from src.core.config import ValidationConfig
from src.core.models import (
    ExtractedField,
    ExtractedTable,
    ValidationResult as CoreValidationResult,
)
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.extraction.models import ExtractionResult
from src.tables.models import Table, TableExtractionResult
from src.validation.base import BaseValidationEngine, BaseValidationRule
from src.validation.confidence import ValidationConfidenceCalculator
from src.validation.models import ValidationCategory, ValidationIssue, ValidationReport
from src.validation.rules import RuleRegistry


class DocumentValidationEngine(BaseValidationEngine):
    """Production-quality orchestrator for deterministic and heuristic document validation."""

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        auto_load_default_rules: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.confidence_calculator = ValidationConfidenceCalculator()
        if auto_load_default_rules:
            self.rules = RuleRegistry.build_default_suite(self.config)

    def validate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CoreValidationResult]:
        """Execute validation across fields and tables, updating field statuses in-place.

        Args:
            fields: Map of field name to ExtractedField instance.
            tables: List of extracted tables.
            document_type: Classified document category.
            context: Optional execution context dictionary.

        Returns:
            List of CoreValidationResult (or ValidationIssue) instances.
        """
        all_issues: List[ValidationIssue] = []
        ctx = context or {}

        for rule in self.rules:
            try:
                rule_issues = rule.evaluate(fields, tables, document_type, ctx)
                all_issues.extend(rule_issues)
            except Exception as e:
                all_issues.append(
                    ValidationIssue(
                        rule_id=rule.rule_id,
                        rule_name=rule.rule_name,
                        status=ValidationStatus.INVALID,
                        severity=SeverityLevel.ERROR,
                        message=f"Rule '{rule.rule_name}' failed to execute: {str(e)}",
                        category=rule.category,
                        affected_fields=rule.target_fields,
                    )
                )

        # Update field-level validation status and messages in-place
        self._update_field_statuses(fields, all_issues)
        return all_issues

    def validate_document(
        self,
        document_id: str,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """Execute complete document validation and produce a comprehensive ValidationReport.

        Args:
            document_id: Unique document identifier.
            fields: Map of field name to ExtractedField.
            tables: List of extracted tables.
            document_type: Semantic document type.
            context: Optional execution context.

        Returns:
            ValidationReport instance with aggregated score and issue breakdown.
        """
        start_time = time.perf_counter()
        issues = self.validate(fields, tables, document_type, context)
        elapsed_sec = time.perf_counter() - start_time

        # Calculate validation score & breakdown
        score_data = self.confidence_calculator.calculate_score(issues)

        # Determine overall document validation status
        overall_status = ValidationStatus.VALID
        has_errors = any(
            i.status == ValidationStatus.INVALID and i.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL)
            for i in issues
        )
        has_warnings = any(
            i.status == ValidationStatus.WARNING or (i.status == ValidationStatus.INVALID and i.severity == SeverityLevel.WARNING)
            for i in issues
        )

        if has_errors:
            overall_status = ValidationStatus.INVALID
        elif has_warnings:
            overall_status = ValidationStatus.WARNING

        field_statuses = {
            name: field_obj.validation_status
            for name, field_obj in fields.items()
        }

        # Collect discrepancies from invalid issues
        discrepancies: List[Dict[str, Any]] = []
        for issue in issues:
            if issue.status != ValidationStatus.VALID:
                discrepancies.append({
                    "rule_id": issue.rule_id,
                    "severity": issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity),
                    "message": issue.message,
                    "affected_fields": issue.affected_fields,
                    "expected": issue.expected_value,
                    "actual": issue.actual_value,
                })

        return ValidationReport(
            document_id=document_id,
            document_type=document_type,
            overall_status=overall_status,
            validation_score=score_data["score"],
            rules_evaluated=score_data["rules_evaluated"],
            rules_passed=score_data["rules_passed"],
            rules_failed=score_data["rules_failed"],
            rules_warning=score_data["rules_warning"],
            issues=issues,
            field_statuses=field_statuses,
            discrepancies=discrepancies,
            execution_time_seconds=elapsed_sec,
            metadata={
                "score_breakdown": score_data,
                "strict_mode": getattr(self.config, "strict_mode", False),
            },
        )

    def validate_extraction_result(
        self,
        extraction_result: ExtractionResult,
        table_result: Optional[TableExtractionResult] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """Validate an ExtractionResult and TableExtractionResult combination.

        Args:
            extraction_result: Extraction result from Phase 5.
            table_result: Table extraction result from Phase 6.
            context: Additional runtime context.

        Returns:
            ValidationReport instance.
        """
        ctx = dict(context or {})
        core_tables: List[ExtractedTable] = []
        rich_tables: List[Table] = []

        if table_result:
            rich_tables = table_result.tables
            core_tables = [t.to_extracted_table() for t in rich_tables]
            ctx["rich_tables"] = rich_tables
            ctx["line_items"] = table_result.line_items

        return self.validate_document(
            document_id=extraction_result.document_id,
            fields=extraction_result.fields,
            tables=core_tables,
            document_type=extraction_result.document_type,
            context=ctx,
        )

    def _update_field_statuses(
        self,
        fields: Dict[str, ExtractedField],
        issues: List[ValidationIssue],
    ) -> None:
        """Update ExtractedField validation statuses and validation messages in-place."""
        for issue in issues:
            for field_name in issue.affected_fields:
                if field_name not in fields:
                    continue
                field_obj = fields[field_name]

                if issue.status == ValidationStatus.INVALID:
                    field_obj.validation_status = ValidationStatus.INVALID
                    if issue.message not in field_obj.validation_messages:
                        field_obj.validation_messages.append(issue.message)
                elif issue.status == ValidationStatus.WARNING:
                    if field_obj.validation_status != ValidationStatus.INVALID:
                        field_obj.validation_status = ValidationStatus.WARNING
                    if issue.message not in field_obj.validation_messages:
                        field_obj.validation_messages.append(issue.message)
                elif issue.status == ValidationStatus.VALID:
                    if field_obj.validation_status == ValidationStatus.UNVALIDATED:
                        field_obj.validation_status = ValidationStatus.VALID


__all__ = ["DocumentValidationEngine"]
