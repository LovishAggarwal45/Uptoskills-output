"""Signal extraction and normalization for multi-dimensional confidence assessment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.models import Document, ExtractedField
from src.core.types import SeverityLevel, ValidationStatus
from src.confidence.models import ConfidenceSignals
from src.tables.models import LineItem, Table
from src.validation.models import ValidationIssue, ValidationReport


def extract_field_signals(
    field: ExtractedField,
    validation_report: Optional[ValidationReport] = None,
) -> ConfidenceSignals:
    """Extract raw, non-fabricated confidence signals for an individual extracted field.

    Args:
        field: Extracted field model instance.
        validation_report: Optional document-level validation report for cross-referencing.

    Returns:
        ConfidenceSignals populated strictly with observed signals (None for missing data).
    """
    # 1. OCR Confidence from Provenance
    ocr_conf: Optional[float] = None
    if field.provenance and field.provenance.ocr_confidence is not None:
        ocr_conf = float(field.provenance.ocr_confidence)
    elif field.source_ocr_confidence is not None:
        ocr_conf = float(field.source_ocr_confidence)

    # 2. Extraction Confidence
    ext_conf: float = float(field.extraction_confidence)

    # 3. Validation Status and Field-specific Issues
    val_status = field.validation_status
    issue_count = 0
    error_count = 0
    warning_count = 0

    if validation_report:
        field_issues = validation_report.get_issues_for_field(field.name)
        issue_count = len(field_issues)
        error_count = sum(
            1 for i in field_issues
            if i.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL) or i.status == ValidationStatus.INVALID
        )
        warning_count = sum(
            1 for i in field_issues
            if i.severity == SeverityLevel.WARNING or i.status == ValidationStatus.WARNING
        )
        if error_count > 0:
            val_status = ValidationStatus.INVALID
        elif warning_count > 0 and val_status != ValidationStatus.INVALID:
            val_status = ValidationStatus.WARNING

    # 4. Candidate Conflicts
    conflict_count = 0
    if field.candidates and len(field.candidates) > 1:
        # Check if alternative candidates have different normalized values
        distinct_vals = {
            str(c.get("normalized_value") if isinstance(c, dict) else getattr(c, "normalized_value", None))
            for c in field.candidates
        }
        if len(distinct_vals) > 1:
            conflict_count = len(field.candidates) - 1

    # 5. Required Field Missing
    is_missing = False
    if field.is_required:
        if field.value is None or str(field.value).strip() == "":
            is_missing = True

    return ConfidenceSignals(
        ocr_confidence=ocr_conf,
        extraction_confidence=ext_conf,
        validation_status=val_status,
        validation_issue_count=issue_count,
        validation_error_count=error_count,
        validation_warning_count=warning_count,
        conflict_count=conflict_count,
        required_field_missing=is_missing,
        metadata={"field_type": field.field_type.value, "is_ambiguous": field.is_ambiguous},
    )


def extract_table_signals(
    table: Table,
    validation_report: Optional[ValidationReport] = None,
) -> ConfidenceSignals:
    """Extract raw confidence signals for a reconstructed table aggregate."""
    struct_conf = table.confidence_breakdown.get("structure_confidence", table.confidence)
    row_conf = table.confidence_breakdown.get("row_confidence", table.confidence)
    cell_conf = table.confidence_breakdown.get("cell_confidence", table.confidence)

    val_status = table.validation_status
    error_cnt = 0
    warning_cnt = 0
    issue_cnt = 0

    if table.validation_result:
        if not table.validation_result.is_valid:
            val_status = ValidationStatus.INVALID
            error_cnt += table.validation_result.row_checks_failed
            if table.validation_result.subtotal_valid is False:
                error_cnt += 1
            if table.validation_result.grand_total_valid is False:
                error_cnt += 1
        issue_cnt = len(table.validation_result.discrepancies) + len(table.validation_result.messages)

    if validation_report:
        tbl_issues = validation_report.get_issues_by_category(
            getattr(validation_report, "ValidationCategory", None) or "table"
        )
        if tbl_issues:
            issue_cnt += len(tbl_issues)
            error_cnt += sum(1 for i in tbl_issues if i.severity in (SeverityLevel.ERROR, SeverityLevel.CRITICAL))
            warning_cnt += sum(1 for i in tbl_issues if i.severity == SeverityLevel.WARNING)

    return ConfidenceSignals(
        table_confidence=table.confidence,
        validation_status=val_status,
        validation_issue_count=issue_cnt,
        validation_error_count=error_cnt,
        validation_warning_count=warning_cnt,
        metadata={
            "structure_confidence": struct_conf,
            "row_confidence": row_conf,
            "cell_confidence": cell_conf,
            "row_count": table.row_count,
            "column_count": table.column_count,
        },
    )


def extract_document_signals(
    document: Document,
    fields: Dict[str, ExtractedField],
    tables: List[Table],
    validation_report: Optional[ValidationReport] = None,
) -> ConfidenceSignals:
    """Extract document-level raw signals across classification, OCR, fields, and validation."""
    clf_conf = document.classification_confidence

    val_status = ValidationStatus.UNVALIDATED
    val_score = 1.0
    val_issues = 0
    val_errors = 0
    val_warnings = 0

    if validation_report:
        val_status = validation_report.overall_status
        val_score = validation_report.validation_score
        val_issues = len(validation_report.issues)
        val_errors = validation_report.error_count
        val_warnings = validation_report.warning_count

    # Check for missing required fields across document
    missing_required = any(
        f.is_required and (f.value is None or str(f.value).strip() == "")
        for f in fields.values()
    )

    # Compute mean OCR confidence from pages if available
    ocr_confs: List[float] = []
    for p in document.pages:
        for tr in p.ocr_text_regions:
            if tr.confidence is not None:
                ocr_confs.append(tr.confidence)
        if "ocr_words" in p.metadata:
            for w in p.metadata["ocr_words"]:
                if hasattr(w, "confidence") and w.confidence is not None:
                    ocr_confs.append(w.confidence)

    mean_ocr = sum(ocr_confs) / len(ocr_confs) if ocr_confs else None

    # Source quality if present in metadata
    quality_score: Optional[float] = None
    if document.pages and "quality_metrics" in document.pages[0].metadata:
        metrics = document.pages[0].metadata["quality_metrics"]
        if isinstance(metrics, dict) and "overall_quality" in metrics:
            quality_score = float(metrics["overall_quality"])

    return ConfidenceSignals(
        ocr_confidence=mean_ocr,
        classification_confidence=clf_conf,
        validation_status=val_status,
        validation_score=val_score,
        validation_issue_count=val_issues,
        validation_error_count=val_errors,
        validation_warning_count=val_warnings,
        required_field_missing=missing_required,
        source_quality_score=quality_score,
        metadata={
            "page_count": len(document.pages),
            "field_count": len(fields),
            "table_count": len(tables),
        },
    )
