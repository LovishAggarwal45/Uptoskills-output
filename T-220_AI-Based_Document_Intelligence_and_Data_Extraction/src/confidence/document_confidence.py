"""Document-level composite confidence synthesis and risk aggregation."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.config import DocumentConfidenceConfig
from src.core.models import Document
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.confidence.aggregator import (
    aggregate_weighted_signals,
    compute_confidence_band,
)
from src.confidence.models import (
    ConfidenceBand,
    DocumentConfidence,
    FieldConfidence,
    TableConfidence,
)
from src.confidence.signals import extract_document_signals
from src.validation.models import ValidationReport


class DocumentConfidenceCalculator:
    """Aggregates multi-subsystem confidence dimensions into an explainable document-level score."""

    def __init__(self, config: Optional[DocumentConfidenceConfig] = None) -> None:
        self.config = config or DocumentConfidenceConfig()

    def calculate(
        self,
        document: Document,
        field_confidences: Dict[str, FieldConfidence],
        table_confidences: List[TableConfidence],
        validation_report: Optional[ValidationReport] = None,
    ) -> DocumentConfidence:
        """Synthesize document confidence and determine document-level review status.

        CRITICAL ARCHITECTURAL RULE:
        Document confidence is NOT the same as document validity. A document with
        0.95 confidence but a critical validation failure MUST still trigger review.

        Args:
            document: Ingested and classified document aggregate.
            field_confidences: Map of evaluated field confidences.
            table_confidences: List of evaluated table confidences.
            validation_report: Optional validation report.

        Returns:
            DocumentConfidence with composite score, component breakdown, and review decision.
        """
        signals = extract_document_signals(
            document=document,
            fields={},  # already processed into field_confidences
            tables=[],
            validation_report=validation_report,
        )

        # 1. Compute Mean Field Confidence
        if field_confidences:
            mean_fld_conf = sum(fc.aggregated_confidence for fc in field_confidences.values()) / len(field_confidences)
        else:
            mean_fld_conf = 0.0

        # 2. Compute Mean Table Confidence (None if no tables exist)
        if table_confidences:
            mean_tbl_conf: Optional[float] = sum(tc.aggregated_confidence for tc in table_confidences) / len(table_confidences)
        else:
            mean_tbl_conf = None

        # 3. Validation Score
        val_score = float(validation_report.validation_score) if validation_report else 1.0
        val_status = validation_report.overall_status if validation_report else ValidationStatus.UNVALIDATED

        # 4. Prepare Weighted Signal Pairs
        signal_pairs = [
            (signals.classification_confidence, self.config.classification_weight),
            (mean_fld_conf, self.config.field_weight),
            (mean_tbl_conf, self.config.table_weight if mean_tbl_conf is not None else 0.0),
            (val_score, self.config.validation_weight),
        ]

        # 5. Aggregate Weighted Score
        overall_score, raw_breakdown = aggregate_weighted_signals(signal_pairs)

        # 6. Map to Confidence Band
        band = compute_confidence_band(
            score=overall_score,
            high_threshold=self.config.high_threshold,
            medium_threshold=self.config.medium_threshold,
            low_threshold=self.config.low_threshold,
        )

        # 7. Document-level Review Triggers
        review_required = False
        reasons: List[str] = []

        # Trigger A: Overall Confidence below Threshold
        if overall_score < self.config.review_confidence_threshold:
            review_required = True
            reasons.append(
                f"Overall document confidence ({overall_score:.2f}) below threshold ({self.config.review_confidence_threshold:.2f})"
            )

        # Trigger B: Critical / Error Validation Failures (Overrides high confidence!)
        if validation_report and validation_report.error_count > 0:
            review_required = True
            reasons.append(f"Document contains {validation_report.error_count} blocking validation error(s)")

        if val_status == ValidationStatus.INVALID:
            review_required = True
            reasons.append("Document validation status is INVALID")

        # Trigger C: Missing Required Fields
        missing_fields = [
            fc.field_name for fc in field_confidences.values()
            if fc.metadata.get("is_required") and (fc.value is None or str(fc.value).strip() == "")
        ]
        if missing_fields:
            review_required = True
            reasons.append(f"Mandatory required field(s) missing: {', '.join(missing_fields)}")

        # Trigger D: Unrecognized or Ambiguous Document Type
        if document.classified_type == DocumentType.UNKNOWN:
            review_required = True
            reasons.append("Document type is UNKNOWN / unclassified")

        # Trigger E: Individual Field or Table Review Failures
        fld_review_count = sum(1 for fc in field_confidences.values() if fc.review_required)
        if fld_review_count > 0:
            review_required = True
            reasons.append(f"{fld_review_count} extracted field(s) flagged for human review")

        tbl_review_count = sum(1 for tc in table_confidences if tc.review_required)
        if tbl_review_count > 0:
            review_required = True
            reasons.append(f"{tbl_review_count} table structure(s) flagged for human review")

        # 8. Construct Narrative Explanation
        breakdown_dict = {
            "classification": signals.classification_confidence or 0.0,
            "mean_field": mean_fld_conf,
            "mean_table": mean_tbl_conf if mean_tbl_conf is not None else 0.0,
            "validation_score": val_score,
            "overall_score": overall_score,
        }

        clf_desc = f"{signals.classification_confidence:.2f}" if signals.classification_confidence is not None else "N/A"
        tbl_desc = f"{mean_tbl_conf:.2f}" if mean_tbl_conf is not None else "N/A"
        explanation = (
            f"Classification: {clf_desc}, Mean Fields ({len(field_confidences)}): {mean_fld_conf:.2f}, "
            f"Mean Tables ({len(table_confidences)}): {tbl_desc}, Validation Score: {val_score:.2f} -> "
            f"Document Composite: {overall_score:.2f} [{band.value.upper()}]. "
            f"Review Required: {review_required}"
        )

        return DocumentConfidence(
            document_id=document.id,
            document_type=document.classified_type,
            overall_confidence=overall_score,
            confidence_band=band,
            classification_confidence=signals.classification_confidence,
            mean_field_confidence=mean_fld_conf,
            mean_table_confidence=mean_tbl_conf,
            mean_ocr_confidence=signals.ocr_confidence,
            validation_score=val_score,
            validation_status=val_status,
            review_required=review_required,
            review_reasons=reasons,
            field_confidences=field_confidences,
            table_confidences=table_confidences,
            component_breakdown=breakdown_dict,
            explanation=explanation,
            metadata={"total_pages": len(document.pages)},
        )
