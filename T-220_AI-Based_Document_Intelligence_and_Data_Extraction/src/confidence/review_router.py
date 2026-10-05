"""Human review router orchestrating risk assessment, review queue items, and STP decisions."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.config import ConfidenceConfig
from src.core.logging import get_logger
from src.core.models import Document, ExtractedField, ReviewFlag
from src.core.types import ReviewTriggerType, SeverityLevel, ValidationStatus
from src.confidence.document_confidence import DocumentConfidenceCalculator
from src.confidence.field_confidence import FieldConfidenceCalculator
from src.confidence.models import (
    DocumentConfidence,
    FieldConfidence,
    ReviewIssue,
    ReviewPriority,
    ReviewQueueItem,
    ReviewRoutingResult,
    ReviewStatus,
    ReviewTarget,
    TableConfidence,
)
from src.confidence.priority import PriorityEvaluator
from src.confidence.review_rules import ReviewRuleCatalog
from src.confidence.table_confidence import TableConfidenceCalculator
from src.tables.models import Table
from src.validation.models import ValidationReport

logger = get_logger("confidence.router")


class ReviewRouter:
    """Orchestrates multi-tier confidence evaluation and routes documents to review or STP."""

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        self.config = config or ConfidenceConfig()
        self.field_calculator = FieldConfidenceCalculator(config=self.config.field)
        self.table_calculator = TableConfidenceCalculator(config=self.config.table)
        self.doc_calculator = DocumentConfidenceCalculator(config=self.config.document)
        self.rule_catalog = ReviewRuleCatalog()
        self.priority_evaluator = PriorityEvaluator(config=self.config.priority)

    def route_document(
        self,
        document: Document,
        fields: Dict[str, ExtractedField],
        tables: Optional[List[Table]] = None,
        validation_report: Optional[ValidationReport] = None,
    ) -> ReviewRoutingResult:
        """Execute complete multi-tier confidence evaluation and generate human review routing.

        Args:
            document: Processed and classified document aggregate.
            fields: Map of extracted fields from Phase 5.
            tables: List of reconstructed tables from Phase 6.
            validation_report: Validation outcomes from Phase 7.

        Returns:
            ReviewRoutingResult containing field/table/document confidences and review queue item.
        """
        tables = tables or []
        logger.info(f"Initiating confidence scoring and review routing for document '{document.id}'")

        # 1. Field-Level Confidence
        field_confs: Dict[str, FieldConfidence] = {}
        for name, field_obj in fields.items():
            fc = self.field_calculator.calculate(field_obj, validation_report)
            field_confs[name] = fc
        logger.debug(f"Calculated confidence for {len(field_confs)} fields")

        # 2. Table-Level Confidence
        table_confs: List[TableConfidence] = []
        for tbl in tables:
            tc = self.table_calculator.calculate_table(tbl, validation_report)
            table_confs.append(tc)
        logger.debug(f"Calculated confidence for {len(table_confs)} tables")

        # 3. Document-Level Composite Confidence
        doc_conf = self.doc_calculator.calculate(
            document=document,
            field_confidences=field_confs,
            table_confidences=table_confs,
            validation_report=validation_report,
        )
        logger.info(
            f"Document '{document.id}' composite confidence: {doc_conf.overall_confidence:.4f} "
            f"[{doc_conf.confidence_band.value.upper()}], Review Required: {doc_conf.review_required}"
        )

        # 4. Evaluate Review Rules and Collect Issues
        issues: List[ReviewIssue] = self.rule_catalog.evaluate_all(
            document=document,
            field_confidences=field_confs,
            table_confidences=table_confs,
            document_confidence=doc_conf,
            validation_report=validation_report,
            config=self.config.review,
        )

        # 5. Collect All Review Targets
        targets: List[ReviewTarget] = []
        for issue in issues:
            targets.extend(issue.affected_targets)

        # 6. Evaluate Queue Priority
        priority = self.priority_evaluator.evaluate(issues, doc_conf)

        # 7. Straight-Through Processing (STP) Eligibility
        is_straight_through = (not doc_conf.review_required) and (len(issues) == 0)
        status = ReviewStatus.AUTO_APPROVED if is_straight_through else ReviewStatus.PENDING

        # 8. Consolidate Human-Readable Reasons
        reasons = list(doc_conf.review_reasons)
        for tc in table_confs:
            for r in tc.review_reasons:
                if r not in reasons:
                    reasons.append(r)
        for issue in issues:
            if issue.message not in reasons:
                reasons.append(issue.message)

        # 9. Build Review Queue Item
        queue_item = ReviewQueueItem(
            document_id=document.id,
            document_type=document.classified_type,
            priority=priority,
            status=status,
            document_confidence=doc_conf.overall_confidence,
            confidence_band=doc_conf.confidence_band,
            validation_status=doc_conf.validation_status,
            review_required=not is_straight_through,
            reasons=reasons,
            issues=issues,
            targets=targets,
            recommended_action=(
                "Eligible for automated straight-through processing"
                if is_straight_through
                else f"Review {len(targets)} flagged target(s) with priority [{priority.value.upper()}]"
            ),
            metadata={
                "issue_count": len(issues),
                "target_count": len(targets),
                "is_straight_through": is_straight_through,
            },
        )

        # 10. Generate Legacy ReviewFlags for compatibility
        review_flags: List[ReviewFlag] = []
        for issue in issues:
            affected_field = None
            for t in issue.affected_targets:
                if t.field_name:
                    affected_field = t.field_name
                    break

            review_flags.append(
                ReviewFlag(
                    reason=issue.message,
                    severity=issue.severity,
                    affected_field=affected_field,
                    message=issue.message,
                    trigger_type=self._map_signal_to_trigger_type(issue.source_signal),
                )
            )

        logger.info(
            f"Review routing completed for '{document.id}': Status={status.value.upper()}, "
            f"Priority={priority.value.upper()}, Issues={len(issues)}, Targets={len(targets)}"
        )

        return ReviewRoutingResult(
            document_id=document.id,
            is_straight_through=is_straight_through,
            document_confidence=doc_conf,
            queue_item=queue_item,
            field_confidences=field_confs,
            table_confidences=table_confs,
            review_flags=review_flags,
        )

    @staticmethod
    def _map_signal_to_trigger_type(source_signal: str) -> ReviewTriggerType:
        """Helper to map issue source signals to domain ReviewTriggerType."""
        if "ocr" in source_signal:
            return ReviewTriggerType.LOW_OCR_CONFIDENCE
        elif "extraction" in source_signal or "aggregated" in source_signal:
            return ReviewTriggerType.LOW_EXTRACTION_CONFIDENCE
        elif "required" in source_signal:
            return ReviewTriggerType.MISSING_REQUIRED_FIELD
        elif "arithmetic" in source_signal:
            return ReviewTriggerType.ARITHMETIC_MISMATCH
        elif "classification" in source_signal:
            return ReviewTriggerType.UNRECOGNIZED_DOCUMENT_TYPE
        else:
            return ReviewTriggerType.VALIDATION_FAILURE
