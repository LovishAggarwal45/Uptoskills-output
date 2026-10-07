"""Confidence scoring framework and review routing contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from src.core.config import ConfidenceConfig
from src.core.models import (
    Document,
    ExtractedField,
    ExtractedTable,
    ReviewFlag,
    ValidationResult,
)
from src.core.types import ConfidenceSource
from src.confidence.models import ReviewRoutingResult
from src.tables.models import Table
from src.validation.models import ValidationReport


@dataclass
class ConfidenceEvaluation:
    """Consolidated confidence score and human-in-the-loop review routing decision."""
    overall_score: float
    confidence_source: ConfidenceSource
    mean_ocr_confidence: Optional[float] = None
    mean_extraction_confidence: float = 0.0
    validation_penalty: float = 0.0
    review_required: bool = False
    review_flags: List[ReviewFlag] = field(default_factory=list)
    field_scores: Dict[str, float] = field(default_factory=dict)
    routing_result: Optional[ReviewRoutingResult] = None


class BaseConfidenceEngine(ABC):
    """Abstract interface for computing multi-dimensional confidence and routing flags."""

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        self.config = config or ConfidenceConfig()

    @abstractmethod
    def evaluate(
        self,
        document: Document,
        fields: Dict[str, ExtractedField],
        tables: Optional[List[Union[ExtractedTable, Table]]] = None,
        validation_results: Optional[Union[List[ValidationResult], ValidationReport]] = None,
    ) -> ConfidenceEvaluation:
        """Synthesize OCR confidence, extraction confidence, and validation findings.

        Args:
            document: Processed document aggregate.
            fields: Map of extracted fields.
            tables: List of extracted tables.
            validation_results: List of all validation outcomes or ValidationReport.

        Returns:
            ConfidenceEvaluation with composite score, component metrics,
            and review flags.
        """
        pass


class ConfidenceEngine(BaseConfidenceEngine):
    """Concrete production-grade engine implementing multi-tier scoring and human review routing."""

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        super().__init__(config=config)
        from src.confidence.review_router import ReviewRouter
        self.router = ReviewRouter(config=self.config)

    def evaluate(
        self,
        document: Document,
        fields: Dict[str, ExtractedField],
        tables: Optional[List[Union[ExtractedTable, Table]]] = None,
        validation_results: Optional[Union[List[ValidationResult], ValidationReport]] = None,
    ) -> ConfidenceEvaluation:
        """Execute full confidence evaluation and review routing."""
        tables_list: List[Table] = []
        if tables:
            for t in tables:
                if isinstance(t, Table):
                    tables_list.append(t)
                elif isinstance(t, ExtractedTable):
                    # Convert Core ExtractedTable to Table if needed
                    tables_list.append(
                        Table(
                            table_id=t.table_id,
                            page_number=t.page_number,
                            headers=t.headers,
                            confidence=t.confidence,
                            validation_status=t.validation_status,
                        )
                    )

        # Handle validation input
        val_report: Optional[ValidationReport] = None
        if isinstance(validation_results, ValidationReport):
            val_report = validation_results
        elif isinstance(validation_results, list):
            # Create minimal report from list of ValidationResult
            from src.validation.models import ValidationIssue, ValidationReport
            issues = [
                i if isinstance(i, ValidationIssue) else ValidationIssue(
                    rule_id=i.rule_id,
                    rule_name=i.rule_name,
                    status=i.status,
                    severity=i.severity,
                    message=i.message,
                    affected_fields=i.affected_fields,
                    timestamp=i.timestamp,
                )
                for i in validation_results
            ]
            val_report = ValidationReport(
                document_id=document.id,
                document_type=document.classified_type,
                issues=issues,
            )

        routing_result = self.router.route_document(
            document=document,
            fields=fields,
            tables=tables_list,
            validation_report=val_report,
        )

        mean_ocr = routing_result.document_confidence.mean_ocr_confidence
        mean_ext = (
            sum(fc.original_extraction_confidence for fc in routing_result.field_confidences.values()) / len(fields)
            if fields else 0.0
        )
        val_penalty = 1.0 - routing_result.document_confidence.validation_score

        return ConfidenceEvaluation(
            overall_score=routing_result.document_confidence.overall_confidence,
            confidence_source=ConfidenceSource.COMPOSITE,
            mean_ocr_confidence=mean_ocr,
            mean_extraction_confidence=round(mean_ext, 4),
            validation_penalty=round(val_penalty, 4),
            review_required=routing_result.review_required,
            review_flags=routing_result.review_flags,
            field_scores={k: v.aggregated_confidence for k, v in routing_result.field_confidences.items()},
            routing_result=routing_result,
        )


__all__ = [
    "ConfidenceEvaluation",
    "BaseConfidenceEngine",
    "ConfidenceEngine",
]
