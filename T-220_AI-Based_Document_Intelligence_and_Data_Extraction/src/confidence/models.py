"""Typed domain models and data structures for confidence scoring and human review routing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.models import BoundingBox, Provenance, ReviewFlag
from src.core.types import (
    ConfidenceSource,
    DocumentType,
    ExtractionMethod,
    FieldType,
    ReviewTriggerType,
    SeverityLevel,
    ValidationStatus,
)


class ConfidenceBand(str, Enum):
    """Categorical quality bands for deterministic confidence scores."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ReviewPriority(str, Enum):
    """Urgency priority assigned to human review queue items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ReviewStatus(str, Enum):
    """Lifecycle status for a review queue entry."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class ReviewTargetType(str, Enum):
    """Spatial and structural entity type requiring human reviewer attention."""
    DOCUMENT = "document"
    PAGE = "page"
    FIELD = "field"
    TABLE = "table"
    LINE_ITEM = "line_item"
    CELL = "cell"


@dataclass
class ConfidenceSignals:
    """Explicit collection of raw, unaggregated confidence and quality signals."""
    ocr_confidence: Optional[float] = None
    extraction_confidence: Optional[float] = None
    table_confidence: Optional[float] = None
    classification_confidence: Optional[float] = None
    validation_status: Optional[ValidationStatus] = None
    validation_score: Optional[float] = None
    validation_issue_count: int = 0
    validation_error_count: int = 0
    validation_warning_count: int = 0
    conflict_count: int = 0
    required_field_missing: bool = False
    source_quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for val, name in [
            (self.ocr_confidence, "ocr_confidence"),
            (self.extraction_confidence, "extraction_confidence"),
            (self.table_confidence, "table_confidence"),
            (self.classification_confidence, "classification_confidence"),
            (self.validation_score, "validation_score"),
            (self.source_quality_score, "source_quality_score"),
        ]:
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"Confidence signal {name}={val} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize confidence signals to dictionary."""
        return {
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "extraction_confidence": round(self.extraction_confidence, 4) if self.extraction_confidence is not None else None,
            "table_confidence": round(self.table_confidence, 4) if self.table_confidence is not None else None,
            "classification_confidence": round(self.classification_confidence, 4) if self.classification_confidence is not None else None,
            "validation_status": self.validation_status.value if self.validation_status else None,
            "validation_score": round(self.validation_score, 4) if self.validation_score is not None else None,
            "validation_issue_count": self.validation_issue_count,
            "validation_error_count": self.validation_error_count,
            "validation_warning_count": self.validation_warning_count,
            "conflict_count": self.conflict_count,
            "required_field_missing": self.required_field_missing,
            "source_quality_score": round(self.source_quality_score, 4) if self.source_quality_score is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConfidenceSignals:
        """Construct ConfidenceSignals from dictionary."""
        val_status_raw = data.get("validation_status")
        val_status = ValidationStatus(val_status_raw) if val_status_raw else None
        return cls(
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            extraction_confidence=float(data["extraction_confidence"]) if data.get("extraction_confidence") is not None else None,
            table_confidence=float(data["table_confidence"]) if data.get("table_confidence") is not None else None,
            classification_confidence=float(data["classification_confidence"]) if data.get("classification_confidence") is not None else None,
            validation_status=val_status,
            validation_score=float(data["validation_score"]) if data.get("validation_score") is not None else None,
            validation_issue_count=int(data.get("validation_issue_count", 0)),
            validation_error_count=int(data.get("validation_error_count", 0)),
            validation_warning_count=int(data.get("validation_warning_count", 0)),
            conflict_count=int(data.get("conflict_count", 0)),
            required_field_missing=bool(data.get("required_field_missing", False)),
            source_quality_score=float(data["source_quality_score"]) if data.get("source_quality_score") is not None else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class FieldConfidence:
    """Field-level confidence evaluation preserving multi-source signals and review triggers."""
    field_name: str
    value: Any
    normalized_value: Optional[Any] = None
    original_extraction_confidence: float = 0.0
    ocr_confidence: Optional[float] = None
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    aggregated_confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    signals: Optional[ConfidenceSignals] = None
    provenance: Optional[Provenance] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.aggregated_confidence <= 1.0):
            raise ValueError(f"Field aggregated_confidence {self.aggregated_confidence} must be in [0.0, 1.0]")
        if not (0.0 <= self.original_extraction_confidence <= 1.0):
            raise ValueError(f"Field original_extraction_confidence {self.original_extraction_confidence} must be in [0.0, 1.0]")
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"Field ocr_confidence {self.ocr_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize FieldConfidence to dictionary."""
        return {
            "field_name": self.field_name,
            "value": self.value,
            "normalized_value": (
                self.normalized_value.isoformat()
                if isinstance(self.normalized_value, datetime)
                else self.normalized_value
            ),
            "original_extraction_confidence": round(self.original_extraction_confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "validation_status": self.validation_status.value if hasattr(self.validation_status, "value") else str(self.validation_status),
            "aggregated_confidence": round(self.aggregated_confidence, 4),
            "confidence_band": self.confidence_band.value if hasattr(self.confidence_band, "value") else str(self.confidence_band),
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "signals": self.signals.to_dict() if self.signals else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FieldConfidence:
        """Construct FieldConfidence from dictionary."""
        sig_data = data.get("signals")
        prov_data = data.get("provenance")
        val_status_raw = data.get("validation_status", ValidationStatus.UNVALIDATED.value)
        try:
            val_status = ValidationStatus(val_status_raw)
        except ValueError:
            val_status = ValidationStatus.UNVALIDATED

        band_raw = data.get("confidence_band", ConfidenceBand.MEDIUM.value)
        try:
            band = ConfidenceBand(band_raw)
        except ValueError:
            band = ConfidenceBand.MEDIUM

        return cls(
            field_name=data["field_name"],
            value=data.get("value"),
            normalized_value=data.get("normalized_value"),
            original_extraction_confidence=float(data.get("original_extraction_confidence", 0.0)),
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            validation_status=val_status,
            aggregated_confidence=float(data.get("aggregated_confidence", 0.0)),
            confidence_band=band,
            review_required=bool(data.get("review_required", False)),
            review_reasons=list(data.get("review_reasons", [])),
            signals=ConfidenceSignals.from_dict(sig_data) if sig_data else None,
            provenance=Provenance.from_dict(prov_data) if prov_data else None,
            explanation=data.get("explanation", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LineItemConfidence:
    """Confidence evaluation for an individual tabular line item."""
    row_index: int
    description: str = ""
    aggregated_confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    original_confidence: float = 0.0
    ocr_confidence: Optional[float] = None
    is_valid_arithmetic: Optional[bool] = None
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    cell_confidences: Dict[str, float] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.aggregated_confidence <= 1.0):
            raise ValueError(f"LineItem aggregated_confidence {self.aggregated_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LineItemConfidence to dictionary."""
        return {
            "row_index": self.row_index,
            "description": self.description,
            "aggregated_confidence": round(self.aggregated_confidence, 4),
            "confidence_band": self.confidence_band.value,
            "original_confidence": round(self.original_confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "is_valid_arithmetic": self.is_valid_arithmetic,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "cell_confidences": {k: round(v, 4) for k, v in self.cell_confidences.items()},
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineItemConfidence:
        """Construct LineItemConfidence from dictionary."""
        prov_data = data.get("provenance")
        band_raw = data.get("confidence_band", ConfidenceBand.MEDIUM.value)
        try:
            band = ConfidenceBand(band_raw)
        except ValueError:
            band = ConfidenceBand.MEDIUM

        return cls(
            row_index=int(data["row_index"]),
            description=data.get("description", ""),
            aggregated_confidence=float(data.get("aggregated_confidence", 0.0)),
            confidence_band=band,
            original_confidence=float(data.get("original_confidence", 0.0)),
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            is_valid_arithmetic=data.get("is_valid_arithmetic"),
            review_required=bool(data.get("review_required", False)),
            review_reasons=list(data.get("review_reasons", [])),
            cell_confidences={k: float(v) for k, v in data.get("cell_confidences", {}).items()},
            provenance=Provenance.from_dict(prov_data) if prov_data else None,
        )


@dataclass
class TableConfidence:
    """Confidence evaluation for an entire reconstructed 2D table."""
    table_id: str
    page_number: int = 1
    structure_confidence: float = 0.0
    row_confidence: float = 0.0
    cell_confidence: float = 0.0
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    aggregated_confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    line_item_confidences: List[LineItemConfidence] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.aggregated_confidence <= 1.0):
            raise ValueError(f"Table aggregated_confidence {self.aggregated_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableConfidence to dictionary."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "structure_confidence": round(self.structure_confidence, 4),
            "row_confidence": round(self.row_confidence, 4),
            "cell_confidence": round(self.cell_confidence, 4),
            "validation_status": self.validation_status.value,
            "aggregated_confidence": round(self.aggregated_confidence, 4),
            "confidence_band": self.confidence_band.value,
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "line_item_confidences": [li.to_dict() for li in self.line_item_confidences],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableConfidence:
        """Construct TableConfidence from dictionary."""
        bbox_data = data.get("bounding_box")
        val_status_raw = data.get("validation_status", ValidationStatus.UNVALIDATED.value)
        try:
            val_status = ValidationStatus(val_status_raw)
        except ValueError:
            val_status = ValidationStatus.UNVALIDATED

        band_raw = data.get("confidence_band", ConfidenceBand.MEDIUM.value)
        try:
            band = ConfidenceBand(band_raw)
        except ValueError:
            band = ConfidenceBand.MEDIUM

        return cls(
            table_id=data["table_id"],
            page_number=int(data.get("page_number", 1)),
            structure_confidence=float(data.get("structure_confidence", 0.0)),
            row_confidence=float(data.get("row_confidence", 0.0)),
            cell_confidence=float(data.get("cell_confidence", 0.0)),
            validation_status=val_status,
            aggregated_confidence=float(data.get("aggregated_confidence", 0.0)),
            confidence_band=band,
            review_required=bool(data.get("review_required", False)),
            review_reasons=list(data.get("review_reasons", [])),
            line_item_confidences=[LineItemConfidence.from_dict(li) for li in data.get("line_item_confidences", [])],
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
        )


@dataclass
class DocumentConfidence:
    """Document-level composite confidence synthesis combining all subsystem dimensions."""
    document_id: str
    document_type: DocumentType = DocumentType.UNKNOWN
    overall_confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    classification_confidence: Optional[float] = None
    mean_field_confidence: float = 0.0
    mean_table_confidence: Optional[float] = None
    mean_ocr_confidence: Optional[float] = None
    validation_score: float = 1.0
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    field_confidences: Dict[str, FieldConfidence] = field(default_factory=dict)
    table_confidences: List[TableConfidence] = field(default_factory=list)
    component_breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(f"Document overall_confidence {self.overall_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DocumentConfidence to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value if hasattr(self.document_type, "value") else str(self.document_type),
            "overall_confidence": round(self.overall_confidence, 4),
            "confidence_band": self.confidence_band.value if hasattr(self.confidence_band, "value") else str(self.confidence_band),
            "classification_confidence": round(self.classification_confidence, 4) if self.classification_confidence is not None else None,
            "mean_field_confidence": round(self.mean_field_confidence, 4),
            "mean_table_confidence": round(self.mean_table_confidence, 4) if self.mean_table_confidence is not None else None,
            "mean_ocr_confidence": round(self.mean_ocr_confidence, 4) if self.mean_ocr_confidence is not None else None,
            "validation_score": round(self.validation_score, 4),
            "validation_status": self.validation_status.value if hasattr(self.validation_status, "value") else str(self.validation_status),
            "review_required": self.review_required,
            "review_reasons": self.review_reasons,
            "field_confidences": {k: v.to_dict() for k, v in self.field_confidences.items()},
            "table_confidences": [t.to_dict() for t in self.table_confidences],
            "component_breakdown": {k: round(v, 4) for k, v in self.component_breakdown.items()},
            "explanation": self.explanation,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize DocumentConfidence to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentConfidence:
        """Construct DocumentConfidence from dictionary."""
        doc_type_raw = data.get("document_type", DocumentType.UNKNOWN.value)
        try:
            doc_type = DocumentType(doc_type_raw)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

        val_status_raw = data.get("validation_status", ValidationStatus.UNVALIDATED.value)
        try:
            val_status = ValidationStatus(val_status_raw)
        except ValueError:
            val_status = ValidationStatus.UNVALIDATED

        band_raw = data.get("confidence_band", ConfidenceBand.MEDIUM.value)
        try:
            band = ConfidenceBand(band_raw)
        except ValueError:
            band = ConfidenceBand.MEDIUM

        fld_confs = {
            k: FieldConfidence.from_dict(v)
            for k, v in data.get("field_confidences", {}).items()
        }
        tbl_confs = [
            TableConfidence.from_dict(t)
            for t in data.get("table_confidences", [])
        ]

        return cls(
            document_id=data["document_id"],
            document_type=doc_type,
            overall_confidence=float(data.get("overall_confidence", 0.0)),
            confidence_band=band,
            classification_confidence=float(data["classification_confidence"]) if data.get("classification_confidence") is not None else None,
            mean_field_confidence=float(data.get("mean_field_confidence", 0.0)),
            mean_table_confidence=float(data["mean_table_confidence"]) if data.get("mean_table_confidence") is not None else None,
            mean_ocr_confidence=float(data["mean_ocr_confidence"]) if data.get("mean_ocr_confidence") is not None else None,
            validation_score=float(data.get("validation_score", 1.0)),
            validation_status=val_status,
            review_required=bool(data.get("review_required", False)),
            review_reasons=list(data.get("review_reasons", [])),
            field_confidences=fld_confs,
            table_confidences=tbl_confs,
            component_breakdown={k: float(v) for k, v in data.get("component_breakdown", {}).items()},
            explanation=data.get("explanation", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReviewTarget:
    """Exact visual, spatial, and structural location for human inspection."""
    target_type: ReviewTargetType
    page_number: int = 1
    field_name: Optional[str] = None
    table_id: Optional[str] = None
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    reason: str = ""
    severity: SeverityLevel = SeverityLevel.WARNING

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewTarget to dictionary."""
        return {
            "target_type": self.target_type.value,
            "page_number": self.page_number,
            "field_name": self.field_name,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "column_name": self.column_name,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "reason": self.reason,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewTarget:
        """Construct ReviewTarget from dictionary."""
        bbox_data = data.get("bounding_box")
        return cls(
            target_type=ReviewTargetType(data["target_type"]),
            page_number=int(data.get("page_number", 1)),
            field_name=data.get("field_name"),
            table_id=data.get("table_id"),
            row_index=int(data["row_index"]) if data.get("row_index") is not None else None,
            column_name=data.get("column_name"),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            reason=data.get("reason", ""),
            severity=SeverityLevel(data.get("severity", SeverityLevel.WARNING.value)),
        )


@dataclass
class ReviewIssue:
    """Individual granular issue identified during review routing analysis."""
    issue_id: str
    rule_name: str
    severity: SeverityLevel
    message: str
    affected_targets: List[ReviewTarget] = field(default_factory=list)
    source_signal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewIssue to dictionary."""
        return {
            "issue_id": self.issue_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "affected_targets": [t.to_dict() for t in self.affected_targets],
            "source_signal": self.source_signal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewIssue:
        """Construct ReviewIssue from dictionary."""
        targets_data = data.get("affected_targets", [])
        return cls(
            issue_id=data["issue_id"],
            rule_name=data["rule_name"],
            severity=SeverityLevel(data["severity"]),
            message=data["message"],
            affected_targets=[ReviewTarget.from_dict(t) for t in targets_data],
            source_signal=data.get("source_signal", ""),
        )


@dataclass
class ReviewReason:
    """Standardized reason code and description triggering review."""
    code: str
    message: str
    severity: SeverityLevel = SeverityLevel.WARNING
    target_type: ReviewTargetType = ReviewTargetType.DOCUMENT
    field_name: Optional[str] = None
    table_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewReason to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "target_type": self.target_type.value,
            "field_name": self.field_name,
            "table_id": self.table_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewReason:
        """Construct ReviewReason from dictionary."""
        return cls(
            code=data["code"],
            message=data["message"],
            severity=SeverityLevel(data.get("severity", SeverityLevel.WARNING.value)),
            target_type=ReviewTargetType(data.get("target_type", ReviewTargetType.DOCUMENT.value)),
            field_name=data.get("field_name"),
            table_id=data.get("table_id"),
        )


@dataclass
class ReviewQueueItem:
    """Comprehensive work item placed in the human reviewer operational queue."""
    document_id: str
    document_type: DocumentType = DocumentType.UNKNOWN
    priority: ReviewPriority = ReviewPriority.MEDIUM
    status: ReviewStatus = ReviewStatus.PENDING
    document_confidence: float = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.MEDIUM
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    review_required: bool = True
    reasons: List[str] = field(default_factory=list)
    issues: List[ReviewIssue] = field(default_factory=list)
    targets: List[ReviewTarget] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recommended_action: str = "Perform visual verification of flagged targets"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_count(self) -> int:
        """Total distinct review targets."""
        return len(self.targets)

    @property
    def critical_issue_count(self) -> int:
        """Number of CRITICAL issues."""
        return sum(1 for i in self.issues if i.severity == SeverityLevel.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewQueueItem to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value if hasattr(self.document_type, "value") else str(self.document_type),
            "priority": self.priority.value if hasattr(self.priority, "value") else str(self.priority),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "document_confidence": round(self.document_confidence, 4),
            "confidence_band": self.confidence_band.value if hasattr(self.confidence_band, "value") else str(self.confidence_band),
            "validation_status": self.validation_status.value if hasattr(self.validation_status, "value") else str(self.validation_status),
            "review_required": self.review_required,
            "reasons": self.reasons,
            "issues": [i.to_dict() for i in self.issues],
            "targets": [t.to_dict() for t in self.targets],
            "created_at": self.created_at.isoformat(),
            "recommended_action": self.recommended_action,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize ReviewQueueItem to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewQueueItem:
        """Construct ReviewQueueItem from dictionary."""
        ts_str = data.get("created_at")
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)

        doc_type_raw = data.get("document_type", DocumentType.UNKNOWN.value)
        try:
            doc_type = DocumentType(doc_type_raw)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

        prio_raw = data.get("priority", ReviewPriority.MEDIUM.value)
        try:
            priority = ReviewPriority(prio_raw)
        except ValueError:
            priority = ReviewPriority.MEDIUM

        status_raw = data.get("status", ReviewStatus.PENDING.value)
        try:
            status = ReviewStatus(status_raw)
        except ValueError:
            status = ReviewStatus.PENDING

        band_raw = data.get("confidence_band", ConfidenceBand.MEDIUM.value)
        try:
            band = ConfidenceBand(band_raw)
        except ValueError:
            band = ConfidenceBand.MEDIUM

        val_status_raw = data.get("validation_status", ValidationStatus.UNVALIDATED.value)
        try:
            val_status = ValidationStatus(val_status_raw)
        except ValueError:
            val_status = ValidationStatus.UNVALIDATED

        return cls(
            document_id=data["document_id"],
            document_type=doc_type,
            priority=priority,
            status=status,
            document_confidence=float(data.get("document_confidence", 0.0)),
            confidence_band=band,
            validation_status=val_status,
            review_required=bool(data.get("review_required", True)),
            reasons=list(data.get("reasons", [])),
            issues=[ReviewIssue.from_dict(i) for i in data.get("issues", [])],
            targets=[ReviewTarget.from_dict(t) for t in data.get("targets", [])],
            created_at=ts,
            recommended_action=data.get("recommended_action", "Perform visual verification of flagged targets"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReviewRoutingResult:
    """Unified outcome of confidence scoring and human review routing dispatch."""
    document_id: str
    is_straight_through: bool = False
    document_confidence: DocumentConfidence = field(default_factory=lambda: DocumentConfidence(document_id=""))
    queue_item: Optional[ReviewQueueItem] = None
    field_confidences: Dict[str, FieldConfidence] = field(default_factory=dict)
    table_confidences: List[TableConfidence] = field(default_factory=list)
    review_flags: List[ReviewFlag] = field(default_factory=list)

    @property
    def review_required(self) -> bool:
        """True if manual human review is required."""
        return not self.is_straight_through

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewRoutingResult to dictionary."""
        return {
            "document_id": self.document_id,
            "is_straight_through": self.is_straight_through,
            "review_required": self.review_required,
            "document_confidence": self.document_confidence.to_dict(),
            "queue_item": self.queue_item.to_dict() if self.queue_item else None,
            "field_confidences": {k: v.to_dict() for k, v in self.field_confidences.items()},
            "table_confidences": [t.to_dict() for t in self.table_confidences],
            "review_flags": [rf.to_dict() for rf in self.review_flags],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize ReviewRoutingResult to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewRoutingResult:
        """Construct ReviewRoutingResult from dictionary."""
        doc_conf_data = data.get("document_confidence", {})
        q_item_data = data.get("queue_item")
        fld_confs = {k: FieldConfidence.from_dict(v) for k, v in data.get("field_confidences", {}).items()}
        tbl_confs = [TableConfidence.from_dict(t) for t in data.get("table_confidences", [])]
        flags = [ReviewFlag.from_dict(rf) for rf in data.get("review_flags", [])]

        return cls(
            document_id=data["document_id"],
            is_straight_through=bool(data.get("is_straight_through", False)),
            document_confidence=DocumentConfidence.from_dict(doc_conf_data) if doc_conf_data else DocumentConfidence(document_id=data["document_id"]),
            queue_item=ReviewQueueItem.from_dict(q_item_data) if q_item_data else None,
            field_confidences=fld_confs,
            table_confidences=tbl_confs,
            review_flags=flags,
        )


__all__ = [
    "ConfidenceBand",
    "ReviewPriority",
    "ReviewStatus",
    "ReviewTargetType",
    "ConfidenceSignals",
    "FieldConfidence",
    "LineItemConfidence",
    "TableConfidence",
    "DocumentConfidence",
    "ReviewTarget",
    "ReviewIssue",
    "ReviewReason",
    "ReviewQueueItem",
    "ReviewRoutingResult",
]
