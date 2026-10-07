"""Data models for database persistence and audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentProcessingStatus(str, Enum):
    """Processing lifecycle states for stored documents."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ReviewAction(str, Enum):
    """Actions performed during human review."""
    CONFIRM = "confirm"
    CORRECT = "correct"
    APPROVE = "approve"
    OVERRIDE = "override"
    REJECT = "reject"


@dataclass
class DocumentRecord:
    """Database record for an ingested and processed document."""
    document_id: str
    filename: str
    file_type: str = "application/pdf"
    file_size_bytes: int = 0
    checksum_sha256: str = ""
    page_count: int = 1
    uploaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: DocumentProcessingStatus = DocumentProcessingStatus.UPLOADED
    document_type: str = "unknown"
    confidence_score: Optional[float] = None
    confidence_band: Optional[str] = None
    validation_status: Optional[str] = None
    review_priority: Optional[str] = None
    review_status: str = "pending"
    target_count: int = 0
    error_message: Optional[str] = None
    storage_path: Optional[str] = None
    result_json_path: Optional[str] = None
    metadata_json: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert DocumentRecord to dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "page_count": self.page_count,
            "uploaded_at": self.uploaded_at,
            "status": self.status.value if isinstance(self.status, DocumentProcessingStatus) else str(self.status),
            "document_type": self.document_type,
            "confidence_score": round(self.confidence_score, 4) if self.confidence_score is not None else None,
            "confidence_band": self.confidence_band,
            "validation_status": self.validation_status,
            "review_priority": self.review_priority,
            "review_status": self.review_status,
            "target_count": self.target_count,
            "error_message": self.error_message,
            "storage_path": self.storage_path,
            "result_json_path": self.result_json_path,
            "updated_at": self.updated_at,
        }


@dataclass
class CorrectionRecord:
    """Record of a human correction applied to an extracted field."""
    correction_id: str
    document_id: str
    field_name: str
    original_value: Optional[str]
    corrected_value: str
    reviewer_id: str = "reviewer_default"
    reason: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_applied: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert CorrectionRecord to dictionary."""
        return {
            "correction_id": self.correction_id,
            "document_id": self.document_id,
            "field_name": self.field_name,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "reviewer_id": self.reviewer_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "is_applied": self.is_applied,
        }


@dataclass
class DecisionRecord:
    """Record of a formal human review decision."""
    decision_id: str
    document_id: str
    action: ReviewAction
    reviewer_id: str = "reviewer_default"
    notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert DecisionRecord to dictionary."""
        return {
            "decision_id": self.decision_id,
            "document_id": self.document_id,
            "action": self.action.value if isinstance(self.action, ReviewAction) else str(self.action),
            "reviewer_id": self.reviewer_id,
            "notes": self.notes,
            "created_at": self.created_at,
        }


@dataclass
class AuditRecord:
    """Chronological event in document lifecycle audit trail."""
    audit_id: str
    document_id: str
    event_type: str
    description: str
    actor: str = "system"
    details_json: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert AuditRecord to dictionary."""
        return {
            "audit_id": self.audit_id,
            "document_id": self.document_id,
            "event_type": self.event_type,
            "description": self.description,
            "actor": self.actor,
            "details_json": self.details_json,
            "timestamp": self.timestamp,
        }


__all__ = [
    "DocumentProcessingStatus",
    "ReviewAction",
    "DocumentRecord",
    "CorrectionRecord",
    "DecisionRecord",
    "AuditRecord",
]
