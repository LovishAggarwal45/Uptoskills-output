"""Pydantic schemas for DocuMind AI REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and dependency readiness status."""
    status: str = "healthy"
    version: str = "0.1.0"
    phase: str = "Phase 10: Professional Review Dashboard"
    ocr_ready: bool = False
    ocr_engine: str = "unknown"
    tesseract_version: Optional[str] = None
    poppler_detected: bool = False
    database_ready: bool = True
    storage_directories_ready: bool = True


class OverviewStatsResponse(BaseModel):
    """Aggregate statistics for the overview dashboard."""
    total_documents: int
    completed_documents: int
    awaiting_review: int
    failed_documents: int
    validation_errors_count: int
    priority_breakdown: Dict[str, int]
    document_type_distribution: Dict[str, int]
    recent_documents: List[Dict[str, Any]]


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading a document."""
    document_id: str
    id: Optional[str] = None
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    status: str
    uploaded_at: str
    message: str


class DocumentListItem(BaseModel):
    """Summary record for document listing."""
    document_id: str
    id: Optional[str] = None
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: int
    uploaded_at: str
    status: str
    document_type: str
    confidence_score: Optional[float] = None
    confidence_band: Optional[str] = None
    validation_status: Optional[str] = None
    review_priority: Optional[str] = None
    review_status: str
    target_count: int = 0
    error_message: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Paginated list of document records."""
    total: int
    limit: int
    offset: int
    items: List[DocumentListItem]


class CorrectionCreateRequest(BaseModel):
    """Request payload to submit a human field correction."""
    field_name: str
    original_value: Optional[str] = None
    corrected_value: str
    reviewer_id: str = "reviewer_default"
    reason: Optional[str] = None


class CorrectionResponse(BaseModel):
    """Response payload for a recorded human correction."""
    correction_id: str
    document_id: str
    field_name: str
    original_value: Optional[str]
    corrected_value: str
    reviewer_id: str
    reason: Optional[str]
    created_at: str
    is_applied: bool


class ReviewDecisionRequest(BaseModel):
    """Request payload to record a review decision."""
    action: str = Field(..., description="confirm, correct, approve, override, or reject")
    reviewer_id: str = "reviewer_default"
    notes: Optional[str] = None


class ReviewDecisionResponse(BaseModel):
    """Response payload for a recorded review decision."""
    decision_id: str
    document_id: str
    action: str
    reviewer_id: str
    notes: Optional[str]
    created_at: str
    updated_document_status: str
    updated_review_status: str


class AuditItemResponse(BaseModel):
    """Chronological event in document audit log."""
    audit_id: str
    document_id: str
    event_type: str
    description: str
    actor: str
    details_json: Optional[str] = None
    timestamp: str


__all__ = [
    "HealthResponse",
    "OverviewStatsResponse",
    "DocumentUploadResponse",
    "DocumentListItem",
    "DocumentListResponse",
    "CorrectionCreateRequest",
    "CorrectionResponse",
    "ReviewDecisionRequest",
    "ReviewDecisionResponse",
    "AuditItemResponse",
]
