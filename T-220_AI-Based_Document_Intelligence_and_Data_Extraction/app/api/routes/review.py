"""Human review queue, field corrections, review decisions, and audit trail routes."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.core.logging import get_logger
from src.persistence.models import DocumentProcessingStatus, ReviewAction
from src.persistence.repository import (
    AuditRepository,
    CorrectionRepository,
    DecisionRepository,
    DocumentRepository,
)
from app.api.schemas import (
    AuditItemResponse,
    CorrectionCreateRequest,
    CorrectionResponse,
    DocumentListItem,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
)

logger = get_logger("api.review")
router = APIRouter(tags=["Review & Audit"])


@router.get("/review-queue", response_model=Dict[str, Any])
def get_review_queue(
    priority: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None, alias="type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Retrieve documents awaiting human review, sorted by priority (urgent > high > medium > low)."""
    repo = DocumentRepository()
    records, total = repo.list_all(
        status="needs_review",
        document_type=document_type,
        review_priority=priority,
        limit=limit,
        offset=offset,
    )

    # Priority sorting weight
    prio_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3, None: 4}
    sorted_records = sorted(
        records,
        key=lambda r: (prio_order.get(r.review_priority, 4), r.uploaded_at),
    )

    items = [
        DocumentListItem(
            document_id=r.document_id,
            filename=r.filename,
            file_type=r.file_type,
            file_size_bytes=r.file_size_bytes,
            page_count=r.page_count,
            uploaded_at=r.uploaded_at,
            status=r.status.value,
            document_type=r.document_type,
            confidence_score=r.confidence_score,
            confidence_band=r.confidence_band,
            validation_status=r.validation_status,
            review_priority=r.review_priority,
            review_status=r.review_status,
            target_count=r.target_count,
            error_message=r.error_message,
            updated_at=r.updated_at,
        )
        for r in sorted_records
    ]

    return {
        "total_queued": total,
        "limit": limit,
        "offset": offset,
        "items": [item.model_dump() for item in items],
    }


@router.get("/documents/{document_id}/corrections", response_model=List[CorrectionResponse])
def get_document_corrections(document_id: str) -> List[CorrectionResponse]:
    """Fetch all human corrections applied to fields of a document."""
    repo = CorrectionRepository()
    records = repo.get_by_document(document_id)
    return [
        CorrectionResponse(
            correction_id=c.correction_id,
            document_id=c.document_id,
            field_name=c.field_name,
            original_value=c.original_value,
            corrected_value=c.corrected_value,
            reviewer_id=c.reviewer_id,
            reason=c.reason,
            created_at=c.created_at,
            is_applied=c.is_applied,
        )
        for c in records
    ]


@router.post("/documents/{document_id}/corrections", response_model=CorrectionResponse)
def add_field_correction(
    document_id: str,
    payload: CorrectionCreateRequest,
) -> CorrectionResponse:
    """Submit a verified human correction for an extracted field."""
    doc_repo = DocumentRepository()
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    corr_repo = CorrectionRepository()
    corr = corr_repo.add_correction(
        document_id=document_id,
        field_name=payload.field_name,
        original_value=payload.original_value,
        corrected_value=payload.corrected_value,
        reviewer_id=payload.reviewer_id,
        reason=payload.reason,
    )

    audit_repo = AuditRepository()
    audit_repo.log_event(
        document_id,
        "FIELD_CORRECTED",
        f"Corrected field '{payload.field_name}': '{payload.original_value}' -> '{payload.corrected_value}'",
        actor=payload.reviewer_id,
        details={
            "field_name": payload.field_name,
            "original_value": payload.original_value,
            "corrected_value": payload.corrected_value,
            "reason": payload.reason,
        },
    )

    return CorrectionResponse(
        correction_id=corr.correction_id,
        document_id=corr.document_id,
        field_name=corr.field_name,
        original_value=corr.original_value,
        corrected_value=corr.corrected_value,
        reviewer_id=corr.reviewer_id,
        reason=corr.reason,
        created_at=corr.created_at,
        is_applied=corr.is_applied,
    )


@router.post("/documents/{document_id}/review-decision", response_model=ReviewDecisionResponse)
def submit_review_decision(
    document_id: str,
    payload: ReviewDecisionRequest,
) -> ReviewDecisionResponse:
    """Record a formal reviewer decision (APPROVE, CONFIRM, OVERRIDE, REJECT)."""
    doc_repo = DocumentRepository()
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    if doc.status == DocumentProcessingStatus.UPLOADED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit review decision for unprocessed document '{document_id}'. Please run the processing pipeline first.",
        )

    try:
        action_enum = ReviewAction(payload.action.lower())
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid review action '{payload.action}'. Valid: confirm, correct, approve, override, reject.",
        )

    dec_repo = DecisionRepository()
    decision = dec_repo.record_decision(
        document_id=document_id,
        action=action_enum,
        reviewer_id=payload.reviewer_id,
        notes=payload.notes,
    )

    audit_repo = AuditRepository()
    audit_repo.log_event(
        document_id,
        "REVIEW_DECISION_RECORDED",
        f"Reviewer decision: {action_enum.value.upper()} by {payload.reviewer_id}",
        actor=payload.reviewer_id,
        details={"action": action_enum.value, "notes": payload.notes},
    )

    updated_doc = doc_repo.get_by_id(document_id)
    doc_stat = updated_doc.status.value if updated_doc else "completed"
    rev_stat = updated_doc.review_status if updated_doc else "approved"

    return ReviewDecisionResponse(
        decision_id=decision.decision_id,
        document_id=decision.document_id,
        action=decision.action.value,
        reviewer_id=decision.reviewer_id,
        notes=decision.notes,
        created_at=decision.created_at,
        updated_document_status=doc_stat,
        updated_review_status=rev_stat,
    )


@router.get("/documents/{document_id}/audit-trail", response_model=List[AuditItemResponse])
def get_document_audit_trail(document_id: str) -> List[AuditItemResponse]:
    """Retrieve full chronological audit history for a document."""
    audit_repo = AuditRepository()
    records = audit_repo.get_by_document(document_id)
    return [
        AuditItemResponse(
            audit_id=a.audit_id,
            document_id=a.document_id,
            event_type=a.event_type,
            description=a.description,
            actor=a.actor,
            details_json=a.details_json,
            timestamp=a.timestamp,
        )
        for a in records
    ]
