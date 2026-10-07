"""Repository pattern providing data access operations for documents, corrections, decisions, and audit history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.persistence.database import DatabaseManager
from src.persistence.models import (
    AuditRecord,
    CorrectionRecord,
    DecisionRecord,
    DocumentProcessingStatus,
    DocumentRecord,
    ReviewAction,
)

logger = get_logger("persistence.repository")


class DocumentRepository:
    """DAO for Document records in SQLite."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager.get_instance()

    def create(self, doc: DocumentRecord) -> DocumentRecord:
        """Insert a new document record."""
        sql = """
        INSERT INTO documents (
            document_id, filename, file_type, file_size_bytes, checksum_sha256,
            page_count, uploaded_at, status, document_type, confidence_score,
            confidence_band, validation_status, review_priority, review_status,
            target_count, error_message, storage_path, result_json_path,
            metadata_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self.db.get_connection()
        try:
            conn.execute(
                sql,
                (
                    doc.document_id,
                    doc.filename,
                    doc.file_type,
                    doc.file_size_bytes,
                    doc.checksum_sha256,
                    doc.page_count,
                    doc.uploaded_at or now,
                    doc.status.value if isinstance(doc.status, DocumentProcessingStatus) else str(doc.status),
                    doc.document_type,
                    doc.confidence_score,
                    doc.confidence_band,
                    doc.validation_status,
                    doc.review_priority,
                    doc.review_status,
                    doc.target_count,
                    doc.error_message,
                    doc.storage_path,
                    doc.result_json_path,
                    doc.metadata_json,
                    doc.updated_at or now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug(f"Created document record '{doc.document_id}' ({doc.filename})")
        return doc

    def get_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        """Fetch a document by its unique ID."""
        sql = "SELECT * FROM documents WHERE document_id = ?"
        conn = self.db.get_connection()
        try:
            cursor = conn.execute(sql, (document_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_document(row)
        finally:
            conn.close()

    def list_all(
        self,
        status: Optional[str] = None,
        document_type: Optional[str] = None,
        review_priority: Optional[str] = None,
        validation_status: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[DocumentRecord], int]:
        """List documents with optional filtering, search, and pagination.

        Returns:
            Tuple of (records_list, total_matching_count).
        """
        conditions: List[str] = []
        params: List[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status.lower())

        if document_type:
            conditions.append("document_type = ?")
            params.append(document_type.lower())

        if review_priority:
            conditions.append("review_priority = ?")
            params.append(review_priority.lower())

        if validation_status:
            conditions.append("validation_status = ?")
            params.append(validation_status.lower())

        if search_query:
            conditions.append("(filename LIKE ? OR document_id LIKE ?)")
            term = f"%{search_query}%"
            params.extend([term, term])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM documents {where_clause}"
        query_sql = f"""
        SELECT * FROM documents
        {where_clause}
        ORDER BY uploaded_at DESC
        LIMIT ? OFFSET ?
        """

        conn = self.db.get_connection()
        try:
            total = conn.execute(count_sql, params).fetchone()[0]
            cursor = conn.execute(query_sql, params + [limit, offset])
            rows = cursor.fetchall()
            records = [self._row_to_document(r) for r in rows]
            return records, total
        finally:
            conn.close()

    def update_status(
        self,
        document_id: str,
        status: DocumentProcessingStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update the processing status and optional error message of a document."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
        UPDATE documents
        SET status = ?, error_message = ?, updated_at = ?
        WHERE document_id = ?
        """
        conn = self.db.get_connection()
        try:
            cursor = conn.execute(
                sql,
                (
                    status.value if isinstance(status, DocumentProcessingStatus) else str(status),
                    error_message,
                    now,
                    document_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def update_processing_results(
        self,
        document_id: str,
        document_type: str,
        status: DocumentProcessingStatus,
        confidence_score: float,
        confidence_band: str,
        validation_status: str,
        review_priority: str,
        review_status: str,
        target_count: int,
        result_json_path: Optional[str] = None,
        metadata_json: Optional[str] = None,
    ) -> bool:
        """Update full extraction and routing results for a document."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
        UPDATE documents
        SET document_type = ?, status = ?, confidence_score = ?,
            confidence_band = ?, validation_status = ?, review_priority = ?,
            review_status = ?, target_count = ?, result_json_path = ?,
            metadata_json = ?, error_message = NULL, updated_at = ?
        WHERE document_id = ?
        """
        conn = self.db.get_connection()
        try:
            cursor = conn.execute(
                sql,
                (
                    document_type,
                    status.value if isinstance(status, DocumentProcessingStatus) else str(status),
                    confidence_score,
                    confidence_band,
                    validation_status,
                    review_priority,
                    review_status,
                    target_count,
                    result_json_path,
                    metadata_json,
                    now,
                    document_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_overview_statistics(self) -> Dict[str, Any]:
        """Compute aggregated live dashboard statistics from database."""
        conn = self.db.get_connection()
        try:
            total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            completed_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'completed'").fetchone()[0]
            needs_review_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'needs_review' OR review_status = 'needs_review'").fetchone()[0]
            failed_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'failed'").fetchone()[0]
            val_errors_docs = conn.execute("SELECT COUNT(*) FROM documents WHERE validation_status = 'invalid'").fetchone()[0]

            # Priority counts
            urgent_prio = conn.execute("SELECT COUNT(*) FROM documents WHERE review_priority = 'urgent' OR review_priority = 'critical'").fetchone()[0]
            high_prio = conn.execute("SELECT COUNT(*) FROM documents WHERE review_priority = 'high'").fetchone()[0]
            med_prio = conn.execute("SELECT COUNT(*) FROM documents WHERE review_priority = 'medium'").fetchone()[0]
            low_prio = conn.execute("SELECT COUNT(*) FROM documents WHERE review_priority = 'low'").fetchone()[0]

            # Type distribution
            type_rows = conn.execute(
                "SELECT document_type, COUNT(*) FROM documents GROUP BY document_type"
            ).fetchall()
            type_dist = {r[0]: r[1] for r in type_rows if r[0]}

            # Recent docs
            recent_cursor = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC LIMIT 5"
            )
            recent_docs = [self._row_to_document(r).to_dict() for r in recent_cursor.fetchall()]

            return {
                "total_documents": total_docs,
                "completed_documents": completed_docs,
                "awaiting_review": needs_review_docs,
                "failed_documents": failed_docs,
                "validation_errors_count": val_errors_docs,
                "priority_breakdown": {
                    "urgent": urgent_prio,
                    "critical": urgent_prio,
                    "high": high_prio,
                    "medium": med_prio,
                    "low": low_prio,
                },
                "document_type_distribution": type_dist,
                "recent_documents": recent_docs,
            }
        finally:
            conn.close()

    def delete(self, document_id: str) -> bool:
        """Delete a document record and cascaded related entries."""
        conn = self.db.get_connection()
        try:
            cursor = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        """Convert a database row to a DocumentRecord."""
        raw_status = row["status"]
        try:
            status_enum = DocumentProcessingStatus(raw_status)
        except Exception:
            status_enum = DocumentProcessingStatus.UPLOADED

        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            file_size_bytes=int(row["file_size_bytes"]),
            checksum_sha256=row["checksum_sha256"],
            page_count=int(row["page_count"]),
            uploaded_at=row["uploaded_at"],
            status=status_enum,
            document_type=row["document_type"] or "unknown",
            confidence_score=float(row["confidence_score"]) if row["confidence_score"] is not None else None,
            confidence_band=row["confidence_band"],
            validation_status=row["validation_status"],
            review_priority=row["review_priority"],
            review_status=row["review_status"] or "pending",
            target_count=int(row["target_count"] or 0),
            error_message=row["error_message"],
            storage_path=row["storage_path"],
            result_json_path=row["result_json_path"],
            metadata_json=row["metadata_json"],
            updated_at=row["updated_at"],
        )

    get_overview_stats = get_overview_statistics
    list_documents = list_all


class CorrectionRepository:
    """DAO for human field corrections."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager.get_instance()

    def add_correction(
        self,
        document_id: str,
        field_name: str,
        original_value: Optional[str],
        corrected_value: str,
        reviewer_id: str = "reviewer_default",
        reason: Optional[str] = None,
    ) -> CorrectionRecord:
        """Record a human correction for a field."""
        corr_id = f"corr_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        record = CorrectionRecord(
            correction_id=corr_id,
            document_id=document_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            reviewer_id=reviewer_id,
            reason=reason,
            created_at=now,
            is_applied=True,
        )
        sql = """
        INSERT INTO human_corrections (
            correction_id, document_id, field_name, original_value,
            corrected_value, reviewer_id, reason, created_at, is_applied
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        conn = self.db.get_connection()
        try:
            conn.execute(
                sql,
                (
                    record.correction_id,
                    record.document_id,
                    record.field_name,
                    record.original_value,
                    record.corrected_value,
                    record.reviewer_id,
                    record.reason,
                    record.created_at,
                    1 if record.is_applied else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Recorded correction '{corr_id}' for field '{field_name}' on doc '{document_id}'")
        return record

    def create(self, record: CorrectionRecord) -> CorrectionRecord:
        """Insert a correction record directly."""
        sql = """
        INSERT INTO human_corrections (
            correction_id, document_id, field_name, original_value,
            corrected_value, reviewer_id, reason, created_at, is_applied
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        conn = self.db.get_connection()
        try:
            conn.execute(
                sql,
                (
                    record.correction_id,
                    record.document_id,
                    record.field_name,
                    record.original_value,
                    record.corrected_value,
                    record.reviewer_id,
                    record.reason,
                    record.created_at or datetime.now(timezone.utc).isoformat(),
                    1 if record.is_applied else 0,
                ),
            )
            conn.commit()
            return record
        finally:
            conn.close()

    def get_by_document(self, document_id: str) -> List[CorrectionRecord]:
        """Fetch all corrections recorded for a specific document."""
        sql = "SELECT * FROM human_corrections WHERE document_id = ? ORDER BY created_at ASC"
        conn = self.db.get_connection()
        try:
            cursor = conn.execute(sql, (document_id,))
            rows = cursor.fetchall()
            return [
                CorrectionRecord(
                    correction_id=r["correction_id"],
                    document_id=r["document_id"],
                    field_name=r["field_name"],
                    original_value=r["original_value"],
                    corrected_value=r["corrected_value"],
                    reviewer_id=r["reviewer_id"],
                    reason=r["reason"],
                    created_at=r["created_at"],
                    is_applied=bool(r["is_applied"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    get_by_document_id = get_by_document


class DecisionRepository:
    """DAO for human review decisions."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager.get_instance()

    def record_decision(
        self,
        document_id: str,
        action: ReviewAction,
        reviewer_id: str = "reviewer_default",
        notes: Optional[str] = None,
    ) -> DecisionRecord:
        """Record a formal review decision."""
        dec_id = f"dec_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        record = DecisionRecord(
            decision_id=dec_id,
            document_id=document_id,
            action=action,
            reviewer_id=reviewer_id,
            notes=notes,
            created_at=now,
        )
        sql = """
        INSERT INTO review_decisions (
            decision_id, document_id, action, reviewer_id, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        conn = self.db.get_connection()
        try:
            conn.execute(
                sql,
                (
                    record.decision_id,
                    record.document_id,
                    record.action.value if isinstance(record.action, ReviewAction) else str(record.action),
                    record.reviewer_id,
                    record.notes,
                    record.created_at,
                ),
            )
            status_map = {
                ReviewAction.APPROVE: "approved",
                ReviewAction.CONFIRM: "confirmed",
                ReviewAction.OVERRIDE: "overridden",
                ReviewAction.CORRECT: "corrected",
                ReviewAction.REJECT: "rejected",
            }
            new_review_status = status_map.get(action, "approved")
            new_doc_status = (
                DocumentProcessingStatus.COMPLETED.value
                if action != ReviewAction.REJECT
                else DocumentProcessingStatus.NEEDS_REVIEW.value
            )
            conn.execute(
                "UPDATE documents SET review_status = ?, status = ?, updated_at = ? WHERE document_id = ?",
                (new_review_status, new_doc_status, now, document_id),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Recorded review decision '{dec_id}' ({action.value}) on doc '{document_id}'")
        return record

    def create(self, record: DecisionRecord) -> DecisionRecord:
        """Insert a decision record directly."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO review_decisions (
            decision_id, document_id, action, reviewer_id, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        conn = self.db.get_connection()
        try:
            conn.execute(
                sql,
                (
                    record.decision_id,
                    record.document_id,
                    record.action.value if isinstance(record.action, ReviewAction) else str(record.action),
                    record.reviewer_id,
                    record.notes,
                    record.created_at or now,
                ),
            )
            conn.commit()
            return record
        finally:
            conn.close()

    def get_by_document(self, document_id: str) -> List[DecisionRecord]:
        """Fetch all decisions for a document."""
        sql = "SELECT * FROM review_decisions WHERE document_id = ? ORDER BY created_at DESC"
        conn = self.db.get_connection()
        try:
            cursor = conn.execute(sql, (document_id,))
            rows = cursor.fetchall()
            return [
                DecisionRecord(
                    decision_id=r["decision_id"],
                    document_id=r["document_id"],
                    action=ReviewAction(r["action"]),
                    reviewer_id=r["reviewer_id"],
                    notes=r["notes"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    get_by_document_id = get_by_document


class AuditRepository:
    """DAO for chronological audit trail logs."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager.get_instance()

    def log_event(
        self,
        document_id: str,
        event_type: str,
        description: str,
        actor: str = "system",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Append an audit log event."""
        audit_id = f"aud_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        details_str = json.dumps(details, default=str) if details else None

        record = AuditRecord(
            audit_id=audit_id,
            document_id=document_id,
            event_type=event_type,
            description=description,
            actor=actor,
            details_json=details_str,
            timestamp=now,
        )
        sql = """
        INSERT INTO audit_trail (
            audit_id, document_id, event_type, description, actor, details_json, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        conn = self.db.get_connection()
        try:
            conn.execute(
                sql,
                (
                    record.audit_id,
                    record.document_id,
                    record.event_type,
                    record.description,
                    record.actor,
                    record.details_json,
                    record.timestamp,
                ),
            )
            conn.commit()
            return record
        finally:
            conn.close()

    log = log_event

    def get_by_document(self, document_id: str) -> List[AuditRecord]:
        """Fetch all audit events for a document in chronological order."""
        sql = "SELECT * FROM audit_trail WHERE document_id = ? ORDER BY timestamp ASC"
        conn = self.db.get_connection()
        try:
            cursor = conn.execute(sql, (document_id,))
            rows = cursor.fetchall()
            return [
                AuditRecord(
                    audit_id=r["audit_id"],
                    document_id=r["document_id"],
                    event_type=r["event_type"],
                    description=r["description"],
                    actor=r["actor"],
                    details_json=r["details_json"],
                    timestamp=r["timestamp"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    get_by_document_id = get_by_document


__all__ = [
    "DocumentRepository",
    "CorrectionRepository",
    "DecisionRepository",
    "AuditRepository",
]
