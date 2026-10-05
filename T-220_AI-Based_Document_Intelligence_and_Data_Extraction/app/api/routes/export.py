"""Export download endpoints for JSON, CSV, and Markdown validation reports."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from src.core.logging import get_logger
from src.export.exporter import DocumentExporter
from src.persistence.repository import (
    AuditRepository,
    CorrectionRepository,
    DocumentRepository,
)

logger = get_logger("api.export")
router = APIRouter(prefix="/documents/{document_id}/export", tags=["Export"])

JSON_DIR = Path("outputs/json")
CSV_DIR = Path("outputs/csv")
REPORTS_DIR = Path("outputs/reports")


@router.get("/json")
def export_json(document_id: str) -> Response:
    """Download original machine-generated extraction JSON file."""
    repo = DocumentRepository()
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    json_path = JSON_DIR / f"{document_id}_result.json"
    if json_path.exists():
        return FileResponse(
            json_path,
            media_type="application/json",
            filename=f"{document_id}_extracted.json",
        )
    elif doc.result_json_path and Path(doc.result_json_path).exists():
        return FileResponse(
            Path(doc.result_json_path),
            media_type="application/json",
            filename=f"{document_id}_extracted.json",
        )

    # Fallback to dynamic JSON dump from document metadata/record
    content = json.dumps(doc.to_dict(), indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{document_id}_extracted.json"'},
    )


@router.get("/reviewed")
def export_reviewed_json(document_id: str) -> Response:
    """Download reviewed JSON with human corrections applied and audit history attached."""
    doc_repo = DocumentRepository()
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    corr_repo = CorrectionRepository()
    corrections = [c.to_dict() for c in corr_repo.get_by_document(document_id)]

    audit_repo = AuditRepository()
    audit_trail = [a.to_dict() for a in audit_repo.get_by_document(document_id)]

    base_data = {}
    if doc.result_json_path and Path(doc.result_json_path).exists():
        with open(doc.result_json_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
    elif doc.metadata_json:
        try:
            base_data = json.loads(doc.metadata_json)
        except Exception:
            pass
    else:
        base_data = doc.to_dict()

    base_data["is_reviewed"] = True
    base_data["corrections_applied"] = corrections
    base_data["audit_trail"] = audit_trail

    # Apply corrections to fields
    if corrections:
        for corr in corrections:
            fname = corr.get("field_name")
            if fname and fname in base_data.get("fields", {}):
                base_data["fields"][fname]["original_extracted_value"] = base_data["fields"][fname].get("value")
                base_data["fields"][fname]["value"] = corr.get("corrected_value")
                base_data["fields"][fname]["normalized_value"] = corr.get("corrected_value")
                base_data["fields"][fname]["is_human_corrected"] = True
                base_data["fields"][fname]["correction_reason"] = corr.get("reason")
                base_data["fields"][fname]["corrected_by"] = corr.get("reviewer_id")

    payload_str = json.dumps(base_data, indent=2, default=str)
    return Response(
        content=payload_str,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{document_id}_reviewed.json"'},
    )


@router.get("/csv")
def export_csv(document_id: str) -> Response:
    """Download tabular CSV data as a single CSV or a ZIP bundle if multiple tables exist."""
    doc_repo = DocumentRepository()
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    doc_csv_dir = CSV_DIR / document_id
    if doc_csv_dir.exists():
        csv_files = list(doc_csv_dir.glob("*.csv"))
        if len(csv_files) == 1:
            return FileResponse(
                csv_files[0],
                media_type="text/csv",
                filename=csv_files[0].name,
            )
        elif len(csv_files) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for csv_f in csv_files:
                    zip_file.write(csv_f, arcname=csv_f.name)
            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{document_id}_tables_csv.zip"'},
            )

    # Fallback to generated CSV representation
    fallback_csv = f"document_id,filename,document_type,status,confidence_score\n{doc.document_id},{doc.filename},{doc.document_type},{doc.status.value},{doc.confidence_score or 0.0}\n"
    return Response(
        content=fallback_csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{document_id}_table.csv"'},
    )


@router.get("/report")
def export_validation_report(document_id: str) -> Response:
    """Download human-readable Markdown validation report."""
    doc_repo = DocumentRepository()
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    report_path = REPORTS_DIR / f"{document_id}_validation_report.md"
    if report_path.exists():
        return FileResponse(
            report_path,
            media_type="text/markdown",
            filename=f"{document_id}_validation_report.md",
        )

    # Fallback dynamically generated report
    md_content = (
        f"# DocuMind AI — Document Validation & Integrity Report\n\n"
        f"**Document ID**: `{doc.document_id}`  \n"
        f"**Filename**: `{doc.filename}`  \n"
        f"**Document Type**: `{doc.document_type}`  \n"
        f"**Processing Status**: `{doc.status.value}`  \n"
        f"**Validation Status**: `{doc.validation_status or 'UNVALIDATED'}`  \n"
        f"**Confidence Score**: `{round(doc.confidence_score, 4) if doc.confidence_score is not None else 'N/A'}`  \n"
        f"**Review Priority**: `{doc.review_priority or 'NONE'}`  \n"
    )
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{document_id}_validation_report.md"'},
    )
