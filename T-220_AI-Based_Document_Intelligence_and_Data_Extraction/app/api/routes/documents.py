"""Document management, ingestion, processing, results, and visualization routes."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from src.core.config import load_config
from src.core.logging import get_logger
from src.ingestion.validators import (
    MAGIC_BYTES_SIGNATURES,
    compute_sha256,
)
from src.persistence.models import DocumentProcessingStatus, DocumentRecord
from src.persistence.repository import (
    AuditRepository,
    CorrectionRepository,
    DecisionRepository,
    DocumentRepository,
)
from src.pipeline.service import DocumentPipelineService
from app.api.schemas import (
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)

logger = get_logger("api.documents")
router = APIRouter(prefix="/documents", tags=["Documents"])

INPUT_DIR = Path("data/input")
PROCESSED_DIR = Path("data/processed")
VIS_DIR = Path("outputs/visualizations")


def _sanitize_filename(filename: str) -> str:
    """Sanitize user-provided filename to prevent path traversal or special character injection."""
    clean = Path(filename).name
    clean = re.sub(r"[^\w\.\-\_]", "_", clean)
    return clean or "document.pdf"


def _verify_safe_path(base_dir: Path, target_path: Path) -> Path:
    """Ensure target path resolves strictly inside base directory to prevent path traversal."""
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    if resolved_base not in resolved_target.parents and resolved_target != resolved_base:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Path traversal attempt detected.",
        )
    return resolved_target


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    auto_process: bool = Form(default=False),
) -> DocumentUploadResponse:
    """Upload a PDF or image document, validate integrity, and optionally trigger processing."""
    config = load_config()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in upload request.")

    safe_name = _sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    allowed_exts = [e.lower() for e in config.ingestion.supported_extensions]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{safe_name}'. Supported: {', '.join(config.ingestion.supported_extensions)}",
        )

    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # Size limit validation
    if len(content) > config.ingestion.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed ({config.ingestion.max_file_size_bytes / (1024*1024):.1f} MB).",
        )

    # Magic byte signature sniffing
    header = content[:16]
    has_valid_signature = False
    if ext == ".pdf" and header.startswith(MAGIC_BYTES_SIGNATURES["application/pdf"]):
        has_valid_signature = True
    elif ext == ".png" and header.startswith(MAGIC_BYTES_SIGNATURES["image/png"]):
        has_valid_signature = True
    elif ext in (".jpg", ".jpeg") and header.startswith(MAGIC_BYTES_SIGNATURES["image/jpeg"]):
        has_valid_signature = True
    elif ext in (".tif", ".tiff") and (header.startswith(MAGIC_BYTES_SIGNATURES["image/tiff_le"]) or header.startswith(MAGIC_BYTES_SIGNATURES["image/tiff_be"])):
        has_valid_signature = True
    elif ext == ".bmp" and header.startswith(MAGIC_BYTES_SIGNATURES["image/bmp"]):
        has_valid_signature = True

    if not has_valid_signature:
        raise HTTPException(
            status_code=400,
            detail=f"File magic byte signature does not match declared extension for '{safe_name}'.",
        )


    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    dest_path = INPUT_DIR / f"{doc_id}_{safe_name}"
    with open(dest_path, "wb") as f:
        f.write(content)

    checksum = hashlib.sha256(content).hexdigest()
    now_iso = datetime.now(timezone.utc).isoformat()

    doc_record = DocumentRecord(
        document_id=doc_id,
        filename=safe_name,
        file_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(content),
        checksum_sha256=checksum,
        page_count=1,
        uploaded_at=now_iso,
        status=DocumentProcessingStatus.UPLOADED,
        storage_path=str(dest_path),
    )

    doc_repo = DocumentRepository()
    doc_repo.create(doc_record)

    audit_repo = AuditRepository()
    audit_repo.log_event(
        doc_id,
        "DOCUMENT_UPLOADED",
        f"Uploaded '{safe_name}' ({len(content)} bytes, SHA: {checksum[:8]}...)",
        actor="user",
    )

    if auto_process:
        service = DocumentPipelineService(config=config, doc_repo=doc_repo, audit_repo=audit_repo)
        try:
            service.process_document(dest_path, document_id=doc_id, actor="user")
            doc_record = doc_repo.get_by_id(doc_id) or doc_record
        except Exception as e:
            logger.error(f"Auto-processing failed for '{doc_id}': {e}")

    return DocumentUploadResponse(
        document_id=doc_id,
        id=doc_id,
        filename=safe_name,
        file_type=doc_record.file_type,
        file_size_bytes=doc_record.file_size_bytes,
        page_count=doc_record.page_count,
        status=doc_record.status.value,
        uploaded_at=doc_record.uploaded_at,
        message="Document uploaded and registered successfully.",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    status_filter: Optional[str] = Query(None, alias="status"),
    document_type: Optional[str] = Query(None, alias="type"),
    review_priority: Optional[str] = Query(None, alias="priority"),
    validation_status: Optional[str] = Query(None, alias="validation"),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DocumentListResponse:
    """Retrieve filtered, searchable, and paginated list of documents."""
    repo = DocumentRepository()
    records, total = repo.list_all(
        status=status_filter,
        document_type=document_type,
        review_priority=review_priority,
        validation_status=validation_status,
        search_query=search,
        limit=limit,
        offset=offset,
    )

    items = [
        DocumentListItem(
            document_id=r.document_id,
            id=r.document_id,
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
        for r in records
    ]

    return DocumentListResponse(total=total, limit=limit, offset=offset, items=items)


@router.get("/{document_id}", response_model=Dict[str, Any])
def get_document(document_id: str) -> Dict[str, Any]:
    """Retrieve detailed metadata, extracted results, visual overlays, and audit trail for a document."""
    if not document_id or document_id.strip().lower() in ("undefined", "null"):
        raise HTTPException(status_code=400, detail=f"Invalid document ID '{document_id}'.")

    repo = DocumentRepository()
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    doc_dict = doc.to_dict()
    doc_dict["id"] = doc.document_id

    # Load pages from processed dir if available
    processed_doc_dir = PROCESSED_DIR / document_id
    pages_list = []
    if processed_doc_dir.exists():
        page_imgs = sorted(processed_doc_dir.glob("page_*_processed.png")) or sorted(processed_doc_dir.glob("page_*_original.png"))
        for idx, pimg in enumerate(page_imgs):
            has_ov = (VIS_DIR / document_id / f"page_{idx+1:03d}_overlay.png").exists() or (VIS_DIR / document_id / f"page_{idx+1}_overlay.png").exists()
            pages_list.append({
                "page_number": idx + 1,
                "image_filename": pimg.name,
                "has_overlay": has_ov,
            })

    doc_dict["pages"] = pages_list

    # Load structured results from result_json_path or metadata_json
    result_data = None
    if doc.result_json_path and Path(doc.result_json_path).exists():
        try:
            with open(doc.result_json_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load result_json_path for '{document_id}': {e}")

    if result_data is None and doc.metadata_json:
        try:
            result_data = json.loads(doc.metadata_json)
        except Exception:
            pass

    if result_data:
        # 1. Extraction Results (fields and entities)
        fields_raw = result_data.get("fields", {})
        if isinstance(fields_raw, dict):
            fields_list = list(fields_raw.values())
        elif isinstance(fields_raw, list):
            fields_list = fields_raw
        else:
            fields_list = []

        entities_list = result_data.get("entities", [])
        doc_dict["extraction_results"] = {
            "fields": fields_list,
            "entities": entities_list,
        }
        doc_dict["fields"] = fields_list

        # 2. Table Results
        tables_list = result_data.get("tables", [])
        doc_dict["table_results"] = {
            "tables": tables_list,
        }
        doc_dict["tables"] = tables_list

        # 3. Validation Results
        val_issues = result_data.get("validation_results", [])
        doc_dict["validation_results"] = {
            "document_id": document_id,
            "document_type": doc.document_type,
            "overall_status": doc.validation_status or "unknown",
            "validation_score": doc.confidence_score or 0.0,
            "rules_evaluated": len(val_issues),
            "rules_passed": sum(1 for i in val_issues if i.get("status") in ("valid", "VALID")),
            "issues": val_issues,
        }

        # 4. Confidence Results
        doc_dict["confidence_results"] = result_data.get("confidence") or {
            "document_id": document_id,
            "composite_score": doc.confidence_score,
            "overall_confidence": doc.confidence_score,
            "confidence_band": doc.confidence_band,
        }

        # 5. Extract full OCR text if present
        doc_obj = result_data.get("document", {})
        pages_data = doc_obj.get("pages", [])
        ocr_text_parts = [p.get("raw_text", "") for p in pages_data if p.get("raw_text")]
        doc_dict["ocr_text"] = "\n\n--- Page Break ---\n\n".join(ocr_text_parts)

    else:
        doc_dict["extraction_results"] = {"fields": [], "entities": []}
        doc_dict["table_results"] = {"tables": []}
        doc_dict["validation_results"] = {"issues": [], "overall_status": doc.validation_status or "unknown"}
        doc_dict["confidence_results"] = {"composite_score": doc.confidence_score, "overall_confidence": doc.confidence_score}
        doc_dict["ocr_text"] = ""

    # Visualizations
    vis_doc_dir = VIS_DIR / document_id
    manifest_path = vis_doc_dir / "evidence_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man_data = json.load(f)
                doc_dict["visualizations"] = {
                    "overlay_annotations": man_data.get("overlay_annotations", []),
                    "evidence_regions": man_data.get("evidence_regions", []),
                }
        except Exception:
            doc_dict["visualizations"] = {"overlay_annotations": [], "evidence_regions": []}
    else:
        doc_dict["visualizations"] = {"overlay_annotations": [], "evidence_regions": []}

    # Audit, Corrections, Decisions
    audit_repo = AuditRepository()
    doc_dict["audit_trail"] = [a.to_dict() for a in audit_repo.get_by_document(document_id)]

    corr_repo = CorrectionRepository()
    doc_dict["corrections"] = [c.to_dict() for c in corr_repo.get_by_document(document_id)]

    dec_repo = DecisionRepository()
    doc_dict["decisions"] = [d.to_dict() for d in dec_repo.get_by_document(document_id)]

    return doc_dict


@router.post("/{document_id}/process", response_model=Dict[str, Any])
def process_document(document_id: str) -> Dict[str, Any]:
    """Trigger pipeline execution for an uploaded document."""
    repo = DocumentRepository()
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    if not doc.storage_path or not Path(doc.storage_path).exists():
        raise HTTPException(status_code=400, detail=f"Source file for document '{document_id}' is missing.")

    service = DocumentPipelineService()
    try:
        result = service.process_document(doc.storage_path, document_id=document_id, actor="user")
        updated_doc = repo.get_by_id(document_id)
        return {
            "status": "success",
            "message": "Document processed successfully.",
            "document": updated_doc.to_dict() if updated_doc else doc.to_dict(),
            "summary": result.to_dict().get("summary"),
        }
    except Exception as e:
        logger.error(f"Processing failed for document '{document_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("/{document_id}/results", response_model=Dict[str, Any])
def get_document_results(document_id: str) -> Dict[str, Any]:
    """Retrieve parsed extraction results, validation issues, tables, and confidence scores."""
    repo = DocumentRepository()
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    if doc.status in (DocumentProcessingStatus.UPLOADED, DocumentProcessingStatus.PROCESSING):
        return {
            "document_id": document_id,
            "status": doc.status.value,
            "message": "Document has not completed processing yet.",
            "fields": {},
            "tables": [],
            "validation_issues": [],
        }

    # Load from result JSON path if exists
    if doc.result_json_path and Path(doc.result_json_path).exists():
        with open(doc.result_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    # Fallback to metadata_json
    if doc.metadata_json:
        try:
            return json.loads(doc.metadata_json)
        except Exception:
            pass

    return {
        "document_id": document_id,
        "status": doc.status.value,
        "message": "No structured results available.",
        "fields": {},
        "tables": [],
        "validation_issues": [],
    }


@router.get("/{document_id}/visualizations", response_model=Dict[str, Any])
def get_document_visualizations(document_id: str) -> Dict[str, Any]:
    """Retrieve visual overlays, evidence manifest, and unlocated evidence for a document."""
    repo = DocumentRepository()
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    vis_doc_dir = VIS_DIR / document_id
    manifest_path = vis_doc_dir / "evidence_manifest.json"
    summary_path = vis_doc_dir / "visualization_summary.json"

    manifest_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    summary_data = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    # Collect available overlay images
    overlay_images = []
    if vis_doc_dir.exists():
        for ov_img in sorted(vis_doc_dir.glob("page_*_overlay.png")):
            p_num = int(ov_img.stem.split("_")[1])
            overlay_images.append({
                "page_number": p_num,
                "filename": ov_img.name,
                "url": f"/api/documents/{document_id}/overlay-image/{p_num}",
            })

    return {
        "document_id": document_id,
        "manifest": manifest_data,
        "summary": summary_data,
        "overlay_images": overlay_images,
        "unlocated_evidence": manifest_data.get("unlocated_evidence", []),
    }


@router.get("/{document_id}/page-image/{page_number}")
@router.get("/{document_id}/pages/{page_number}/image")
def get_page_image(document_id: str, page_number: int) -> FileResponse:
    """Serve original/processed page image for visualization canvas."""
    if not document_id or document_id.strip().lower() in ("undefined", "null"):
        raise HTTPException(status_code=400, detail=f"Invalid document ID '{document_id}'.")

    doc_processed_dir = PROCESSED_DIR / document_id
    if not doc_processed_dir.exists():
        raise HTTPException(status_code=404, detail=f"Processed pages directory for '{document_id}' not found.")

    # Try processed image first, then original
    candidates = [
        doc_processed_dir / f"page_{page_number:03d}_processed.png",
        doc_processed_dir / f"page_{page_number:03d}_original.png",
        doc_processed_dir / f"page_{page_number}_processed.png",
        doc_processed_dir / f"page_{page_number}_original.png",
    ]

    for c in candidates:
        if c.exists():
            safe_file = _verify_safe_path(PROCESSED_DIR, c)
            return FileResponse(safe_file, media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"Page image for page {page_number} of document '{document_id}' not found.",
    )


@router.get("/{document_id}/overlay-image/{page_number}")
@router.get("/{document_id}/pages/{page_number}/overlay")
def get_overlay_image(document_id: str, page_number: int) -> FileResponse:
    """Serve rendered visual overlay image for a page."""
    if not document_id or document_id.strip().lower() in ("undefined", "null"):
        raise HTTPException(status_code=400, detail=f"Invalid document ID '{document_id}'.")

    doc_vis_dir = VIS_DIR / document_id
    if not doc_vis_dir.exists():
        raise HTTPException(status_code=404, detail=f"Visualization directory for '{document_id}' not found.")

    candidates = [
        doc_vis_dir / f"page_{page_number:03d}_overlay.png",
        doc_vis_dir / f"page_{page_number}_overlay.png",
    ]

    for c in candidates:
        if c.exists():
            safe_file = _verify_safe_path(VIS_DIR, c)
            return FileResponse(safe_file, media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"Overlay image for page {page_number} of document '{document_id}' not found.",
    )


@router.delete("/{document_id}", response_model=Dict[str, Any])
def delete_document(document_id: str) -> Dict[str, Any]:
    """Delete a document record and clean up associated intermediate artifacts."""
    repo = DocumentRepository()
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    repo.delete(document_id)
    return {"status": "success", "message": f"Document '{document_id}' deleted successfully."}
