"""Health check and subsystem readiness route."""

from fastapi import APIRouter
from src.core.config import load_config
from src.ingestion.validators import detect_poppler_path
from src.ocr.availability import check_ocr_availability
from app.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return platform readiness status, OCR availability, and environment diagnostics."""
    config = load_config()
    ocr_avail = check_ocr_availability(config.ocr)
    poppler_dir = detect_poppler_path(config.ingestion.poppler_path)

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        phase="Phase 10: Professional Review Dashboard",
        ocr_ready=ocr_avail.is_ready,
        ocr_engine=config.ocr.engine_type,
        tesseract_version=ocr_avail.engine_version,
        poppler_detected=poppler_dir is not None,
        database_ready=True,
        storage_directories_ready=True,
    )

