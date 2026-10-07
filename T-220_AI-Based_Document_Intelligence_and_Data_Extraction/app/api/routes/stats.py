"""Overview dashboard metrics and statistics route."""

from fastapi import APIRouter
from src.persistence.repository import DocumentRepository
from app.api.schemas import OverviewStatsResponse

router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get("/overview", response_model=OverviewStatsResponse)
def get_overview_stats() -> OverviewStatsResponse:
    """Return live metrics and statistics calculated from persistent database records."""
    repo = DocumentRepository()
    stats = repo.get_overview_statistics()
    return OverviewStatsResponse(**stats)
