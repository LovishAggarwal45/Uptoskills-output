"""Master FastAPI server application for DocuMind AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.core.logging import get_logger
from src.persistence.database import DatabaseManager
from app.api.routes import (
    documents_router,
    export_router,
    health_router,
    review_router,
    stats_router,
)

logger = get_logger("api.server")


def create_app(db_path: Optional[Union[str, Path]] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    # Ensure database schema is initialized on application startup
    if db_path is not None:
        DatabaseManager.reset_instance()
        DatabaseManager.get_instance(db_path=db_path)
    else:
        DatabaseManager.get_instance()

    app = FastAPI(
        title="DocuMind AI API",
        description="AI-Powered Document Intelligence & Human-in-the-Loop Review Platform API",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Configure CORS for local development (React/Vite frontends)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler for uncaught exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred during request processing.",
                "detail": str(exc),
            },
        )

    # Register API routers with /api prefix
    api_prefix = "/api"
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(stats_router, prefix=api_prefix)
    app.include_router(documents_router, prefix=api_prefix)
    app.include_router(review_router, prefix=api_prefix)
    app.include_router(export_router, prefix=api_prefix)

    # Serve compiled frontend build if present
    dist_dir = Path("frontend/dist")
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            # If path matches an existing static file, serve it
            req_file = dist_dir / full_path
            if full_path and req_file.exists() and req_file.is_file():
                return FileResponse(req_file)
            # Otherwise serve index.html for client-side routing
            return FileResponse(dist_dir / "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, reload=True)
