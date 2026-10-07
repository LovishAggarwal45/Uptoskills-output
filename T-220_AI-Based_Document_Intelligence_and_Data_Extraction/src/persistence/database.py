"""SQLite Database manager providing connection pooling, thread-safety, and schema migrations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union

from src.core.logging import get_logger

logger = get_logger("persistence.database")

DEFAULT_DB_PATH = Path("data/documind.db")


class DatabaseManager:
    """Manages SQLite connection lifecycle and schema setup for DocuMind AI."""

    _instance: Optional[DatabaseManager] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def get_instance(cls, db_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
        """Singleton accessor for DatabaseManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path=db_path)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for test isolation)."""
        with cls._lock:
            cls._instance = None

    def get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection with foreign keys enabled and row factory."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_schema(self) -> None:
        """Initialize database tables, constraints, and indices."""
        schema_sql = """
        -- 1. Documents Table
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 1,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            document_type TEXT NOT NULL DEFAULT 'unknown',
            confidence_score REAL,
            confidence_band TEXT,
            validation_status TEXT,
            review_priority TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            target_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            storage_path TEXT,
            result_json_path TEXT,
            metadata_json TEXT,
            updated_at TEXT
        );

        -- 2. Human Corrections Table
        CREATE TABLE IF NOT EXISTS human_corrections (
            correction_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            original_value TEXT,
            corrected_value TEXT NOT NULL,
            reviewer_id TEXT NOT NULL DEFAULT 'reviewer_default',
            reason TEXT,
            created_at TEXT NOT NULL,
            is_applied INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        -- 3. Review Decisions Table
        CREATE TABLE IF NOT EXISTS review_decisions (
            decision_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reviewer_id TEXT NOT NULL DEFAULT 'reviewer_default',
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        -- 4. Audit Trail Table
        CREATE TABLE IF NOT EXISTS audit_trail (
            audit_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT 'system',
            details_json TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        -- Indices for fast searching and filtering
        CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(document_type);
        CREATE INDEX IF NOT EXISTS idx_docs_priority ON documents(review_priority);
        CREATE INDEX IF NOT EXISTS idx_docs_uploaded ON documents(uploaded_at);
        CREATE INDEX IF NOT EXISTS idx_corrections_doc ON human_corrections(document_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_doc ON review_decisions(document_id);
        CREATE INDEX IF NOT EXISTS idx_audit_doc ON audit_trail(document_id);
        """

        conn = self.get_connection()
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Database schema initialized successfully at {self.db_path}")


__all__ = ["DatabaseManager", "DEFAULT_DB_PATH"]
