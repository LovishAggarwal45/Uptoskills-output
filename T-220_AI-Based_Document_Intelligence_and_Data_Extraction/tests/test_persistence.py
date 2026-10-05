"""Tests for Phase 10 SQLite persistence layer."""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.persistence.database import DatabaseManager
from src.persistence.models import (
    AuditRecord,
    CorrectionRecord,
    DecisionRecord,
    DocumentProcessingStatus,
    DocumentRecord,
    ReviewAction,
)
from src.persistence.repository import (
    AuditRepository,
    CorrectionRepository,
    DecisionRepository,
    DocumentRepository,
)


class TestPersistenceLayer(unittest.TestCase):
    """Test suite for SQLite persistence layer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_documind.db"
        self.db_manager = DatabaseManager(self.db_path)
        self.doc_repo = DocumentRepository(self.db_manager)
        self.corr_repo = CorrectionRepository(self.db_manager)
        self.dec_repo = DecisionRepository(self.db_manager)
        self.audit_repo = AuditRepository(self.db_manager)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_database_initialization(self) -> None:
        """Test tables, indices and PRAGMAs are created."""
        conn = self.db_manager.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            self.assertIn("documents", tables)
            self.assertIn("human_corrections", tables)
            self.assertIn("review_decisions", tables)
            self.assertIn("audit_trail", tables)

            # Check journal mode
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]
            self.assertEqual(journal_mode.upper(), "WAL")

            # Check foreign keys
            cursor.execute("PRAGMA foreign_keys;")
            fk = cursor.fetchone()[0]
            self.assertEqual(fk, 1)
        finally:
            conn.close()

    def test_document_lifecycle_crud(self) -> None:
        """Test document record creation, retrieval, status updates, and deletion."""
        doc = DocumentRecord(
            document_id="doc-test-101",
            filename="sample_invoice.pdf",
            file_type="application/pdf",
            file_size_bytes=10240,
            page_count=2,
            storage_path="/tmp/sample_invoice.pdf",
            status=DocumentProcessingStatus.UPLOADED,
        )
        self.doc_repo.create(doc)

        # Retrieve
        retrieved = self.doc_repo.get_by_id("doc-test-101")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.filename, "sample_invoice.pdf")
        self.assertEqual(retrieved.page_count, 2)
        self.assertEqual(retrieved.status, DocumentProcessingStatus.UPLOADED)

        # Update processing outcome
        self.doc_repo.update_processing_results(
            document_id="doc-test-101",
            document_type="invoice",
            status=DocumentProcessingStatus.NEEDS_REVIEW,
            confidence_score=0.74,
            confidence_band="medium",
            validation_status="warning",
            review_priority="high",
            review_status="pending",
            target_count=3,
            metadata_json='{"total_amount": 450.00}',
        )

        updated = self.doc_repo.get_by_id("doc-test-101")
        self.assertEqual(updated.status, DocumentProcessingStatus.NEEDS_REVIEW)
        self.assertEqual(updated.document_type, "invoice")
        self.assertEqual(updated.confidence_score, 0.74)
        self.assertEqual(updated.review_priority, "high")
        self.assertEqual(updated.target_count, 3)

        # List with filter
        docs, total = self.doc_repo.list_documents(status="needs_review")
        self.assertEqual(total, 1)
        self.assertEqual(docs[0].document_id, "doc-test-101")

        # Delete
        self.assertTrue(self.doc_repo.delete("doc-test-101"))
        self.assertIsNone(self.doc_repo.get_by_id("doc-test-101"))

    def test_corrections_and_audit_trail(self) -> None:
        """Test recording human field corrections and immutable audit entries."""
        doc = DocumentRecord(
            document_id="doc-test-102",
            filename="receipt.png",
            file_type="image/png",
            file_size_bytes=4096,
            page_count=1,
            storage_path="/tmp/receipt.png",
        )
        self.doc_repo.create(doc)

        # Record human correction
        corr = CorrectionRecord(
            correction_id="corr-1",
            document_id="doc-test-102",
            field_name="total_amount",
            original_value="140.00",
            corrected_value="149.00",
            reviewer_id="reviewer_alice",
            reason="OCR digit misread 9 as 0",
        )
        self.corr_repo.create(corr)

        corrections = self.corr_repo.get_by_document_id("doc-test-102")
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0].corrected_value, "149.00")
        self.assertEqual(corrections[0].reviewer_id, "reviewer_alice")

        # Record review decision
        dec = DecisionRecord(
            decision_id="dec-1",
            document_id="doc-test-102",
            action=ReviewAction.OVERRIDE,
            reviewer_id="reviewer_alice",
            notes="Overridden after correcting OCR total.",
        )
        self.dec_repo.create(dec)

        decisions = self.dec_repo.get_by_document_id("doc-test-102")
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, ReviewAction.OVERRIDE)

        # Record audit log entries
        self.audit_repo.log(
            document_id="doc-test-102",
            event_type="FIELD_CORRECTED",
            description="Field total_amount corrected to 149.00",
            actor="reviewer_alice",
            details={"field": "total_amount", "from": "140.00", "to": "149.00"},
        )
        self.audit_repo.log(
            document_id="doc-test-102",
            event_type="REVIEW_DECISION_RECORDED",
            description="Reviewer Alice overrode document validation warnings.",
            actor="reviewer_alice",
            details={"action": "override"},
        )

        audit_trail = self.audit_repo.get_by_document_id("doc-test-102")
        self.assertEqual(len(audit_trail), 2)
        self.assertEqual(audit_trail[0].event_type, "FIELD_CORRECTED")
        self.assertEqual(audit_trail[1].event_type, "REVIEW_DECISION_RECORDED")

    def test_overview_statistics_aggregation(self) -> None:
        """Test overview statistics calculation."""
        # Insert 3 documents
        doc1 = DocumentRecord(
            document_id="doc-1",
            filename="inv1.pdf",
            file_type="application/pdf",
            file_size_bytes=1000,
            page_count=1,
            storage_path="/tmp/1",
            status=DocumentProcessingStatus.COMPLETED,
            document_type="invoice",
            review_priority="low",
            review_status="approved",
        )
        doc2 = DocumentRecord(
            document_id="doc-2",
            filename="inv2.pdf",
            file_type="application/pdf",
            file_size_bytes=1000,
            page_count=1,
            storage_path="/tmp/2",
            status=DocumentProcessingStatus.NEEDS_REVIEW,
            document_type="invoice",
            review_priority="critical",
            review_status="pending",
        )
        doc3 = DocumentRecord(
            document_id="doc-3",
            filename="rcpt1.png",
            file_type="image/png",
            file_size_bytes=1000,
            page_count=1,
            storage_path="/tmp/3",
            status=DocumentProcessingStatus.FAILED,
            document_type="receipt",
            review_priority="high",
            review_status="rejected",
        )
        self.doc_repo.create(doc1)
        self.doc_repo.create(doc2)
        self.doc_repo.create(doc3)

        stats = self.doc_repo.get_overview_stats()
        self.assertEqual(stats["total_documents"], 3)
        self.assertEqual(stats["completed_documents"], 1)
        self.assertEqual(stats["awaiting_review"], 1)
        self.assertEqual(stats["failed_documents"], 1)
        self.assertEqual(stats["document_type_distribution"].get("invoice"), 2)
        self.assertEqual(stats["document_type_distribution"].get("receipt"), 1)
        self.assertEqual(stats["priority_breakdown"].get("critical"), 1)


if __name__ == "__main__":
    unittest.main()
