"""End-to-end universal pipeline integration tests for diverse document formats and types."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.core.config import DocuMindConfig, load_config
from src.core.types import DocumentType
from src.persistence.database import DatabaseManager
from src.persistence.models import DocumentProcessingStatus, ReviewAction
from src.persistence.repository import (
    AuditRepository,
    CorrectionRepository,
    DecisionRepository,
    DocumentRepository,
)
from src.pipeline.service import DocumentPipelineService


class TestUniversalPipelineE2E(unittest.TestCase):
    """Full end-to-end integration tests across the 10 pipeline stages for universal documents."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_documind.db"
        self.db = DatabaseManager(db_path=self.db_path)

        self.doc_repo = DocumentRepository(db_manager=self.db)
        self.audit_repo = AuditRepository(db_manager=self.db)
        self.corr_repo = CorrectionRepository(db_manager=self.db)
        self.dec_repo = DecisionRepository(db_manager=self.db)

        self.config = load_config()
        self.config.storage.json_dir = str(self.temp_dir / "outputs" / "json")
        self.config.storage.csv_dir = str(self.temp_dir / "outputs" / "csv")
        self.config.storage.reports_dir = str(self.temp_dir / "outputs" / "reports")

        self.service = DocumentPipelineService(
            config=self.config,
            doc_repo=self.doc_repo,
            audit_repo=self.audit_repo,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_process_image_resume(self) -> None:
        """Verify full end-to-end pipeline execution on an image resume input."""
        # Create test resume image
        img_path = self.temp_dir / "sample_resume.png"
        img = Image.new("RGB", (800, 1000), color="white")
        img.save(img_path, format="PNG")

        doc_id = "doc_test_resume_e2e"
        result = self.service.process_document(
            file_path=img_path,
            document_id=doc_id,
            actor="test_user",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.document.id, doc_id)

        # Check document in repository
        record = self.doc_repo.get_by_id(doc_id)
        self.assertIsNotNone(record)
        self.assertIn(record.status, (DocumentProcessingStatus.COMPLETED, DocumentProcessingStatus.NEEDS_REVIEW))

        # Check audit trail recorded events
        events = self.audit_repo.get_by_document(doc_id)
        event_types = [e.event_type for e in events]
        self.assertIn("PIPELINE_STARTED", event_types)
        self.assertIn("PREPROCESSING_COMPLETED", event_types)
        self.assertIn("CLASSIFICATION_COMPLETED", event_types)
        self.assertIn("EXTRACTION_COMPLETED", event_types)
        self.assertIn("VALIDATION_COMPLETED", event_types)
        self.assertIn("PIPELINE_COMPLETED", event_types)

    def test_review_corrections_and_decisions_workflow(self) -> None:
        """Verify adding field corrections and recording formal review decisions updates document state."""
        img_path = self.temp_dir / "sample_doc.png"
        img = Image.new("RGB", (600, 800), color="white")
        img.save(img_path, format="PNG")

        doc_id = "doc_test_review_flow"
        result = self.service.process_document(img_path, document_id=doc_id)

        # 1. Add field correction
        corr = self.corr_repo.add_correction(
            document_id=doc_id,
            field_name="candidate_name",
            original_value="John Doe",
            corrected_value="Jonathan Doe",
            reviewer_id="lead_reviewer",
            reason="Corrected full legal name",
        )
        self.assertIsNotNone(corr.correction_id)

        # 2. Record review decision
        decision = self.dec_repo.record_decision(
            document_id=doc_id,
            action=ReviewAction.APPROVE,
            reviewer_id="lead_reviewer",
            notes="Approved after legal name verification",
        )
        self.assertIsNotNone(decision.decision_id)

        # 3. Verify document record updated to approved
        updated_doc = self.doc_repo.get_by_id(doc_id)
        self.assertEqual(updated_doc.review_status, "approved")
        self.assertEqual(updated_doc.status, DocumentProcessingStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
