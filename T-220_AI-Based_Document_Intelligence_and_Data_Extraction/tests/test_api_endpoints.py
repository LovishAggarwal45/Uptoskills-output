"""Integration tests for DocuMind AI FastAPI REST endpoints."""

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.server import create_app
from src.persistence.database import DatabaseManager
from src.persistence.models import DocumentProcessingStatus, DocumentRecord
from src.persistence.repository import DocumentRepository
from src.pipeline.service import DocumentPipelineService


class TestApiEndpoints(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_api.db"
        self.app = create_app(db_path=self.db_path)
        self.client = TestClient(self.app)

        # Pre-seed a document in database for tests
        self.db_manager = DatabaseManager(self.db_path)
        self.doc_repo = DocumentRepository(self.db_manager)

        doc = DocumentRecord(
            document_id="doc-api-001",
            filename="test_invoice.pdf",
            file_type="application/pdf",
            file_size_bytes=2048,
            page_count=1,
            storage_path=str(Path(self.temp_dir) / "test_invoice.pdf"),
            status=DocumentProcessingStatus.COMPLETED,
            document_type="invoice",
            confidence_score=0.92,
            confidence_band="high",
            validation_status="valid",
            review_priority="none",
            review_status="approved",
        )
        self.doc_repo.create(doc)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_health_endpoint(self) -> None:
        """Test GET /api/health."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("phase", data)
        self.assertTrue(data["database_ready"])

    def test_stats_overview_endpoint(self) -> None:
        """Test GET /api/stats/overview."""
        response = self.client.get("/api/stats/overview")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_documents"], 1)
        self.assertIn("priority_breakdown", data)
        self.assertIn("document_type_distribution", data)

    def test_list_documents_endpoint(self) -> None:
        """Test GET /api/documents with search and filter parameters."""
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["filename"], "test_invoice.pdf")

        # Test search query
        res_search = self.client.get("/api/documents?search=invoice")
        self.assertEqual(res_search.status_code, 200)
        self.assertEqual(len(res_search.json()["items"]), 1)

        # Test filter non-matching
        res_none = self.client.get("/api/documents?status=failed")
        self.assertEqual(res_none.status_code, 200)
        self.assertEqual(len(res_none.json()["items"]), 0)

    def test_get_single_document_endpoint(self) -> None:
        """Test GET /api/documents/{id}."""
        response = self.client.get("/api/documents/doc-api-001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_id"], "doc-api-001")
        self.assertEqual(data["filename"], "test_invoice.pdf")

        # Non-existent ID
        res_404 = self.client.get("/api/documents/doc-non-existent")
        self.assertEqual(res_404.status_code, 404)

    def test_get_document_undefined_or_null_id_returns_400(self) -> None:
        """Test GET /api/documents/undefined and /null return 400 Bad Request."""
        res_undef = self.client.get("/api/documents/undefined")
        self.assertEqual(res_undef.status_code, 400)
        self.assertIn("Invalid document ID", res_undef.json()["detail"])

        res_null = self.client.get("/api/documents/null")
        self.assertEqual(res_null.status_code, 400)

        res_overlay_undef = self.client.get("/api/documents/undefined/overlay-image/1")
        self.assertEqual(res_overlay_undef.status_code, 400)

        res_page_undef = self.client.get("/api/documents/undefined/page-image/1")
        self.assertEqual(res_page_undef.status_code, 400)

    def test_document_response_has_both_id_and_document_id(self) -> None:
        """Test document list and single document responses provide both 'id' and 'document_id'."""
        res_single = self.client.get("/api/documents/doc-api-001")
        self.assertEqual(res_single.status_code, 200)
        data = res_single.json()
        self.assertEqual(data["id"], "doc-api-001")
        self.assertEqual(data["document_id"], "doc-api-001")

        res_list = self.client.get("/api/documents")
        self.assertEqual(res_list.status_code, 200)
        items = res_list.json()["items"]
        self.assertGreaterEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "doc-api-001")
        self.assertEqual(items[0]["document_id"], "doc-api-001")

    def test_upload_document_validation(self) -> None:
        """Test POST /api/documents upload format rejection and acceptance."""
        # Reject invalid mime / extension
        bad_file = io.BytesIO(b"malicious script contents")
        response = self.client.post(
            "/api/documents",
            files={"file": ("test.exe", bad_file, "application/x-msdownload")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file", response.json()["detail"])

        # Accept valid PNG (with PNG magic bytes)
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        valid_file = io.BytesIO(png_bytes)
        res_ok = self.client.post(
            "/api/documents",
            files={"file": ("sample.png", valid_file, "image/png")},
            data={"auto_process": "false"},
        )
        self.assertEqual(res_ok.status_code, 201)
        data = res_ok.json()
        self.assertEqual(data["filename"], "sample.png")
        self.assertEqual(data["status"], "uploaded")

    def test_review_queue_and_corrections_flow(self) -> None:
        """Test review queue query, human field correction addition, and review decision."""
        # Insert a document in review state
        doc_rev = DocumentRecord(
            document_id="doc-api-review",
            filename="receipt_needs_review.png",
            file_type="image/png",
            file_size_bytes=1024,
            page_count=1,
            storage_path=str(Path(self.temp_dir) / "receipt_needs_review.png"),
            status=DocumentProcessingStatus.NEEDS_REVIEW,
            document_type="receipt",
            confidence_score=0.62,
            review_priority="high",
            review_status="pending",
        )
        self.doc_repo.create(doc_rev)

        # GET /api/review-queue
        res_queue = self.client.get("/api/review-queue")
        self.assertEqual(res_queue.status_code, 200)
        queue_data = res_queue.json()
        self.assertGreaterEqual(queue_data["total_queued"], 1)

        # POST /api/documents/{id}/corrections
        corr_payload = {
            "field_name": "total_amount",
            "original_value": "45.00",
            "corrected_value": "48.00",
            "reviewer_id": "auditor_bob",
            "reason": "Corrected smudged digit",
        }
        res_corr = self.client.post(
            "/api/documents/doc-api-review/corrections",
            json=corr_payload,
        )
        self.assertEqual(res_corr.status_code, 200)
        self.assertEqual(res_corr.json()["corrected_value"], "48.00")

        # GET /api/documents/{id}/corrections
        res_get_corr = self.client.get("/api/documents/doc-api-review/corrections")
        self.assertEqual(res_get_corr.status_code, 200)
        self.assertEqual(len(res_get_corr.json()), 1)

        # POST /api/documents/{id}/review-decision
        dec_payload = {
            "action": "override",
            "reviewer_id": "auditor_bob",
            "notes": "Verified corrected amount with receipt merchant header.",
        }
        res_dec = self.client.post(
            "/api/documents/doc-api-review/review-decision",
            json=dec_payload,
        )
        self.assertEqual(res_dec.status_code, 200)
        self.assertEqual(res_dec.json()["updated_review_status"], "overridden")

        # GET /api/documents/{id}/audit-trail
        res_audit = self.client.get("/api/documents/doc-api-review/audit-trail")
        self.assertEqual(res_audit.status_code, 200)
        audit_records = res_audit.json()
        self.assertGreaterEqual(len(audit_records), 2)

    def test_export_endpoints(self) -> None:
        """Test document export endpoints (/export/json, /export/reviewed, /export/csv, /export/report)."""
        # Test JSON export
        res_json = self.client.get("/api/documents/doc-api-001/export/json")
        self.assertEqual(res_json.status_code, 200)
        self.assertEqual(res_json.headers["content-type"], "application/json")

        # Test Reviewed export
        res_rev = self.client.get("/api/documents/doc-api-001/export/reviewed")
        self.assertEqual(res_rev.status_code, 200)

        # Test CSV export
        res_csv = self.client.get("/api/documents/doc-api-001/export/csv")
        self.assertEqual(res_csv.status_code, 200)

        # Test Validation Report export
        res_rep = self.client.get("/api/documents/doc-api-001/export/report")
        self.assertEqual(res_rep.status_code, 200)
        self.assertIn("text/markdown", res_rep.headers["content-type"])

    def test_process_document_success_and_artifacts(self) -> None:
        """Test POST /api/documents/{id}/process end-to-end with page image and overlay retrieval."""
        from PIL import Image

        # Create a valid test image in temp dir
        test_img_path = Path(self.temp_dir) / "test_doc_proc.png"
        img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        img.save(test_img_path)

        doc = DocumentRecord(
            document_id="doc-proc-test-01",
            filename="test_doc_proc.png",
            file_type="image/png",
            file_size_bytes=test_img_path.stat().st_size,
            storage_path=str(test_img_path),
            status=DocumentProcessingStatus.UPLOADED,
        )
        self.doc_repo.create(doc)

        # Process the document via API
        res_proc = self.client.post("/api/documents/doc-proc-test-01/process")
        self.assertEqual(res_proc.status_code, 200)
        proc_data = res_proc.json()
        self.assertEqual(proc_data["status"], "success")

        # Verify updated status in database
        updated = self.doc_repo.get_by_id("doc-proc-test-01")
        self.assertIsNotNone(updated)
        self.assertIn(updated.status, (DocumentProcessingStatus.COMPLETED, DocumentProcessingStatus.NEEDS_REVIEW))

        # Test GET /api/documents/{id} returns rich detail
        res_detail = self.client.get("/api/documents/doc-proc-test-01")
        self.assertEqual(res_detail.status_code, 200)
        detail_data = res_detail.json()
        self.assertEqual(detail_data["id"], "doc-proc-test-01")
        self.assertIn("extraction_results", detail_data)
        self.assertIn("table_results", detail_data)
        self.assertIn("validation_results", detail_data)
        self.assertIn("confidence_results", detail_data)
        self.assertIn("visualizations", detail_data)
        self.assertIn("pages", detail_data)
        self.assertGreaterEqual(len(detail_data["pages"]), 1)

        # Test page image endpoint
        res_page = self.client.get("/api/documents/doc-proc-test-01/page-image/1")
        self.assertEqual(res_page.status_code, 200)
        self.assertEqual(res_page.headers["content-type"], "image/png")

        # Test overlay image endpoint
        res_overlay = self.client.get("/api/documents/doc-proc-test-01/overlay-image/1")
        self.assertEqual(res_overlay.status_code, 200)
        self.assertEqual(res_overlay.headers["content-type"], "image/png")

    def test_missing_page_and_overlay_returns_404(self) -> None:
        """Test GET page-image and overlay-image return 404 for non-existent pages/overlays."""
        res_page_404 = self.client.get("/api/documents/doc-api-001/page-image/99")
        self.assertEqual(res_page_404.status_code, 404)
        self.assertIn("not found", res_page_404.json()["detail"].lower())

        res_overlay_404 = self.client.get("/api/documents/doc-api-001/overlay-image/99")
        self.assertEqual(res_overlay_404.status_code, 404)
        self.assertIn("not found", res_overlay_404.json()["detail"].lower())

    def test_review_decision_blocked_on_unprocessed_document(self) -> None:
        """Test submitting review decision on unprocessed (UPLOADED) document returns 400."""
        doc_unproc = DocumentRecord(
            document_id="doc-unproc-01",
            filename="raw_upload.pdf",
            file_type="application/pdf",
            file_size_bytes=1024,
            storage_path=str(Path(self.temp_dir) / "raw_upload.pdf"),
            status=DocumentProcessingStatus.UPLOADED,
        )
        self.doc_repo.create(doc_unproc)

        res_dec = self.client.post(
            "/api/documents/doc-unproc-01/review-decision",
            json={"action": "approve", "reviewer_id": "auditor_bob"},
        )
        self.assertEqual(res_dec.status_code, 400)
        self.assertIn("unprocessed document", res_dec.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
