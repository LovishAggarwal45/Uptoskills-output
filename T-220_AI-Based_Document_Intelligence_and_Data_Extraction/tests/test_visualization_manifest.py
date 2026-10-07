"""Unit tests for VisualizationSerializer and manifest persistence."""

import tempfile
import unittest
from pathlib import Path

from src.core.models import BoundingBox
from src.visualization.models import (
    AnnotationType,
    EvidenceRegion,
    PageVisualization,
    VisualizationManifest,
    VisualizationResult,
)
from src.visualization.serializer import VisualizationSerializer


class TestVisualizationManifest(unittest.TestCase):
    """Test suite for evidence manifest and summary serialization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_load_manifest(self) -> None:
        """Verify saving and reloading evidence_manifest.json."""
        manifest = VisualizationManifest(
            document_id="doc_manifest_01",
            total_pages=1,
            evidence_regions=[
                EvidenceRegion(
                    evidence_id="ev_01",
                    document_id="doc_manifest_01",
                    page_number=1,
                    region_type=AnnotationType.EXTRACTED_FIELD,
                    bbox=BoundingBox(10, 20, 100, 50),
                    field_name="total",
                    is_spatial=True,
                )
            ],
            unlocated_evidence=[
                EvidenceRegion(
                    evidence_id="ev_02",
                    document_id="doc_manifest_01",
                    page_number=1,
                    region_type=AnnotationType.UNLOCATED_ISSUE,
                    field_name="due_date",
                    is_spatial=False,
                )
            ],
        )

        saved_path = VisualizationSerializer.save_manifest(
            manifest=manifest,
            output_dir=self.base_dir,
        )

        self.assertTrue(saved_path.exists())

        loaded = VisualizationSerializer.load_manifest(saved_path)
        self.assertEqual(loaded.document_id, "doc_manifest_01")
        self.assertEqual(len(loaded.evidence_regions), 1)
        self.assertEqual(len(loaded.unlocated_evidence), 1)
        self.assertEqual(loaded.evidence_regions[0].field_name, "total")

    def test_save_and_load_summary(self) -> None:
        """Verify saving and reloading visualization_summary.json."""
        result = VisualizationResult(
            document_id="doc_manifest_01",
            page_visualizations=[
                PageVisualization(
                    document_id="doc_manifest_01",
                    page_number=1,
                    image_width=600,
                    image_height=800,
                    rendering_status="rendered",
                )
            ],
            artifact_paths={"page_1": Path("outputs/page_1.png")},
        )

        saved_path = VisualizationSerializer.save_summary(
            result=result,
            output_dir=self.base_dir,
        )

        self.assertTrue(saved_path.exists())
        loaded = VisualizationSerializer.load_result(saved_path)
        self.assertEqual(loaded.document_id, "doc_manifest_01")
        self.assertEqual(len(loaded.page_visualizations), 1)


if __name__ == "__main__":
    unittest.main()
