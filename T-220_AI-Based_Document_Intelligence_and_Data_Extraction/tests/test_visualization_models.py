"""Unit tests for Phase 9 visualization domain models and serialization."""

import unittest
from pathlib import Path

from src.core.models import BoundingBox, Provenance
from src.core.types import ExtractionMethod, SeverityLevel, ValidationStatus
from src.confidence.models import ConfidenceBand
from src.visualization.models import (
    AnnotationType,
    EvidenceRegion,
    OverlayAnnotation,
    PageVisualization,
    VisualCoordinateSystem,
    VisualizationManifest,
    VisualizationResult,
)


class TestVisualizationModels(unittest.TestCase):
    """Test suite for visualization domain models."""

    def setUp(self) -> None:
        self.bbox = BoundingBox(xmin=100.0, ymin=150.0, xmax=300.0, ymax=200.0)
        self.prov = Provenance(
            document_id="doc_vis_001",
            page_number=1,
            raw_text="INV-2026-9901",
            bounding_box=self.bbox,
            ocr_confidence=0.98,
            extraction_method=ExtractionMethod.REGEX_PATTERN,
        )

    def test_evidence_region_creation_and_serialization(self) -> None:
        """Verify EvidenceRegion initialization and dictionary round-trip."""
        region = EvidenceRegion(
            evidence_id="ev_001",
            document_id="doc_vis_001",
            page_number=1,
            region_type=AnnotationType.EXTRACTED_FIELD,
            bbox=self.bbox,
            source_text="INV-2026-9901",
            normalized_value="INV-2026-9901",
            field_name="invoice_number",
            ocr_confidence=0.98,
            extraction_confidence=0.95,
            aggregated_confidence=0.96,
            confidence_band=ConfidenceBand.HIGH,
            validation_status=ValidationStatus.VALID,
            review_required=False,
            provenance=self.prov,
            is_spatial=True,
        )

        self.assertEqual(region.evidence_id, "ev_001")
        self.assertEqual(region.field_name, "invoice_number")
        self.assertEqual(region.confidence_band, ConfidenceBand.HIGH)
        self.assertTrue(region.is_spatial)

        # Dictionary round-trip
        data = region.to_dict()
        self.assertEqual(data["evidence_id"], "ev_001")
        self.assertEqual(data["region_type"], AnnotationType.EXTRACTED_FIELD.value)
        self.assertEqual(data["confidence_band"], ConfidenceBand.HIGH.value)

        recon = EvidenceRegion.from_dict(data)
        self.assertEqual(recon.evidence_id, region.evidence_id)
        self.assertEqual(recon.field_name, region.field_name)
        self.assertEqual(recon.confidence_band, ConfidenceBand.HIGH)
        self.assertEqual(recon.bbox, self.bbox)

    def test_evidence_region_invalid_bounds_raise(self) -> None:
        """Verify EvidenceRegion validates confidence ranges and page numbers."""
        with self.assertRaises(ValueError):
            EvidenceRegion(
                evidence_id="ev_bad",
                document_id="doc_vis_001",
                page_number=0,  # Must be >= 1
                region_type=AnnotationType.EXTRACTED_FIELD,
            )

        with self.assertRaises(ValueError):
            EvidenceRegion(
                evidence_id="ev_bad",
                document_id="doc_vis_001",
                page_number=1,
                region_type=AnnotationType.EXTRACTED_FIELD,
                ocr_confidence=1.5,  # Must be in [0.0, 1.0]
            )

    def test_overlay_annotation_creation_and_serialization(self) -> None:
        """Verify OverlayAnnotation construction and dictionary round-trip."""
        ann = OverlayAnnotation(
            annotation_id="ann_001",
            page_number=1,
            annotation_type=AnnotationType.EXTRACTED_FIELD,
            bounding_box=self.bbox,
            label="[FLD] invoice_number (0.95)",
            tooltip="invoice_number: INV-2026-9901",
            severity=SeverityLevel.INFO,
            confidence_band=ConfidenceBand.HIGH,
            confidence_score=0.95,
            source_evidence_id="ev_001",
            field_name="invoice_number",
            color_rgba=(30, 144, 255, 255),
            fill_rgba=(30, 144, 255, 45),
            line_width=2,
            badge_text="[FLD]",
            is_visible=True,
        )

        self.assertEqual(ann.annotation_id, "ann_001")
        self.assertEqual(ann.badge_text, "[FLD]")

        data = ann.to_dict()
        self.assertEqual(data["annotation_id"], "ann_001")
        self.assertEqual(data["color_rgba"], [30, 144, 255, 255])

        recon = OverlayAnnotation.from_dict(data)
        self.assertEqual(recon.annotation_id, ann.annotation_id)
        self.assertEqual(recon.color_rgba, (30, 144, 255, 255))
        self.assertEqual(recon.bounding_box, self.bbox)

    def test_page_visualization_and_manifest_roundtrip(self) -> None:
        """Verify PageVisualization and VisualizationManifest JSON round-trip."""
        region = EvidenceRegion(
            evidence_id="ev_001",
            document_id="doc_vis_001",
            page_number=1,
            region_type=AnnotationType.EXTRACTED_FIELD,
            bbox=self.bbox,
            field_name="total",
            is_spatial=True,
        )
        unloc = EvidenceRegion(
            evidence_id="ev_002",
            document_id="doc_vis_001",
            page_number=1,
            region_type=AnnotationType.UNLOCATED_ISSUE,
            bbox=None,
            field_name="tax_id",
            is_spatial=False,
            review_reasons=["Missing mandatory tax_id"],
        )

        manifest = VisualizationManifest(
            document_id="doc_vis_001",
            total_pages=1,
            evidence_regions=[region],
            unlocated_evidence=[unloc],
        )

        json_str = manifest.to_json()
        self.assertIn("ev_001", json_str)
        self.assertIn("ev_002", json_str)

        recon_manifest = VisualizationManifest.from_dict(manifest.to_dict())
        self.assertEqual(len(recon_manifest.evidence_regions), 1)
        self.assertEqual(len(recon_manifest.unlocated_evidence), 1)
        self.assertFalse(recon_manifest.unlocated_evidence[0].is_spatial)

    def test_visualization_result_properties(self) -> None:
        """Verify VisualizationResult metrics and status properties."""
        page_vis = PageVisualization(
            document_id="doc_vis_001",
            page_number=1,
            image_width=1000,
            image_height=1400,
            overlay_image_path=Path("outputs/visualizations/doc_vis_001/page_001_overlay.png"),
            rendered_layers=["ocr", "fields", "review"],
            rendering_status="rendered",
        )

        res = VisualizationResult(
            document_id="doc_vis_001",
            page_visualizations=[page_vis],
            artifact_paths={"page_1": Path("outputs/page_1.png")},
        )

        self.assertEqual(res.total_pages, 1)
        self.assertFalse(res.has_errors)


if __name__ == "__main__":
    unittest.main()
