"""Unit tests for FieldOverlayBuilder creating field and entity visual annotations."""

import unittest

from src.core.models import BoundingBox
from src.core.types import ValidationStatus
from src.confidence.models import ConfidenceBand
from src.visualization.field_overlay import FieldOverlayBuilder
from src.visualization.models import AnnotationType, EvidenceRegion


class TestFieldOverlay(unittest.TestCase):
    """Test suite for field overlay annotation builder."""

    def test_build_field_annotation(self) -> None:
        """Verify building field annotation with confidence and normalized value."""
        region = EvidenceRegion(
            evidence_id="ev_001",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.EXTRACTED_FIELD,
            bbox=BoundingBox(100, 200, 400, 250),
            field_name="total",
            source_text="$1,500.00",
            normalized_value=1500.00,
            extraction_confidence=0.96,
            confidence_band=ConfidenceBand.HIGH,
            validation_status=ValidationStatus.VALID,
            is_spatial=True,
        )

        ann = FieldOverlayBuilder.build_field_annotation(
            region=region,
            show_confidence=True,
            line_width=2,
        )

        self.assertIsNotNone(ann)
        self.assertEqual(ann.field_name, "total")
        self.assertIn("0.96", ann.label)
        self.assertEqual(ann.badge_text, "[FLD]")
        self.assertEqual(ann.line_width, 2)
        self.assertTrue(ann.is_visible)

    def test_build_all_filters_non_field_regions(self) -> None:
        """Verify build_all processes only field and entity regions."""
        fld_reg = EvidenceRegion(
            evidence_id="ev_fld",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.EXTRACTED_FIELD,
            bbox=BoundingBox(50, 50, 150, 100),
            field_name="invoice_date",
            is_spatial=True,
        )
        ocr_reg = EvidenceRegion(
            evidence_id="ev_ocr",
            document_id="doc_01",
            page_number=1,
            region_type=AnnotationType.OCR_WORD,
            bbox=BoundingBox(10, 10, 30, 30),
            is_spatial=True,
        )

        anns = FieldOverlayBuilder.build_all([fld_reg, ocr_reg])
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0].field_name, "invoice_date")


if __name__ == "__main__":
    unittest.main()
