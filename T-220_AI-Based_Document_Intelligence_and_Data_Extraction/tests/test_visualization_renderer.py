"""Unit tests for OverlayRenderer creating image overlay artifacts."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.core.config import VisualizationConfig
from src.core.models import BoundingBox, DocumentPage
from src.core.types import SeverityLevel
from src.visualization.exceptions import SourcePageUnavailableError
from src.visualization.models import AnnotationType, OverlayAnnotation
from src.visualization.overlay_renderer import OverlayRenderer


class TestVisualizationRenderer(unittest.TestCase):
    """Test suite for OverlayRenderer operations."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        # Create a synthetic white document page image
        self.doc_id = "doc_render_test"
        self.img_dir = self.base_dir / "data" / "processed" / self.doc_id
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.img_path = self.img_dir / "page_001_processed.png"

        img = Image.new("RGB", (600, 800), color="white")
        img.save(self.img_path)

        self.page = DocumentPage(
            page_number=1,
            width=600.0,
            height=800.0,
            image_path=self.img_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_render_page_generates_overlay_artifact(self) -> None:
        """Verify rendering page produces expected PNG overlay file with legend."""
        config = VisualizationConfig()
        renderer = OverlayRenderer(config=config)

        ann = OverlayAnnotation(
            annotation_id="ann_01",
            page_number=1,
            annotation_type=AnnotationType.EXTRACTED_FIELD,
            bounding_box=BoundingBox(50, 100, 250, 140),
            label="[FLD] invoice_number (0.98)",
            color_rgba=(30, 144, 255, 255),
            fill_rgba=(30, 144, 255, 45),
        )

        out_dir = self.base_dir / "outputs" / "visualizations"
        page_vis = renderer.render_page(
            document_id=self.doc_id,
            page=self.page,
            annotations=[ann],
            output_dir=out_dir,
        )

        self.assertEqual(page_vis.rendering_status, "rendered")
        self.assertIsNotNone(page_vis.overlay_image_path)
        self.assertTrue(page_vis.overlay_image_path.exists())

        # Verify output image can be opened and has proper dimensions
        with Image.open(page_vis.overlay_image_path) as out_img:
            self.assertEqual(out_img.width, 600)
            # Height includes legend banner height (800 + 44 = 844)
            self.assertGreaterEqual(out_img.height, 800)

    def test_render_page_missing_source_raises_error(self) -> None:
        """Verify missing source image raises SourcePageUnavailableError."""
        bad_page = DocumentPage(
            page_number=2,
            image_path=self.base_dir / "nonexistent.png",
        )
        renderer = OverlayRenderer()

        with self.assertRaises(SourcePageUnavailableError):
            renderer.render_page(
                document_id=self.doc_id,
                page=bad_page,
                annotations=[],
                output_dir=self.base_dir / "outputs",
            )

    def test_layer_filtering(self) -> None:
        """Verify disabling specific layers filters out corresponding annotations."""
        config = VisualizationConfig()
        config.layers.ocr = False
        config.layers.fields = True

        renderer = OverlayRenderer(config=config)

        ocr_ann = OverlayAnnotation(
            annotation_id="ann_ocr",
            page_number=1,
            annotation_type=AnnotationType.OCR_WORD,
            bounding_box=BoundingBox(10, 10, 50, 30),
        )
        fld_ann = OverlayAnnotation(
            annotation_id="ann_fld",
            page_number=1,
            annotation_type=AnnotationType.EXTRACTED_FIELD,
            bounding_box=BoundingBox(100, 100, 300, 150),
        )

        active, layers = renderer._filter_active_annotations([ocr_ann, fld_ann])
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].annotation_type, AnnotationType.EXTRACTED_FIELD)


if __name__ == "__main__":
    unittest.main()
