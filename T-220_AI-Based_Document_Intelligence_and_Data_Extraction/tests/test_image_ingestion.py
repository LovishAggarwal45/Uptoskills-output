"""Unit tests for raster image ingestion and page decomposition."""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.core.exceptions import ImageDecodingError
from src.ingestion.image_ingestion import ImageIngestionHandler
from src.ingestion.validators import compute_sha256


class TestImageIngestion(unittest.TestCase):
    """Test suite covering ImageIngestionHandler and color modes."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.input_dir = self.root_path / "input"
        self.output_dir = self.root_path / "processed"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.handler = ImageIngestionHandler()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ingest_rgb_image(self) -> None:
        img_path = self.input_dir / "receipt_sample.png"
        img = Image.new("RGB", (800, 1200), color=(240, 240, 240))
        img.save(img_path, format="PNG")

        orig_hash_before = compute_sha256(img_path)

        doc = self.handler.ingest_image(
            file_path=img_path,
            document_id="doc_rgb_001",
            output_dir=self.output_dir,
        )

        # Source document must remain untouched
        orig_hash_after = compute_sha256(img_path)
        self.assertEqual(orig_hash_before, orig_hash_after)

        self.assertEqual(doc.id, "doc_rgb_001")
        self.assertEqual(doc.metadata.page_count, 1)
        self.assertEqual(len(doc.pages), 1)

        page = doc.pages[0]
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.width, 800.0)
        self.assertEqual(page.height, 1200.0)
        self.assertTrue(Path(page.image_path).exists())
        self.assertEqual(Path(page.image_path).name, "page_001_original.png")

    def test_ingest_grayscale_image(self) -> None:
        img_path = self.input_dir / "scan_gray.jpg"
        img = Image.new("L", (600, 900), color=180)
        img.save(img_path, format="JPEG")

        doc = self.handler.ingest_image(
            file_path=img_path,
            document_id="doc_gray_002",
            output_dir=self.output_dir,
        )

        self.assertEqual(doc.pages[0].width, 600.0)
        self.assertEqual(doc.pages[0].height, 900.0)
        self.assertEqual(doc.pages[0].metadata["source_color_mode"], "L")

    def test_ingest_rgba_image(self) -> None:
        img_path = self.input_dir / "transparent_form.png"
        img = Image.new("RGBA", (400, 400), color=(255, 0, 0, 128))
        img.save(img_path, format="PNG")

        doc = self.handler.ingest_image(
            file_path=img_path,
            document_id="doc_rgba_003",
            output_dir=self.output_dir,
        )

        self.assertEqual(doc.pages[0].width, 400.0)
        self.assertEqual(doc.pages[0].height, 400.0)
        self.assertEqual(doc.pages[0].metadata["source_color_mode"], "RGBA")

    def test_corrupted_image_raises_error(self) -> None:
        corrupted_path = self.input_dir / "broken.png"
        with open(corrupted_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nCorruptedImageDataThatCannotBeParsed")

        with self.assertRaises(ImageDecodingError):
            self.handler.ingest_image(
                file_path=corrupted_path,
                document_id="doc_broken",
                output_dir=self.output_dir,
            )


if __name__ == "__main__":
    unittest.main()
