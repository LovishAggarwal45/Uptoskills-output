"""Unit tests for spatial label-value extraction and bounding box proximity logic."""

import unittest
from src.core.models import BoundingBox, DocumentPage, OCRTextRegion
from src.extraction.spatial import (
    extract_inline_label_value,
    find_label_value_matches,
    locate_value_in_regions,
    merge_bounding_boxes,
)


class TestFieldExtractors(unittest.TestCase):
    """Test inline colon parsing, spatial region proximity, and bounding box merging."""

    def test_extract_inline_label_value(self) -> None:
        labels = ["invoice number", "inv #"]
        pair = extract_inline_label_value("Invoice Number: INV-2026-9810", labels)
        self.assertIsNotNone(pair)
        self.assertEqual(pair[0], "invoice number")
        self.assertEqual(pair[1], "INV-2026-9810")

        pair2 = extract_inline_label_value("INV #: 98124", labels)
        self.assertIsNotNone(pair2)
        self.assertEqual(pair2[1], "98124")

    def test_find_label_value_matches_same_line_and_colon(self) -> None:
        reg_lbl = OCRTextRegion(
            text="Invoice Number:",
            page_number=1,
            confidence=0.98,
            bounding_box=BoundingBox(xmin=50.0, ymin=100.0, xmax=180.0, ymax=125.0),
        )
        reg_val = OCRTextRegion(
            text="INV-9921",
            page_number=1,
            confidence=0.95,
            bounding_box=BoundingBox(xmin=190.0, ymin=100.0, xmax=280.0, ymax=125.0),
        )
        page = DocumentPage(
            page_number=1,
            raw_text="Invoice Number: INV-9921\nDate: 2026-10-01",
            ocr_text_regions=[reg_lbl, reg_val],
        )

        matches = find_label_value_matches(page, ["invoice number"])
        self.assertGreaterEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match.value_text, "INV-9921")
        self.assertIsNotNone(match.value_bbox)
        self.assertEqual(match.value_bbox.xmin, 190.0)

    def test_merge_bounding_boxes(self) -> None:
        b1 = BoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=40.0)
        b2 = BoundingBox(xmin=60.0, ymin=15.0, xmax=100.0, ymax=45.0)
        merged = merge_bounding_boxes([b1, b2])

        self.assertIsNotNone(merged)
        self.assertEqual(merged.xmin, 10.0)
        self.assertEqual(merged.ymin, 15.0)
        self.assertEqual(merged.xmax, 100.0)
        self.assertEqual(merged.ymax, 45.0)

    def test_locate_value_in_regions(self) -> None:
        reg = OCRTextRegion(
            text="INV-1001",
            page_number=1,
            confidence=0.97,
            bounding_box=BoundingBox(xmin=100.0, ymin=100.0, xmax=200.0, ymax=130.0),
        )
        box, conf = locate_value_in_regions("INV-1001", [reg])
        self.assertIsNotNone(box)
        self.assertEqual(box.xmin, 100.0)
        self.assertEqual(conf, 0.97)


if __name__ == "__main__":
    unittest.main()
