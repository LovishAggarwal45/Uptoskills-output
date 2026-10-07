"""Unit tests for ReceiptExtractor on point-of-sale receipt documents."""

import unittest
from pathlib import Path
from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    OCRTextRegion,
)
from src.core.types import FieldType
from src.extraction.receipt_extractor import ReceiptExtractor


def build_receipt_doc(doc_id: str, text: str) -> Document:
    meta = DocumentMetadata(
        document_id=doc_id,
        filename=f"{doc_id}.png",
        file_path=Path(f"data/{doc_id}.png"),
        file_type="image/png",
        file_size_bytes=512,
        checksum_sha256="dummy_sha",
        page_count=1,
    )
    regions = [
        OCRTextRegion(
            text=line,
            page_number=1,
            confidence=0.97,
            bounding_box=BoundingBox(xmin=20.0, ymin=float(i * 25), xmax=280.0, ymax=float(i * 25 + 18)),
        )
        for i, line in enumerate(text.split("\n"))
        if line.strip()
    ]
    page = DocumentPage(page_number=1, raw_text=text, ocr_text_regions=regions)
    return Document(metadata=meta, pages=[page])


class TestReceiptExtractor(unittest.TestCase):
    """Test point-of-sale receipt field extraction and provenance."""

    def test_extract_receipt_fields(self) -> None:
        receipt_text = (
            "STORE RECEIPT\n"
            "Central Mart #4821\n"
            "Receipt Number: RCPT-98214\n"
            "Date: 2026-10-01\n"
            "Time: 14:32:05\n"
            "Cashier: Robert M.\n"
            "Register #: 04\n"
            "Subtotal: $45.50\n"
            "Tax: $3.64\n"
            "Total: $49.14\n"
            "VISA **** 1234\n"
            "Thank you for shopping!\n"
        )
        doc = build_receipt_doc("doc_rcpt_01", receipt_text)
        extractor = ReceiptExtractor()
        fields = extractor.extract(doc)

        self.assertIn("merchant_name", fields)
        self.assertIn("receipt_number", fields)
        self.assertEqual(fields["receipt_number"].normalized_value, "RCPT-98214")
        self.assertIn("transaction_date", fields)
        self.assertEqual(fields["transaction_date"].normalized_value, "2026-10-01")
        self.assertIn("transaction_time", fields)
        self.assertEqual(fields["transaction_time"].value, "14:32:05")
        self.assertIn("cashier", fields)
        self.assertEqual(fields["cashier"].value, "Robert M.")
        self.assertIn("terminal_id", fields)
        self.assertEqual(fields["terminal_id"].value, "04")
        self.assertIn("subtotal", fields)
        self.assertEqual(fields["subtotal"].normalized_value, 45.50)
        self.assertIn("tax", fields)
        self.assertEqual(fields["tax"].normalized_value, 3.64)
        self.assertIn("total", fields)
        self.assertEqual(fields["total"].normalized_value, 49.14)
        self.assertIn("payment_method", fields)
        self.assertEqual(fields["payment_method"].normalized_value, "VISA")


if __name__ == "__main__":
    unittest.main()
