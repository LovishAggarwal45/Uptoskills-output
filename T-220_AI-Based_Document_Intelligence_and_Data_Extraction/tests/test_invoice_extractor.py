"""Unit tests for InvoiceExtractor on single and multi-page invoices."""

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
from src.extraction.invoice_extractor import InvoiceExtractor


def build_invoice_doc(doc_id: str, pages_text: list[str]) -> Document:
    meta = DocumentMetadata(
        document_id=doc_id,
        filename=f"{doc_id}.pdf",
        file_path=Path(f"data/{doc_id}.pdf"),
        file_type="application/pdf",
        file_size_bytes=1024,
        checksum_sha256="dummy_sha",
        page_count=len(pages_text),
    )
    pages = []
    for idx, txt in enumerate(pages_text, start=1):
        regions = [
            OCRTextRegion(
                text=line,
                page_number=idx,
                confidence=0.96,
                bounding_box=BoundingBox(xmin=50.0, ymin=float(i * 30), xmax=350.0, ymax=float(i * 30 + 20)),
            )
            for i, line in enumerate(txt.split("\n"))
            if line.strip()
        ]
        pages.append(DocumentPage(page_number=idx, raw_text=txt, ocr_text_regions=regions))
    return Document(metadata=meta, pages=pages)


class TestInvoiceExtractor(unittest.TestCase):
    """Test invoice-specific field extraction, confidence separation, and provenance."""

    def test_extract_complete_invoice_fields(self) -> None:
        invoice_text = (
            "Apex Industrial Solutions LLC\n"
            "TAX INVOICE\n"
            "Invoice Number: INV-2026-9810\n"
            "Invoice Date: 2026-10-01\n"
            "Due Date: 2026-10-31\n"
            "Bill To: Global Logistics Inc\n"
            "Payment Terms: NET 30\n"
            "Purchase Order: PO-98124\n"
            "Tax ID: VAT-98765432\n"
            "Subtotal: $4,500.00\n"
            "Tax: $450.00\n"
            "Total Due: $4,950.00\n"
        )
        doc = build_invoice_doc("doc_inv_test_01", [invoice_text])
        extractor = InvoiceExtractor()
        fields = extractor.extract(doc)

        # 1. Invoice Number
        self.assertIn("invoice_number", fields)
        inv_f = fields["invoice_number"]
        self.assertEqual(inv_f.normalized_value, "INV-2026-9810")
        self.assertEqual(inv_f.field_type, FieldType.IDENTIFIER)
        self.assertGreaterEqual(inv_f.extraction_confidence, 0.70)
        self.assertIsNotNone(inv_f.provenance.ocr_confidence)

        # 2. Dates
        self.assertIn("invoice_date", fields)
        self.assertEqual(fields["invoice_date"].normalized_value, "2026-10-01")
        self.assertIn("due_date", fields)
        self.assertEqual(fields["due_date"].normalized_value, "2026-10-31")

        # 3. Vendor & Customer
        self.assertIn("vendor_name", fields)
        self.assertIn("Apex Industrial Solutions", fields["vendor_name"].value)
        self.assertIn("customer_name", fields)
        self.assertEqual(fields["customer_name"].normalized_value, "Global Logistics Inc")

        # 4. Monetary Fields
        self.assertIn("subtotal", fields)
        self.assertEqual(fields["subtotal"].normalized_value, 4500.00)
        self.assertIn("tax", fields)
        self.assertEqual(fields["tax"].normalized_value, 450.00)
        self.assertIn("total", fields)
        self.assertEqual(fields["total"].normalized_value, 4950.00)

        # 5. Currency
        self.assertIn("currency", fields)
        self.assertEqual(fields["currency"].normalized_value, "USD")

        # 6. Terms, PO, Tax ID
        self.assertIn("payment_terms", fields)
        self.assertEqual(fields["payment_terms"].value, "NET 30")
        self.assertIn("purchase_order_number", fields)
        self.assertEqual(fields["purchase_order_number"].value, "PO-98124")
        self.assertIn("tax_id", fields)

    def test_multi_page_invoice_extraction(self) -> None:
        p1 = (
            "Apex Industrial Solutions LLC\n"
            "Invoice Number: INV-2026-9810\n"
            "Invoice Date: 2026-10-01\n"
            "Bill To: Acme Corp\n"
        )
        p2 = (
            "Subtotal: $2,000.00\n"
            "Tax: $200.00\n"
            "Total: $2,200.00\n"
        )
        doc = build_invoice_doc("doc_inv_multi", [p1, p2])
        extractor = InvoiceExtractor()
        fields = extractor.extract(doc)

        self.assertEqual(fields["invoice_number"].page_number, 1)
        self.assertEqual(fields["total"].page_number, 2)
        self.assertEqual(fields["total"].normalized_value, 2200.00)


if __name__ == "__main__":
    unittest.main()
