"""Comprehensive regression tests for invoice extraction, normalizers, table extraction, and validation improvements."""

from pathlib import Path
import unittest

from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    OCRTextRegion,
    Provenance,
)
from src.core.types import DocumentType, ExtractionMethod, FieldType, ValidationStatus
from src.classification.classifier import RuleBasedDocumentClassifier
from src.extraction.invoice_extractor import InvoiceExtractor
from src.extraction.normalizers import (
    normalize_identifier,
    normalize_money,
    normalize_text,
)
from src.extraction.spatial import extract_inline_label_value
from src.tables.extractor import TableExtractor
from src.tables.line_item_extractor import LineItemExtractor
from src.tables.models import Table, TableCell, TableColumn, TableRow, TableValueType
from src.tables.row_parser import RowParser, is_summary_text
from src.tables.validators import TableValidator
from src.validation.arithmetic_validators import SubtotalTaxTotalRule
from src.validation.validator import DocumentValidationEngine


def make_test_doc(doc_id: str, pages: list[DocumentPage]) -> Document:
    meta = DocumentMetadata(
        document_id=doc_id,
        filename=f"{doc_id}.pdf",
        file_path=Path(f"/tmp/{doc_id}.pdf"),
        file_type="pdf",
        file_size_bytes=1024,
        checksum_sha256="abc123sha",
        page_count=len(pages),
    )
    return Document(metadata=meta, pages=pages)


class TestInvoiceExtractionImprovements(unittest.TestCase):
    """Test suite verifying invoice extraction precision, normalizers, table extraction, and validation."""

    def test_normalize_money_tax_rate_handling(self) -> None:
        """Verify normalize_money strips tax rate qualifiers and rejects standalone percentages."""
        # Rate qualifiers with amounts
        amt, curr = normalize_money("Tax (10%): $450.00")
        self.assertEqual(amt, 450.0)
        self.assertEqual(curr, "USD")

        amt, curr = normalize_money("(18%) 900.00 INR")
        self.assertEqual(amt, 900.0)
        self.assertEqual(curr, "INR")

        amt, curr = normalize_money("@ 5.5% $24.75")
        self.assertEqual(amt, 24.75)
        self.assertEqual(curr, "USD")

        # Subtotal, Tax, and Total monetary parsing
        amt, curr = normalize_money("$4,500.00")
        self.assertEqual(amt, 4500.0)
        self.assertEqual(curr, "USD")

        amt, curr = normalize_money("$450.00")
        self.assertEqual(amt, 450.0)
        self.assertEqual(curr, "USD")

        amt, curr = normalize_money("$4,950.00")
        self.assertEqual(amt, 4950.0)
        self.assertEqual(curr, "USD")

        # Standalone percentages must be rejected as currency amounts
        amt, curr = normalize_money("10%")
        self.assertIsNone(amt)

        amt, curr = normalize_money("18.5 %")
        self.assertIsNone(amt)

    def test_extract_inline_label_value_with_tax_rates(self) -> None:
        """Verify spatial inline label matching handles parenthesized rates and @ rates."""
        labels = ["tax", "vat", "gst", "sales tax", "cgst", "sgst", "igst"]

        res = extract_inline_label_value("Tax (10%): $450.00", labels)
        self.assertIsNotNone(res)
        lbl, val = res
        self.assertEqual(lbl, "tax")
        self.assertEqual(val, "$450.00")

        res = extract_inline_label_value("GST @ 18%: 900.00", labels)
        self.assertIsNotNone(res)
        lbl, val = res
        self.assertEqual(lbl, "gst")
        self.assertEqual(val, "900.00")

        res = extract_inline_label_value("Sales Tax (5.5%): $24.75", labels)
        self.assertIsNotNone(res)
        lbl, val = res
        self.assertEqual(lbl, "sales tax")
        self.assertEqual(val, "$24.75")

    def test_normalize_identifier_po_and_tax_id(self) -> None:
        """Verify PO and tax ID normalization strips prefixes without stripping valid ID tokens."""
        self.assertEqual(normalize_identifier("INV-2026-9810"), "INV-2026-9810")
        self.assertEqual(normalize_identifier("PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier("PO # PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier("Purchase Order: PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier("PO No. PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier("Tax ID: 12-3456789"), "12-3456789")
        self.assertEqual(normalize_identifier("GSTIN: 29ABCDE1234F1Z5"), "29ABCDE1234F1Z5")
        self.assertEqual(normalize_identifier("VAT No. GB123456789"), "GB123456789")

    def test_invoice_extraction_4500_450_4950(self) -> None:
        """Verify full invoice extraction with subtotal $4,500, tax $450, total $4,950."""
        text = """Acme Industrial Corp
100 Corporate Parkway, New York, NY 10001
Tax ID: US-EIN-98765432

Bill To: Global Logistics Inc
456 Commerce Blvd, Chicago, IL 60601

Invoice Number: INV-2026-8800
Invoice Date: 2026-03-15
Due Date: 2026-04-15
Purchase Order: PO # PO-98124

Description          Qty    Unit Price    Amount
Server Maintenance     1      4500.00    4500.00

Subtotal: $4,500.00
Tax (10%): $450.00
Total: $4,950.00
"""
        regions = [
            OCRTextRegion(text="Acme Industrial Corp", page_number=1, bounding_box=BoundingBox(50, 50, 300, 70), confidence=0.98),
            OCRTextRegion(text="Tax ID: US-EIN-98765432", page_number=1, bounding_box=BoundingBox(50, 100, 300, 120), confidence=0.96),
            OCRTextRegion(text="Bill To: Global Logistics Inc", page_number=1, bounding_box=BoundingBox(50, 150, 350, 170), confidence=0.97),
            OCRTextRegion(text="Invoice Number: INV-2026-8800", page_number=1, bounding_box=BoundingBox(400, 50, 700, 70), confidence=0.99),
            OCRTextRegion(text="Invoice Date: 2026-03-15", page_number=1, bounding_box=BoundingBox(400, 80, 700, 100), confidence=0.98),
            OCRTextRegion(text="Due Date: 2026-04-15", page_number=1, bounding_box=BoundingBox(400, 110, 700, 130), confidence=0.98),
            OCRTextRegion(text="Purchase Order: PO # PO-98124", page_number=1, bounding_box=BoundingBox(400, 140, 700, 160), confidence=0.97),
            OCRTextRegion(text="Subtotal: $4,500.00", page_number=1, bounding_box=BoundingBox(450, 300, 700, 320), confidence=0.99),
            OCRTextRegion(text="Tax (10%): $450.00", page_number=1, bounding_box=BoundingBox(450, 330, 700, 350), confidence=0.99),
            OCRTextRegion(text="Total: $4,950.00", page_number=1, bounding_box=BoundingBox(450, 360, 700, 380), confidence=0.99),
        ]
        page = DocumentPage(page_number=1, raw_text=text, ocr_text_regions=regions, width=800, height=1000)
        doc = make_test_doc("doc_test_inv_4500", [page])

        extractor = InvoiceExtractor()
        fields = extractor.extract(doc)

        self.assertIn("invoice_number", fields)
        self.assertEqual(fields["invoice_number"].normalized_value, "INV-2026-8800")

        self.assertIn("purchase_order_number", fields)
        self.assertEqual(fields["purchase_order_number"].normalized_value, "PO-98124")

        self.assertIn("tax_id", fields)
        self.assertEqual(fields["tax_id"].normalized_value, "US-EIN-98765432")

        self.assertIn("subtotal", fields)
        self.assertEqual(fields["subtotal"].normalized_value, 4500.0)

        self.assertIn("tax", fields)
        self.assertEqual(fields["tax"].normalized_value, 450.0)

        self.assertIn("total", fields)
        self.assertEqual(fields["total"].normalized_value, 4950.0)

        # Validate arithmetic consistency
        arith_rule = SubtotalTaxTotalRule()
        issues = arith_rule.evaluate(fields, [], DocumentType.INVOICE)
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0].status, ValidationStatus.VALID)

    def test_summary_rows_excluded_from_line_items(self) -> None:
        """Verify Subtotal, Tax, Total, and Notes rows are not extracted as ordinary line items."""
        cols = [
            TableColumn(index=0, name="Description", inferred_type=TableValueType.TEXT, canonical_field="description"),
            TableColumn(index=1, name="Qty", inferred_type=TableValueType.INTEGER, canonical_field="quantity"),
            TableColumn(index=2, name="Unit Price", inferred_type=TableValueType.CURRENCY, canonical_field="unit_price"),
            TableColumn(index=3, name="Amount", inferred_type=TableValueType.CURRENCY, canonical_field="amount"),
        ]

        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row_index=0, col_index=0, text="Consulting Services", value_type=TableValueType.TEXT),
                TableCell(row_index=0, col_index=1, text="10", value_type=TableValueType.INTEGER),
                TableCell(row_index=0, col_index=2, text="450.00", value_type=TableValueType.CURRENCY),
                TableCell(row_index=0, col_index=3, text="4500.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.95,
        )
        row_subtotal = TableRow(
            row_index=1,
            cells=[
                TableCell(row_index=1, col_index=0, text="Subtotal", value_type=TableValueType.TEXT),
                TableCell(row_index=1, col_index=1, text="", value_type=TableValueType.TEXT),
                TableCell(row_index=1, col_index=2, text="", value_type=TableValueType.TEXT),
                TableCell(row_index=1, col_index=3, text="4500.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.95,
            is_summary=True,
        )
        row_tax = TableRow(
            row_index=2,
            cells=[
                TableCell(row_index=2, col_index=0, text="Tax (10%)", value_type=TableValueType.TEXT),
                TableCell(row_index=2, col_index=1, text="", value_type=TableValueType.TEXT),
                TableCell(row_index=2, col_index=2, text="", value_type=TableValueType.TEXT),
                TableCell(row_index=2, col_index=3, text="450.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.95,
        )
        row_total = TableRow(
            row_index=3,
            cells=[
                TableCell(row_index=3, col_index=0, text="Total", value_type=TableValueType.TEXT),
                TableCell(row_index=3, col_index=1, text="", value_type=TableValueType.TEXT),
                TableCell(row_index=3, col_index=2, text="", value_type=TableValueType.TEXT),
                TableCell(row_index=3, col_index=3, text="4950.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.95,
            is_summary=True,
        )

        table = Table(
            table_id="tbl_1",
            page_number=1,
            columns=cols,
            rows=[row1, row_subtotal, row_tax, row_total],
            confidence=0.95,
        )

        extractor = LineItemExtractor()
        line_items = extractor.extract_line_items(table)

        # Only row1 should be extracted as a line item; subtotal, tax, total must be filtered out
        self.assertEqual(len(line_items), 1)
        self.assertEqual(line_items[0].description, "Consulting Services")
        self.assertEqual(line_items[0].quantity, 10.0)
        self.assertEqual(line_items[0].unit_price, 450.0)
        self.assertEqual(line_items[0].amount, 4500.0)
        self.assertTrue(line_items[0].is_valid_arithmetic)

    def test_table_validator_with_document_totals(self) -> None:
        """Verify TableValidator passes grand total validation when document_totals includes tax."""
        cols = [
            TableColumn(index=0, name="Description", inferred_type=TableValueType.TEXT, canonical_field="description"),
            TableColumn(index=1, name="Amount", inferred_type=TableValueType.CURRENCY, canonical_field="amount"),
        ]
        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row_index=0, col_index=0, text="Widget A", value_type=TableValueType.TEXT),
                TableCell(row_index=0, col_index=1, text="4500.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.95,
        )
        table = Table(
            table_id="tbl_1",
            page_number=1,
            columns=cols,
            rows=[row1],
            confidence=0.95,
        )
        extractor = LineItemExtractor()
        table.line_items = extractor.extract_line_items(table)

        validator = TableValidator()
        doc_totals = {"subtotal": 4500.0, "tax": 450.0, "total": 4950.0}
        res = validator.validate_table(table, document_totals=doc_totals)

        self.assertTrue(res.is_valid)
        self.assertTrue(res.grand_total_valid)
        self.assertEqual(len(res.discrepancies), 0)

    def test_isolate_tax_identifier_and_po_normalization(self) -> None:
        """Verify isolate_tax_identifier extracts GSTIN/EIN/VAT from noisy strings and PO normalization."""
        from src.extraction.normalizers import isolate_tax_identifier

        # Mixed tax ID and currency
        raw_tax = "27AAACA1234F1Z5 | VAT: US123456789 Currency: USD"
        self.assertEqual(isolate_tax_identifier(raw_tax), "27AAACA1234F1Z5")

        # US EIN
        self.assertEqual(isolate_tax_identifier("Tax ID: 12-3456789 (State Dept)"), "12-3456789")
        self.assertEqual(isolate_tax_identifier("EIN: US-EIN-98765432"), "US-EIN-98765432")

        # EU VAT
        self.assertEqual(isolate_tax_identifier("VAT Registration: GB987654321"), "GB987654321")

        # PO normalization with varying prefixes and punctuation
        self.assertEqual(normalize_identifier("PO: PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier("Purchase Order:   PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier("PO # PO-98124"), "PO-98124")
        self.assertEqual(normalize_identifier(":::PO-98124:::"), "PO-98124")

    def test_vendor_and_customer_name_isolation_with_tricky_headers(self) -> None:
        """Verify vendor_name and customer_name are accurately extracted without picking labels like 'Ship To:' or 'invoice date.'."""
        text = """Apex Global Logistics Inc.
123 Logistics Way, Suite 400
Chicago, IL 60607
Tax ID: 27AAACA1234F1Z5 | VAT: US123456789 Currency: USD

Invoice Date: 2026-10-01
Due Date: 2026-10-31
Invoice Number: INV-2026-9810
Purchase Order: PO: PO-98124

Bill To:                           Ship To:
Acme Technologies Corp.            Acme Technologies Corp.
100 Innovation Way                 200 Distribution Center
Silicon Valley, CA 94025           Reno, NV 89501

Description               Qty    Unit Price    Amount
1. Cloud Infrastructure     1       2500.00   2500.00
2. Network Security         1        900.00    900.00
3. Storage Allocation       1        500.00    500.00

Subtotal: $4,500.00
Tax (10%): $450.00
Total: $4,950.00
"""
        regions = [
            OCRTextRegion(text="Apex Global Logistics Inc.", page_number=1, bounding_box=BoundingBox(50, 40, 350, 60), confidence=0.99),
            OCRTextRegion(text="123 Logistics Way, Suite 400", page_number=1, bounding_box=BoundingBox(50, 65, 300, 80), confidence=0.95),
            OCRTextRegion(text="Tax ID: 27AAACA1234F1Z5 | VAT: US123456789 Currency: USD", page_number=1, bounding_box=BoundingBox(50, 90, 400, 110), confidence=0.97),
            OCRTextRegion(text="Invoice Date: 2026-10-01", page_number=1, bounding_box=BoundingBox(500, 40, 750, 60), confidence=0.99),
            OCRTextRegion(text="Due Date: 2026-10-31", page_number=1, bounding_box=BoundingBox(500, 65, 750, 80), confidence=0.99),
            OCRTextRegion(text="Invoice Number: INV-2026-9810", page_number=1, bounding_box=BoundingBox(500, 90, 750, 110), confidence=0.99),
            OCRTextRegion(text="Purchase Order: PO: PO-98124", page_number=1, bounding_box=BoundingBox(500, 115, 750, 135), confidence=0.98),
            OCRTextRegion(text="Bill To:", page_number=1, bounding_box=BoundingBox(50, 160, 120, 180), confidence=0.98),
            OCRTextRegion(text="Ship To:", page_number=1, bounding_box=BoundingBox(400, 160, 470, 180), confidence=0.98),
            OCRTextRegion(text="Acme Technologies Corp.", page_number=1, bounding_box=BoundingBox(50, 185, 300, 205), confidence=0.99),
            OCRTextRegion(text="Subtotal: $4,500.00", page_number=1, bounding_box=BoundingBox(500, 350, 750, 370), confidence=0.99),
            OCRTextRegion(text="Tax (10%): $450.00", page_number=1, bounding_box=BoundingBox(500, 375, 750, 395), confidence=0.99),
            OCRTextRegion(text="Total: $4,950.00", page_number=1, bounding_box=BoundingBox(500, 400, 750, 420), confidence=0.99),
        ]
        page = DocumentPage(page_number=1, raw_text=text, ocr_text_regions=regions, width=800, height=1000)
        doc = make_test_doc("doc_tricky_headers", [page])

        extractor = InvoiceExtractor()
        fields = extractor.extract(doc)

        self.assertIn("vendor_name", fields)
        self.assertEqual(fields["vendor_name"].normalized_value, "Apex Global Logistics Inc.")
        self.assertNotEqual(fields["vendor_name"].normalized_value, "invoice date.")

        self.assertIn("customer_name", fields)
        self.assertEqual(fields["customer_name"].normalized_value, "Acme Technologies Corp.")
        self.assertNotEqual(fields["customer_name"].normalized_value, "Ship To:")

        self.assertIn("purchase_order_number", fields)
        self.assertEqual(fields["purchase_order_number"].normalized_value, "PO-98124")

        self.assertIn("tax_id", fields)
        self.assertEqual(fields["tax_id"].normalized_value, "27AAACA1234F1Z5")

        self.assertEqual(fields["invoice_number"].normalized_value, "INV-2026-9810")
        self.assertEqual(fields["invoice_date"].normalized_value, "2026-10-01")
        self.assertEqual(fields["due_date"].normalized_value, "2026-10-31")
        self.assertEqual(fields["subtotal"].normalized_value, 4500.0)
        self.assertEqual(fields["tax"].normalized_value, 450.0)
        self.assertEqual(fields["total"].normalized_value, 4950.0)

    def test_table_subtotal_mismatch_validation(self) -> None:
        """Verify that when extracted product items sum to $3,900 but invoice subtotal is $4,500, TableValidator flags a subtotal mismatch without crashing."""
        cols = [
            TableColumn(index=0, name="Description", inferred_type=TableValueType.TEXT, canonical_field="description"),
            TableColumn(index=1, name="Qty", inferred_type=TableValueType.INTEGER, canonical_field="quantity"),
            TableColumn(index=2, name="Unit Price", inferred_type=TableValueType.CURRENCY, canonical_field="unit_price"),
            TableColumn(index=3, name="Amount", inferred_type=TableValueType.CURRENCY, canonical_field="amount"),
        ]

        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row_index=0, col_index=0, text="1. Cloud Infrastructure", value_type=TableValueType.TEXT),
                TableCell(row_index=0, col_index=1, text="1", value_type=TableValueType.INTEGER),
                TableCell(row_index=0, col_index=2, text="2500.00", value_type=TableValueType.CURRENCY),
                TableCell(row_index=0, col_index=3, text="2500.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.98,
        )
        row2 = TableRow(
            row_index=1,
            cells=[
                TableCell(row_index=1, col_index=0, text="2. Network Security", value_type=TableValueType.TEXT),
                TableCell(row_index=1, col_index=1, text="1", value_type=TableValueType.INTEGER),
                TableCell(row_index=1, col_index=2, text="900.00", value_type=TableValueType.CURRENCY),
                TableCell(row_index=1, col_index=3, text="900.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.98,
        )
        row3 = TableRow(
            row_index=2,
            cells=[
                TableCell(row_index=2, col_index=0, text="3. Storage Allocation", value_type=TableValueType.TEXT),
                TableCell(row_index=2, col_index=1, text="1", value_type=TableValueType.INTEGER),
                TableCell(row_index=2, col_index=2, text="500.00", value_type=TableValueType.CURRENCY),
                TableCell(row_index=2, col_index=3, text="500.00", value_type=TableValueType.CURRENCY),
            ],
            confidence=0.98,
        )

        table = Table(
            table_id="tbl_mismatch_test",
            page_number=1,
            columns=cols,
            rows=[row1, row2, row3],
            confidence=0.98,
        )

        extractor = LineItemExtractor()
        line_items = extractor.extract_line_items(table)
        self.assertEqual(len(line_items), 3)

        # Verify serial numbers were separated into item_code
        self.assertEqual(line_items[0].item_code, "1")
        self.assertEqual(line_items[0].description, "Cloud Infrastructure")
        self.assertEqual(line_items[0].amount, 2500.0)

        self.assertEqual(line_items[1].item_code, "2")
        self.assertEqual(line_items[1].description, "Network Security")
        self.assertEqual(line_items[1].amount, 900.0)

        self.assertEqual(line_items[2].item_code, "3")
        self.assertEqual(line_items[2].description, "Storage Allocation")
        self.assertEqual(line_items[2].amount, 500.0)

        table.line_items = line_items
        validator = TableValidator()
        doc_totals = {"subtotal": 4500.0, "tax": 450.0, "total": 4950.0}
        res = validator.validate_table(table, document_totals=doc_totals)

        # The lines sum to 3900, which does not match the invoice subtotal of 4500
        # Validator should flag grand_total_valid as False or note discrepancy
        self.assertFalse(res.grand_total_valid)
        self.assertTrue(len(res.discrepancies) > 0)
        self.assertEqual(res.discrepancies[0]["calculated"], 3900.0)
        self.assertEqual(res.discrepancies[0]["reported"], 4500.0)
        self.assertTrue(any("3900" in m or "4500" in m for m in res.messages))

    def test_railway_ticket_negative_classification(self) -> None:
        """Verify IRCTC electronic reservation slips are not classified as invoices."""
        text = """IRCTC e-Ticketing Service Electronic Reservation Slip (Personal User)
Booked From: NDLS (New Delhi)  To: BSB (Varanasi Jn)
PNR: 245-8912345   Train No. & Name: 12560 / SHIV GANGA EXP
Passenger Details:
1. John Doe, Age 32, Male, Confirmed B1/23/LB
Total Fare: Rs. 1,450.00
Payment Details: IRCTC iMudra Transaction ID: 100002345678
"""
        page = DocumentPage(page_number=1, raw_text=text, ocr_text_regions=[], width=800, height=1000)
        doc = make_test_doc("doc_railway_1", [page])

        classifier = RuleBasedDocumentClassifier()
        clf_res = classifier.classify(doc)

        self.assertNotEqual(clf_res.document_type, DocumentType.INVOICE)
        self.assertEqual(clf_res.document_type, DocumentType.GENERAL_DOCUMENT)


if __name__ == "__main__":
    unittest.main()

