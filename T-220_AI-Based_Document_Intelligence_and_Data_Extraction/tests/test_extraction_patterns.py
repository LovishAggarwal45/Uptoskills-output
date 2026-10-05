"""Unit tests for extraction regular expression patterns and label definitions."""

import unittest
from src.extraction.patterns import (
    DMY_DATE_PATTERN,
    EMAIL_PATTERN,
    GENERIC_ID_PATTERN,
    INVOICE_FIELD_LABELS,
    INVOICE_ID_PATTERN,
    ISO_DATE_PATTERN,
    MONEY_PATTERN,
    PHONE_PATTERN,
    RECEIPT_FIELD_LABELS,
    TAX_ID_PATTERN,
    TEXT_DATE_DMY_PATTERN,
)


class TestExtractionPatterns(unittest.TestCase):
    """Test regex pattern matching accuracy across dates, money, emails, phones, and IDs."""

    def test_iso_and_text_date_patterns(self) -> None:
        # ISO
        m = ISO_DATE_PATTERN.search("Date of issue: 2026-10-01 on record")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("year"), "2026")
        self.assertEqual(m.group("month"), "10")
        self.assertEqual(m.group("day"), "01")

        # Text Date
        m2 = TEXT_DATE_DMY_PATTERN.search("Issued on 15 March 2026 at headquarters")
        self.assertIsNotNone(m2)
        self.assertEqual(m2.group("day"), "15")
        self.assertEqual(m2.group("month"), "March")
        self.assertEqual(m2.group("year"), "2026")

    def test_money_pattern_symbols_and_formats(self) -> None:
        # Dollar
        m1 = MONEY_PATTERN.search("Amount: $1,250.00 USD")
        self.assertIsNotNone(m1)
        self.assertEqual(m1.group("currency"), "$")
        self.assertEqual(m1.group("amount"), "1,250.00")

        # Rupee
        m2 = MONEY_PATTERN.search("Total: ₹ 4500.50")
        self.assertIsNotNone(m2)
        self.assertEqual(m2.group("currency"), "₹")

        # Euro with comma
        m3 = MONEY_PATTERN.search("Total: 1.250,50 EUR")
        self.assertIsNotNone(m3)
        self.assertEqual(m3.group("amount"), "1.250,50")

    def test_email_pattern(self) -> None:
        self.assertIsNotNone(EMAIL_PATTERN.search("contact billing.support@company.org for help"))
        self.assertIsNone(EMAIL_PATTERN.search("invalid-email@"))

    def test_phone_pattern(self) -> None:
        self.assertIsNotNone(PHONE_PATTERN.search("Tel: +1 (555) 123-4567"))
        self.assertIsNotNone(PHONE_PATTERN.search("Direct: +91 98765 43210"))
        self.assertIsNotNone(PHONE_PATTERN.search("Phone: 555-123-4567"))

    def test_identifier_patterns(self) -> None:
        self.assertIsNotNone(INVOICE_ID_PATTERN.search("Invoice #: INV-2026-0092"))
        self.assertIsNotNone(GENERIC_ID_PATTERN.search("Order ID: ORD-9812-B"))
        self.assertIsNotNone(TAX_ID_PATTERN.search("Tax ID: VAT-98765432"))

    def test_label_dictionaries_completeness(self) -> None:
        self.assertIn("invoice_number", INVOICE_FIELD_LABELS)
        self.assertIn("total", INVOICE_FIELD_LABELS)
        self.assertIn("merchant_name", RECEIPT_FIELD_LABELS)


if __name__ == "__main__":
    unittest.main()
