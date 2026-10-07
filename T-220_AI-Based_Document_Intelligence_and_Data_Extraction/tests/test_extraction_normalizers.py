"""Unit tests for text, currency, date, and identifier normalization functions."""

import unittest
from src.extraction.normalizers import (
    normalize_date,
    normalize_email,
    normalize_identifier,
    normalize_money,
    normalize_phone,
    normalize_text,
)


class TestExtractionNormalizers(unittest.TestCase):
    """Test value normalization algorithms and edge cases."""

    def test_normalize_text_whitespace(self) -> None:
        self.assertEqual(normalize_text("  Invoice    Number:   \t INV-1001 \n "), "Invoice Number: INV-1001")
        self.assertEqual(normalize_text(None), "")

    def test_normalize_money_formats(self) -> None:
        # Standard US format
        amt, curr = normalize_money("$1,250.50")
        self.assertEqual(amt, 1250.50)
        self.assertEqual(curr, "USD")

        # European format: 1.250,50 €
        amt2, curr2 = normalize_money("1.250,50 €")
        self.assertEqual(amt2, 1250.50)
        self.assertEqual(curr2, "EUR")

        # Indian Rupee
        amt3, curr3 = normalize_money("₹ 4,500.00")
        self.assertEqual(amt3, 4500.00)
        self.assertEqual(curr3, "INR")

        # Plain number
        amt4, curr4 = normalize_money("4950.00")
        self.assertEqual(amt4, 4950.00)
        self.assertIsNone(curr4)

    def test_normalize_date_unambiguous(self) -> None:
        # ISO
        d1, amb1 = normalize_date("2026-03-25")
        self.assertEqual(d1, "2026-03-25")
        self.assertFalse(amb1)

        # Textual
        d2, amb2 = normalize_date("25 March 2026")
        self.assertEqual(d2, "2026-03-25")
        self.assertFalse(amb2)

        # Unambiguous numeric (day 25 > 12)
        d3, amb3 = normalize_date("25/03/2026")
        self.assertEqual(d3, "2026-03-25")
        self.assertFalse(amb3)

    def test_normalize_date_ambiguous_numeric_flags_ambiguity(self) -> None:
        # 01/02/2026 (Both 01 and 02 <= 12)
        d_dmy, amb_dmy = normalize_date("01/02/2026", default_order="DMY")
        self.assertEqual(d_dmy, "2026-02-01")
        self.assertTrue(amb_dmy)

        d_mdy, amb_mdy = normalize_date("01/02/2026", default_order="MDY")
        self.assertEqual(d_mdy, "2026-01-02")
        self.assertTrue(amb_mdy)

    def test_normalize_email_and_phone(self) -> None:
        self.assertEqual(normalize_email("   John.Doe@Company.COM  "), "john.doe@company.com")
        self.assertEqual(normalize_phone("+1 (555) 123-4567"), "+15551234567")
        self.assertEqual(normalize_phone("555.123.4567"), "5551234567")

    def test_normalize_identifier(self) -> None:
        self.assertEqual(normalize_identifier("Invoice #: INV-9810-A:"), "INV-9810-A")
        self.assertEqual(normalize_identifier("###PO-2026-X12###"), "PO-2026-X12")


if __name__ == "__main__":
    unittest.main()
