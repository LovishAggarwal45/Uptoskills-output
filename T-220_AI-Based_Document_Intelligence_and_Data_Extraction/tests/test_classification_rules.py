"""Unit tests for classification rule definitions, signal matching, and provenance location."""

import unittest

from src.classification.models import SignalType
from src.classification.rules import (
    RuleDefinition,
    get_default_classification_rules,
    locate_text_provenance,
)
from src.core.models import BoundingBox, DocumentPage, OCRTextRegion
from src.core.types import DocumentType


class TestClassificationRules(unittest.TestCase):
    """Test rule parsing, keyword/regex matching, and spatial OCR provenance linking."""

    def test_default_rules_coverage(self) -> None:
        rules = get_default_classification_rules()
        self.assertGreater(len(rules), 30)

        target_types = {r.target_type for r in rules}
        self.assertIn(DocumentType.INVOICE, target_types)
        self.assertIn(DocumentType.RECEIPT, target_types)
        self.assertIn(DocumentType.FORM, target_types)
        self.assertIn(DocumentType.GENERAL_DOCUMENT, target_types)

    def test_phrase_and_keyword_matching_boundaries(self) -> None:
        rule_invoice = RuleDefinition(
            name="test_invoice",
            target_type=DocumentType.INVOICE,
            signal_type=SignalType.KEYWORD,
            pattern="invoice",
            weight=2.5,
        )

        # Should match the exact word.
        matches1 = rule_invoice.find_matches(
            "This is a TAX INVOICE for your order."
        )
        self.assertEqual(len(matches1), 1)
        self.assertEqual(matches1[0][0].upper(), "INVOICE")

        # Should not match as part of an unrelated long word.
        matches2 = rule_invoice.find_matches(
            "This document is noninvoiceable."
        )
        self.assertEqual(len(matches2), 0)

    def test_regex_rule_matching(self) -> None:
        rule_inv_id = RuleDefinition(
            name="test_regex",
            target_type=DocumentType.INVOICE,
            signal_type=SignalType.REGEX,
            pattern=(
                r"\b(?:INV[-:#\s]+[A-Z0-9][A-Z0-9-]{2,}"
                r"|INVOICE[-:#\s]+[0-9][A-Z0-9-]{2,})\b"
            ),
            weight=3.0,
            is_regex=True,
        )

        # A genuine structured invoice ID should match.
        text = "Reference code: INV-2026-98234-A on file."
        matches = rule_inv_id.find_matches(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "INV-2026-98234-A")

        # "Invoice Number" alone must not be treated as an invoice ID.
        false_positive = rule_inv_id.find_matches(
            "Indian Railways GST Details: Invoice Number:"
        )
        self.assertEqual(len(false_positive), 0)

        # A numeric ID following "Invoice" should match.
        numeric_id = rule_inv_id.find_matches("Invoice: 12345")
        self.assertEqual(len(numeric_id), 1)
        self.assertEqual(numeric_id[0][0], "Invoice: 12345")

    def test_negative_rule_penalties(self) -> None:
        neg_rule = RuleDefinition(
            name="test_neg",
            target_type=DocumentType.INVOICE,
            signal_type=SignalType.NEGATIVE_SIGNAL,
            pattern="cashier",
            weight=-2.5,
            is_negative=True,
        )

        self.assertTrue(neg_rule.is_negative)
        self.assertEqual(neg_rule.weight, -2.5)

        matches = neg_rule.find_matches(
            "Served by cashier 04 at register 2"
        )
        self.assertEqual(len(matches), 1)

    def test_locate_text_provenance(self) -> None:
        bbox1 = BoundingBox(
            xmin=50.0,
            ymin=50.0,
            xmax=150.0,
            ymax=80.0,
        )
        region1 = OCRTextRegion(
            text="Tax Invoice",
            page_number=1,
            confidence=0.98,
            bounding_box=bbox1,
        )
        page = DocumentPage(
            page_number=1,
            raw_text="Tax Invoice\nBill To: Acme Corp",
            ocr_text_regions=[region1],
        )

        found_box, found_conf = locate_text_provenance(
            "Tax Invoice", page
        )
        self.assertIsNotNone(found_box)
        self.assertIsNotNone(found_conf)
        self.assertEqual(found_box.xmin, 50.0)
        self.assertEqual(found_conf, 0.98)


if __name__ == "__main__":
    unittest.main()