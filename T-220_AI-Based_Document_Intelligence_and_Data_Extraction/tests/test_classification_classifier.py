"""Unit tests for RuleBasedDocumentClassifier across all supported document types and edge cases."""

import unittest
from pathlib import Path

from src.classification.classifier import RuleBasedDocumentClassifier
from src.classification.exceptions import (
    ClassificationConfigurationError,
    InvalidClassificationInput,
)
from src.core.config import ClassificationConfig
from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
    OCRTextRegion,
)
from src.core.types import DocumentType


def create_mock_document(doc_id: str, pages_text: list[str]) -> Document:
    """Helper to create a multi-page or single-page Document with text and mock OCR regions."""
    meta = DocumentMetadata(
        document_id=doc_id,
        filename=f"{doc_id}.pdf",
        file_path=Path(f"data/{doc_id}.pdf"),
        file_type="application/pdf",
        file_size_bytes=1024,
        checksum_sha256="dummy_sha256",
        page_count=len(pages_text),
    )
    pages = []
    for idx, txt in enumerate(pages_text, start=1):
        regions = [
            OCRTextRegion(
                text=line,
                page_number=idx,
                confidence=0.95,
                bounding_box=BoundingBox(xmin=10.0, ymin=float(i * 20), xmax=300.0, ymax=float(i * 20 + 15)),
            )
            for i, line in enumerate(txt.split("\n"))
            if line.strip()
        ]
        pages.append(
            DocumentPage(
                page_number=idx,
                raw_text=txt,
                ocr_text_regions=regions,
            )
        )
    return Document(metadata=meta, pages=pages)


class TestClassificationClassifier(unittest.TestCase):
    """Test classifier behavior, multi-class discrimination, ambiguity handling, and configuration."""

    def setUp(self) -> None:
        self.config = ClassificationConfig(
            confidence_threshold=0.60,
            min_score=2.0,
            min_evidence_count=1,
            ambiguity_margin=0.10,
            header_weight_multiplier=1.25,
        )
        self.classifier = RuleBasedDocumentClassifier(config=self.config)

    def test_classify_invoice_document(self) -> None:
        invoice_text = (
            "TAX INVOICE\n"
            "Invoice Number: INV-2026-9812\n"
            "Invoice Date: 2026-03-15\n"
            "Bill To: Acme Industrial Corp\n"
            "Remit To: Global Logistics LLC\n"
            "Payment Terms: NET 30\n"
            "Due Date: 2026-04-14\n"
            "Subtotal: $1,250.00\n"
            "Amount Due: $1,375.00\n"
        )
        doc = create_mock_document("doc_inv_01", [invoice_text])
        result = self.classifier.classify(doc)

        self.assertEqual(result.document_type, DocumentType.INVOICE)
        self.assertGreaterEqual(result.confidence, 0.60)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(doc.classified_type, DocumentType.INVOICE)
        self.assertGreater(len(result.evidence), 3)

        # Verify top evidence
        top_ev = result.get_top_evidence()
        self.assertTrue(any("invoice" in e.signal_name for e in top_ev))

    def test_classify_receipt_document(self) -> None:
        receipt_text = (
            "STORE RECEIPT\n"
            "Store # 4821 - Central Mall\n"
            "Cashier: Robert M.\n"
            "Register # 04\n"
            "Items Sold: 3\n"
            "Subtotal: $45.50\n"
            "Tax: $3.64\n"
            "Total: $49.14\n"
            "Cash Tendered: $50.00\n"
            "Change Due: $0.86\n"
            "Thank you for shopping with us!\n"
            "Please come again\n"
        )
        doc = create_mock_document("doc_rec_01", [receipt_text])
        result = self.classifier.classify(doc)

        self.assertEqual(result.document_type, DocumentType.RECEIPT)
        self.assertGreaterEqual(result.confidence, 0.60)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(doc.classified_type, DocumentType.RECEIPT)

    def test_classify_form_document(self) -> None:
        form_text = (
            "APPLICATION FORM FOR PATIENT ENROLLMENT\n"
            "Form No: ENR-2026-V1\n"
            "Please print clearly in ink.\n"
            "Applicant Name: ____________________\n"
            "Date of Birth (DOB): _______________\n"
            "Social Security (SSN): _____________\n"
            "Emergency Contact: _________________\n"
            "Section A: General Demographics\n"
            "Signature of Applicant: ____________ Date: ______\n"
            "FOR OFFICE USE ONLY [ ] Approved [ ] Denied\n"
        )
        doc = create_mock_document("doc_form_01", [form_text])
        result = self.classifier.classify(doc)

        self.assertEqual(result.document_type, DocumentType.FORM)
        self.assertGreaterEqual(result.confidence, 0.60)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(doc.classified_type, DocumentType.FORM)

    def test_classify_general_document(self) -> None:
        general_text = (
            "MEMORANDUM\n"
            "TO: All Engineering Staff\n"
            "FROM: Technical Director\n"
            "SUBJECT: Q3 Architectural Review & Roadmap\n"
            "EXECUTIVE SUMMARY\n"
            "This document outlines the strategic priorities for the upcoming quarter.\n"
            "1. Introduction\n"
            "The platform has achieved 99.9% uptime across all staging environments.\n"
            "2. Background\n"
            "3. Conclusion\n"
            "Best regards,\n"
            "Engineering Leadership\n"
        )
        doc = create_mock_document("doc_gen_01", [general_text])
        result = self.classifier.classify(doc)

        self.assertEqual(result.document_type, DocumentType.GENERAL_DOCUMENT)
        self.assertGreaterEqual(result.confidence, 0.60)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(doc.classified_type, DocumentType.GENERAL_DOCUMENT)

    def test_classify_unknown_document_zero_signals(self) -> None:
        gibberish_text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
        doc = create_mock_document("doc_unk_01", [gibberish_text])
        result = self.classifier.classify(doc)

        self.assertEqual(result.document_type, DocumentType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(doc.classified_type, DocumentType.UNKNOWN)

    def test_classify_ambiguous_document_falls_back_or_flags(self) -> None:
        # Document containing nearly identical competing signals for two classes
        custom_cfg = ClassificationConfig(
            min_score=2.0,
            ambiguity_margin=0.15,
            confidence_threshold=0.60,
            custom_rules={
                "form": [{"name": "sig:alpha", "keyword": "ALPHA_SIGNAL", "weight": 5.0}],
                "general_document": [{"name": "sig:beta", "keyword": "BETA_SIGNAL", "weight": 5.0}],
            },
        )
        classifier = RuleBasedDocumentClassifier(config=custom_cfg)
        conflicting_text = "Document header:\nALPHA_SIGNAL detected.\nBETA_SIGNAL detected."
        doc = create_mock_document("doc_ambig_01", [conflicting_text])
        result = classifier.classify(doc)

        self.assertTrue(result.is_ambiguous)
        self.assertIsNotNone(result.ambiguity_reason)

    def test_multi_page_document_aggregation(self) -> None:
        page1 = "TAX INVOICE\nBill To: Global Corp\nInvoice Date: 2026-02-01"
        page2 = "Subtotal: $5,000.00\nPayment Terms: NET 30\nAmount Due: $5,500.00"
        doc = create_mock_document("doc_multi_01", [page1, page2])
        result = self.classifier.classify(doc)

        self.assertEqual(result.document_type, DocumentType.INVOICE)
        self.assertGreaterEqual(result.confidence, 0.60)

        # Verify evidence spans both pages
        p1_ev = result.get_page_evidence(page_number=1)
        p2_ev = result.get_page_evidence(page_number=2)
        self.assertGreater(len(p1_ev), 0)
        self.assertGreater(len(p2_ev), 0)

    def test_header_multiplier_boost_on_page_one(self) -> None:
        # "Tax Invoice" at the very beginning of page 1 vs page 2
        header_text = "TAX INVOICE\nDetails follow..."
        doc1 = create_mock_document("doc_header_p1", [header_text])
        res1 = self.classifier.classify(doc1)
        ev1 = [e for e in res1.evidence if "tax invoice" in e.signal_name][0]

        doc2 = create_mock_document("doc_header_p2", ["Some text...\n" * 20 + "TAX INVOICE"])
        res2 = self.classifier.classify(doc2)
        ev2 = [e for e in res2.evidence if "tax invoice" in e.signal_name][0]

        # ev1 should have received the header multiplier
        self.assertGreater(ev1.weight, ev2.weight)

    def test_invalid_input_raises_invalid_classification_input(self) -> None:
        with self.assertRaises(InvalidClassificationInput):
            self.classifier.classify(None)  # type: ignore

        with self.assertRaises(InvalidClassificationInput):
            self.classifier.classify("not a document")  # type: ignore

    def test_custom_rules_merging(self) -> None:
        custom_cfg = ClassificationConfig(
            custom_rules={
                "invoice": [
                    {"name": "custom:special_bill", "keyword": "SuperBill 9000", "weight": 5.0}
                ]
            }
        )
        custom_classifier = RuleBasedDocumentClassifier(config=custom_cfg)
        doc = create_mock_document("doc_custom_01", ["SuperBill 9000 is ready"])
        res = custom_classifier.classify(doc)

        self.assertEqual(res.document_type, DocumentType.INVOICE)
        self.assertTrue(any(e.signal_name == "custom:special_bill" for e in res.evidence))

    def test_missing_custom_rules_file_raises_error(self) -> None:
        cfg = ClassificationConfig(keyword_weights_path="non_existent_rules.yaml")
        with self.assertRaises(ClassificationConfigurationError):
            RuleBasedDocumentClassifier(config=cfg)


if __name__ == "__main__":
    unittest.main()
