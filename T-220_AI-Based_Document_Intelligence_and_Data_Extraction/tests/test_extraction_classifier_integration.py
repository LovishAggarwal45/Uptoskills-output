"""Integration tests verifying DocumentExtractor dispatching driven by classification."""

import unittest
from pathlib import Path
from src.classification.models import ClassificationEvidence, ClassificationResult, SignalType
from src.core.models import Document, DocumentMetadata, DocumentPage
from src.core.types import DocumentType
from src.extraction.exceptions import InvalidExtractionInput
from src.extraction.extractor import DocumentExtractor


def create_mock_doc(doc_id: str, text: str) -> Document:
    meta = DocumentMetadata(
        document_id=doc_id,
        filename=f"{doc_id}.pdf",
        file_path=Path(f"data/{doc_id}.pdf"),
        file_type="application/pdf",
        file_size_bytes=len(text),
        checksum_sha256="dummy_sha",
        page_count=1,
    )
    page = DocumentPage(page_number=1, raw_text=text)
    return Document(metadata=meta, pages=[page])


def create_mock_classification(doc_id: str, doc_type: DocumentType) -> ClassificationResult:
    return ClassificationResult(
        document_type=doc_type,
        confidence=0.95,
        method="rule_based",
        evidence=[
            ClassificationEvidence(
                signal_name="keyword_signal",
                signal_type=SignalType.KEYWORD,
                matched_text=doc_type.value,
                weight=0.95,
                target_type=doc_type,
                page_number=1,
            )
        ],
    )


class TestExtractionClassifierIntegration(unittest.TestCase):
    """Test classification-driven extraction dispatching and candidate tracking."""

    def setUp(self) -> None:
        self.extractor = DocumentExtractor()

    def test_invalid_input_none_raises_error(self) -> None:
        with self.assertRaises(InvalidExtractionInput):
            self.extractor.extract_document(None, None)

    def test_invoice_dispatch_and_unresolved_fields(self) -> None:
        text = (
            "Acme Global Inc\n"
            "Invoice Number: INV-8921\n"
            "Invoice Date: 2026-10-01\n"
        )
        doc = create_mock_doc("inv_01", text)
        clf = create_mock_classification("inv_01", DocumentType.INVOICE)

        result = self.extractor.extract_document(doc, clf)
        self.assertEqual(result.document_type, DocumentType.INVOICE)
        self.assertIn("invoice_number", result.fields)
        self.assertIn("invoice_date", result.fields)
        self.assertNotIn("total", result.fields)
        # Verify required missing field 'total' is recorded in unresolved_fields
        self.assertIn("total", result.unresolved_fields)
        self.assertGreater(result.extraction_statistics["field_count"], 0)

    def test_receipt_dispatch(self) -> None:
        text = (
            "Starbucks Coffee\n"
            "Date: 2026-10-01\n"
            "Order #: 492\n"
            "Subtotal: $4.50\n"
            "Tax: $0.45\n"
            "Total: $4.95\n"
            "Visa: ************1234\n"
        )
        doc = create_mock_doc("rcpt_01", text)
        clf = create_mock_classification("rcpt_01", DocumentType.RECEIPT)

        result = self.extractor.extract_document(doc, clf)
        self.assertEqual(result.document_type, DocumentType.RECEIPT)
        self.assertIn("total", result.fields)
        self.assertEqual(result.fields["total"].normalized_value, 4.95)
        self.assertIn("subtotal", result.fields)

    def test_form_dispatch(self) -> None:
        text = (
            "PATIENT INTAKE FORM\n"
            "Form #: MED-991\n"
            "Full Name: John Doe\n"
            "DOB: 1985-05-20\n"
            "Email: john.doe@health.org\n"
        )
        doc = create_mock_doc("form_01", text)
        clf = create_mock_classification("form_01", DocumentType.FORM)

        result = self.extractor.extract_document(doc, clf)
        self.assertEqual(result.document_type, DocumentType.FORM)
        self.assertIn("applicant_name", result.fields)
        self.assertEqual(result.fields["applicant_name"].value, "John Doe")
        self.assertIn("date_of_birth", result.fields)

    def test_general_document_only_extracts_generic_entities(self) -> None:
        text = (
            "Quarterly Review Report\n"
            "Authored on 2026-10-01 by Corporate Strategy Group LLC\n"
            "For queries contact strategy@company.org or call +1 555 234 5678\n"
            "Budget allocated: $75,000.00 USD\n"
        )
        doc = create_mock_doc("gen_01", text)
        clf = create_mock_classification("gen_01", DocumentType.GENERAL_DOCUMENT)

        result = self.extractor.extract_document(doc, clf)
        self.assertEqual(result.document_type, DocumentType.GENERAL_DOCUMENT)
        # Should not force invoice/receipt fields on general documents
        self.assertNotIn("invoice_number", result.fields)
        self.assertNotIn("total", result.fields)
        self.assertNotIn("merchant_name", result.fields)
        # Should extract generic entities
        self.assertGreater(len(result.entities), 0)
        ent_types = {e.entity_type.value for e in result.entities}
        self.assertIn("email", ent_types)
        self.assertIn("date", ent_types)
        self.assertIn("money", ent_types)

    def test_unknown_document_fallback_to_entities(self) -> None:
        text = "Random unstructured notes: Call +1 555 444 3322 by 2026-11-15 regarding $500 payment."
        doc = create_mock_doc("unk_01", text)
        clf = create_mock_classification("unk_01", DocumentType.UNKNOWN)

        result = self.extractor.extract_document(doc, clf)
        self.assertEqual(result.document_type, DocumentType.UNKNOWN)
        self.assertEqual(len(result.fields), 0)
        self.assertGreater(len(result.entities), 0)


if __name__ == "__main__":
    unittest.main()
