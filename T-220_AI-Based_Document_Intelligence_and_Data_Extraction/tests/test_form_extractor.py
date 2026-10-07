"""Unit tests for FormExtractor on structured application and intake forms."""

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
from src.extraction.form_extractor import FormExtractor


def build_form_doc(doc_id: str, text: str) -> Document:
    lines = text.split("\n")
    regions = []
    y_curr = 50
    for l in lines:
        if not l.strip():
            continue
        regions.append(
            OCRTextRegion(
                page_number=1,
                text=l,
                bounding_box=BoundingBox(50, y_curr, 700, y_curr + 30),
                confidence=0.97,
            )
        )
        y_curr += 45

    meta = DocumentMetadata(
        document_id=doc_id,
        filename=f"{doc_id}.pdf",
        file_path=Path(f"data/{doc_id}.pdf"),
        file_type="application/pdf",
        file_size_bytes=len(text),
        checksum_sha256="dummy_sha256",
        page_count=1,
    )
    page = DocumentPage(page_number=1, raw_text=text, ocr_text_regions=regions)
    return Document(metadata=meta, pages=[page])


class TestFormExtractor(unittest.TestCase):
    """Test field extraction on application, registration, and survey forms."""

    def test_extract_applicant_and_contact_fields(self) -> None:
        form_text = (
            "APPLICATION FOR MEMBERSHIP\n"
            "Form No: APP-2026-0042\n"
            "Full Name: Eleanor Vance\n"
            "Date of Birth: 1990-08-14\n"
            "Email Address: eleanor.vance@hillhouse.org\n"
            "Phone: +1 (555) 987-6543\n"
            "Address: 450 Blackwood Terrace, Boston MA 02108\n"
            "Signature: [Electronically Signed by Eleanor Vance]\n"
        )
        doc = build_form_doc("form_test_01", form_text)
        extractor = FormExtractor()
        fields = extractor.extract(doc)

        # 1. Form Number
        self.assertIn("form_number", fields)
        self.assertEqual(fields["form_number"].normalized_value, "APP-2026-0042")
        self.assertEqual(fields["form_number"].field_type, FieldType.IDENTIFIER)

        # 2. Applicant Name
        self.assertIn("applicant_name", fields)
        self.assertEqual(fields["applicant_name"].value, "Eleanor Vance")
        self.assertEqual(fields["applicant_name"].field_type, FieldType.STRING)

        # 3. Date of Birth
        self.assertIn("date_of_birth", fields)
        self.assertEqual(fields["date_of_birth"].normalized_value, "1990-08-14")
        self.assertEqual(fields["date_of_birth"].field_type, FieldType.DATE)

        # 4. Email & Phone
        self.assertIn("email", fields)
        self.assertEqual(fields["email"].normalized_value, "eleanor.vance@hillhouse.org")
        self.assertEqual(fields["email"].field_type, FieldType.EMAIL)

        self.assertIn("phone", fields)
        self.assertEqual(fields["phone"].normalized_value, "+15559876543")
        self.assertEqual(fields["phone"].field_type, FieldType.PHONE)

        # 5. Address
        self.assertIn("address", fields)
        self.assertIn("Boston MA", fields["address"].value)
        self.assertEqual(fields["address"].field_type, FieldType.ADDRESS)

        # 6. Signature indicator
        self.assertIn("signature_indicator", fields)
        self.assertTrue(fields["signature_indicator"].normalized_value)

        # 7. Check Provenance & OCR Confidence
        for f_name, f_obj in fields.items():
            self.assertEqual(f_obj.page_number, 1)
            self.assertIsNotNone(f_obj.provenance)
            self.assertEqual(f_obj.provenance.document_id, "form_test_01")
            self.assertGreaterEqual(f_obj.extraction_confidence, 0.50)


if __name__ == "__main__":
    unittest.main()
