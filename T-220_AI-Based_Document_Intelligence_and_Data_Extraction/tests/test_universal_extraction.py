"""Unit tests for universal extraction, resume extraction, general document extraction, and entities."""

import unittest
from typing import Any, Dict, List, Optional

from src.classification.models import ClassificationResult
from src.core.config import ExtractionConfig
from src.core.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    DocumentPage,
)
from src.ocr.models import OCRTextRegion, OCRWord
from src.core.types import DocumentType, FieldType
from src.extraction.entity_extractors import GenericEntityExtractor
from src.extraction.extractor import DocumentExtractor
from src.extraction.general_extractor import GeneralDocumentExtractor
from src.extraction.models import EntityType
from src.extraction.resume_extractor import ResumeExtractor
from src.validation.required_field_validator import RequiredFieldValidator
from src.validation.validator import DocumentValidationEngine


def make_doc_page(
    page_num: int,
    text: str,
    ocr_words: Optional[List[Dict[str, Any]]] = None,
) -> DocumentPage:
    regions: List[OCRTextRegion] = []
    if ocr_words:
        for w in ocr_words:
            regions.append(
                OCRTextRegion(
                    text=w["text"],
                    page_number=page_num,
                    bounding_box=BoundingBox(
                        xmin=w.get("xmin", 0.1),
                        ymin=w.get("ymin", 0.1),
                        xmax=w.get("xmax", 0.3),
                        ymax=w.get("ymax", 0.15),
                    ),
                    confidence=w.get("conf", 0.95),
                )
            )
    else:
        # Generate default word-level regions
        for line in text.split("\n"):
            clean = line.strip()
            if clean:
                regions.append(
                    OCRTextRegion(
                        text=clean,
                        page_number=page_num,
                        bounding_box=BoundingBox(xmin=0.1, ymin=0.1, xmax=0.8, ymax=0.2),
                        confidence=0.92,
                    )
                )

    return DocumentPage(
        page_number=page_num,
        width=1000.0,
        height=1400.0,
        dpi=300,
        raw_text=text,
        ocr_text_regions=regions,
    )


class TestUniversalExtraction(unittest.TestCase):
    """Test extraction across diverse document domains and generic entities."""

    def setUp(self) -> None:
        self.config = ExtractionConfig()
        self.extractor = DocumentExtractor(config=self.config)

    def test_resume_extraction(self) -> None:
        """Verify ResumeExtractor extracts candidate contact, skills, education, and social links."""
        resume_text = (
            "VENKATA RAMARAJU\n"
            "Email: venkata.raju@example.com | Phone: +1 (555) 345-6789\n"
            "Location: San Francisco, CA\n"
            "LinkedIn: linkedin.com/in/venkataraju | GitHub: github.com/venkataraju\n\n"
            "SUMMARY\n"
            "Experienced Machine Learning and Full-Stack Software Engineer with 6+ years in AI systems.\n\n"
            "TECHNICAL SKILLS\n"
            "Python, PyTorch, FastAPI, React, TypeScript, Docker, Kubernetes, PostgreSQL\n\n"
            "EDUCATION\n"
            "Bachelor of Technology in Computer Science, University of California, Berkeley, GPA: 3.85/4.0\n\n"
            "WORK EXPERIENCE\n"
            "Senior AI Engineer at Apex Systems Inc (2022 - Present)\n"
            "- Designed real-time document extraction pipelines.\n"
        )

        page = make_doc_page(1, resume_text)
        meta = DocumentMetadata(
            document_id="doc_resume_01",
            filename="venkata_resume.pdf",
            file_path="venkata_resume.pdf",
            file_type="application/pdf",
            file_size_bytes=50000,
            checksum_sha256="abc123hash",
            page_count=1,
        )
        doc = Document(metadata=meta, pages=[page], classified_type=DocumentType.RESUME)

        clf = ClassificationResult(
            document_type=DocumentType.RESUME,
            confidence=0.95,
            method="rule_based",
        )

        result = self.extractor.extract(doc, clf)
        self.assertEqual(result.document_type, DocumentType.RESUME)

        # Verify candidate contact details
        self.assertIn("candidate_name", result.fields)
        self.assertIn("VENKATA RAMARAJU", result.fields["candidate_name"].value)

        self.assertIn("email", result.fields)
        self.assertEqual(result.fields["email"].value, "venkata.raju@example.com")

        self.assertIn("phone", result.fields)
        self.assertIn("555", result.fields["phone"].value)

        self.assertIn("linkedin_url", result.fields)
        self.assertIn("linkedin.com/in/venkataraju", result.fields["linkedin_url"].value)

        self.assertIn("github_url", result.fields)
        self.assertIn("github.com/venkataraju", result.fields["github_url"].value)

        self.assertIn("location", result.fields)
        self.assertIn("San Francisco", result.fields["location"].value)

        # Verify sections extracted
        self.assertIn("skills", result.fields)
        self.assertIn("Python", result.fields["skills"].value)

        self.assertIn("education", result.fields)
        self.assertIn("Bachelor of Technology", result.fields["education"].value)

        self.assertIn("experience", result.fields)
        self.assertIn("Senior AI Engineer", result.fields["experience"].value)

        # Ensure NO invoice fields are hallucinated
        self.assertNotIn("invoice_number", result.fields)
        self.assertNotIn("tax", result.fields)
        self.assertNotIn("total", result.fields)

    def test_general_document_extraction(self) -> None:
        """Verify GeneralDocumentExtractor parses titles, organizations, and key-values."""
        report_text = (
            "Annual Technical Operations Report 2026\n"
            "Author: Dr. Evelyn Vance\n"
            "Organization: CloudScale Global Solutions LLC\n"
            "Date: 2026-09-15\n"
            "Department: Infrastructure Engineering\n"
            "Status: Approved\n\n"
            "EXECUTIVE SUMMARY\n"
            "This report outlines our quarterly infrastructure stability, cloud migration metrics, and SLA compliance.\n"
        )

        page = make_doc_page(1, report_text)
        meta = DocumentMetadata(
            document_id="doc_report_01",
            filename="operations_report.pdf",
            file_path="operations_report.pdf",
            file_type="application/pdf",
            file_size_bytes=40000,
            checksum_sha256="rep123hash",
            page_count=1,
        )
        doc = Document(metadata=meta, pages=[page], classified_type=DocumentType.GENERAL_DOCUMENT)

        clf = ClassificationResult(
            document_type=DocumentType.GENERAL_DOCUMENT,
            confidence=0.90,
            method="rule_based",
        )

        result = self.extractor.extract(doc, clf)
        self.assertEqual(result.document_type, DocumentType.GENERAL_DOCUMENT)

        self.assertIn("document_title", result.fields)
        self.assertIn("Author", report_text)
        self.assertIn("author", result.fields)
        self.assertEqual(result.fields["author"].value, "Dr. Evelyn Vance")

        self.assertIn("organization", result.fields)
        self.assertIn("CloudScale Global Solutions LLC", result.fields["organization"].value)

        self.assertIn("document_date", result.fields)
        self.assertIn("summary", result.fields)

        # Check key-values
        self.assertIn("department", result.fields)
        self.assertEqual(result.fields["department"].value, "Infrastructure Engineering")

    def test_universal_entities_extraction(self) -> None:
        """Verify GenericEntityExtractor extracts all semantic entity types."""
        text = (
            "Contact info: alice.smith@enterprise.org, Phone: +1 800 555 1212\n"
            "Check profile at https://github.com/alicesmith or linkedin.com/in/alicesmith\n"
            "Payment of $1,450.50 USD due on 2026-11-30. Reference code: REF-9948-AX.\n"
            "Location: 100 Main Street, Suite 400, Austin, TX 78701\n"
            "Issued by: Nexus Global Technologies Inc.\n"
        )
        page = make_doc_page(1, text)
        meta = DocumentMetadata(
            document_id="doc_ent_01",
            filename="entities_sample.pdf",
            file_path="entities_sample.pdf",
            file_type="application/pdf",
            file_size_bytes=30000,
            checksum_sha256="ent123hash",
            page_count=1,
        )
        doc = Document(metadata=meta, pages=[page])

        entity_extractor = GenericEntityExtractor(config=self.config)
        entities = entity_extractor.extract_entities(doc)

        ent_types = {e.entity_type.value for e in entities}
        self.assertIn(EntityType.EMAIL.value, ent_types)
        self.assertIn(EntityType.PHONE.value, ent_types)
        self.assertIn(EntityType.URL.value, ent_types)
        self.assertIn(EntityType.DATE.value, ent_types)
        self.assertIn(EntityType.MONEY.value, ent_types)
        self.assertIn(EntityType.IDENTIFIER.value, ent_types)
        self.assertIn(EntityType.ORGANIZATION.value, ent_types)
        self.assertIn(EntityType.ADDRESS.value, ent_types)

    def test_resume_passes_validation_without_invoice_errors(self) -> None:
        """Verify that a classified resume does NOT fail required invoice fields."""
        resume_text = (
            "JANE DOE\n"
            "Email: jane.doe@example.com\n"
            "Phone: +1 555 987 6543\n"
            "SKILLS: Python, SQL\n"
        )
        page = make_doc_page(1, resume_text)
        meta = DocumentMetadata(
            document_id="doc_val_res_01",
            filename="jane_doe_resume.pdf",
            file_path="jane_doe_resume.pdf",
            file_type="application/pdf",
            file_size_bytes=25000,
            checksum_sha256="valres123",
            page_count=1,
        )
        doc = Document(metadata=meta, pages=[page], classified_type=DocumentType.RESUME)

        clf = ClassificationResult(document_type=DocumentType.RESUME, confidence=0.92, method="rule_based")
        ext_result = self.extractor.extract(doc, clf)

        val_engine = DocumentValidationEngine()
        val_report = val_engine.validate_document(
            document_id=doc.id,
            document_type=doc.classified_type,
            fields=ext_result.fields,
            tables=[],
        )

        # Must have zero missing-invoice-field errors
        missing_errors = [
            i for i in val_report.issues
            if "invoice_number" in i.affected_fields or "total" in i.affected_fields
        ]
        self.assertEqual(len(missing_errors), 0)
        self.assertEqual(val_report.rules_failed, 0)


if __name__ == "__main__":
    unittest.main()
