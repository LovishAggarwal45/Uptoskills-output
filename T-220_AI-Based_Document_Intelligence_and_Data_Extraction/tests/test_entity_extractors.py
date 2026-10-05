"""Unit tests for GenericEntityExtractor on multi-entity document text."""

import unittest
from pathlib import Path
from src.core.models import Document, DocumentMetadata, DocumentPage
from src.extraction.entity_extractors import GenericEntityExtractor
from src.extraction.models import EntityType


class TestEntityExtractors(unittest.TestCase):
    """Test generic entity extraction across dates, money, emails, phones, IDs, orgs, and addresses."""

    def test_extract_all_generic_entities(self) -> None:
        raw_text = (
            "Acme Global Logistics LLC\n"
            "123 Innovation Boulevard, Suite 400, Chicago IL 60601\n"
            "Contact: billing-team@acmeglobal.com\n"
            "Direct Support: +1 (555) 892-4100\n"
            "Statement Date: 2026-10-01\n"
            "Account Ref: ACC-9821-X4\n"
            "Current Balance: $4,500.00 USD\n"
        )
        meta = DocumentMetadata(
            document_id="doc_ent_01",
            filename="doc_ent_01.txt",
            file_path=Path("data/doc_ent_01.txt"),
            file_type="text/plain",
            file_size_bytes=len(raw_text),
            checksum_sha256="dummy_sha256",
            page_count=1,
        )
        page = DocumentPage(page_number=1, raw_text=raw_text)
        doc = Document(metadata=meta, pages=[page])

        extractor = GenericEntityExtractor()
        entities = extractor.extract_entities(doc)

        types_found = {e.entity_type for e in entities}
        self.assertIn(EntityType.ORGANIZATION, types_found)
        self.assertIn(EntityType.EMAIL, types_found)
        self.assertIn(EntityType.PHONE, types_found)
        self.assertIn(EntityType.DATE, types_found)
        self.assertIn(EntityType.IDENTIFIER, types_found)
        self.assertIn(EntityType.MONEY, types_found)

        # Check values
        email_ent = [e for e in entities if e.entity_type == EntityType.EMAIL][0]
        self.assertEqual(email_ent.normalized_value, "billing-team@acmeglobal.com")

        money_ent = [e for e in entities if e.entity_type == EntityType.MONEY][0]
        self.assertEqual(money_ent.normalized_value["amount"], 4500.00)
        self.assertEqual(money_ent.normalized_value["currency"], "USD")


if __name__ == "__main__":
    unittest.main()
