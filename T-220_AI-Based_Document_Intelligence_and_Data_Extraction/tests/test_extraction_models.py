"""Unit tests for extraction domain models, serialization, and provenance properties."""

import unittest
from src.core.models import BoundingBox, ExtractedField, Provenance
from src.core.types import (
    ConfidenceSource,
    ExtractionMethod,
    FieldType,
    ValidationStatus,
)
from src.extraction.models import (
    EntityType,
    ExtractedEntity,
    ExtractionCandidate,
    ExtractionResult,
)


class TestExtractionModels(unittest.TestCase):
    """Test data structures, constraints, properties, and serialization for extraction."""

    def test_extraction_candidate_creation_and_serialization(self) -> None:
        bbox = BoundingBox(xmin=100.0, ymin=150.0, xmax=250.0, ymax=180.0)
        cand = ExtractionCandidate(
            value="INV-2026-001",
            raw_value="INV-2026-001",
            normalized_value="INV-2026-001",
            confidence=0.92,
            source_text="Invoice #: INV-2026-001",
            page_number=1,
            bounding_box=bbox,
            ocr_confidence=0.96,
            matched_label="invoice #",
        )

        self.assertEqual(cand.value, "INV-2026-001")
        self.assertEqual(cand.confidence, 0.92)
        self.assertEqual(cand.ocr_confidence, 0.96)
        self.assertNotEqual(cand.confidence, cand.ocr_confidence)

        # Serialization round-trip
        data = cand.to_dict()
        self.assertEqual(data["value"], "INV-2026-001")
        self.assertEqual(data["confidence"], 0.92)
        self.assertIsNotNone(data["bounding_box"])

        reconstructed = ExtractionCandidate.from_dict(data)
        self.assertEqual(reconstructed.value, cand.value)
        self.assertEqual(reconstructed.confidence, cand.confidence)
        self.assertEqual(reconstructed.ocr_confidence, cand.ocr_confidence)
        self.assertEqual(reconstructed.bounding_box.xmin, bbox.xmin)

    def test_extracted_entity_creation_and_serialization(self) -> None:
        bbox = BoundingBox(xmin=50.0, ymin=60.0, xmax=180.0, ymax=90.0)
        entity = ExtractedEntity(
            entity_type=EntityType.EMAIL,
            value="support@acme.com",
            normalized_value="support@acme.com",
            confidence=0.95,
            page_number=1,
            bounding_box=bbox,
            ocr_confidence=0.98,
            source_text="support@acme.com",
            character_span=(10, 26),
        )

        self.assertEqual(entity.entity_type, EntityType.EMAIL)
        self.assertEqual(entity.value, "support@acme.com")

        # Serialization
        data = entity.to_dict()
        self.assertEqual(data["entity_type"], "email")
        self.assertEqual(data["value"], "support@acme.com")
        self.assertEqual(data["character_span"], [10, 26])

        reconstructed = ExtractedEntity.from_dict(data)
        self.assertEqual(reconstructed.entity_type, EntityType.EMAIL)
        self.assertEqual(reconstructed.value, entity.value)
        self.assertEqual(reconstructed.character_span, (10, 26))

    def test_extracted_field_enriched_properties(self) -> None:
        bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=120.0, ymax=50.0)
        prov = Provenance(
            document_id="doc_test_01",
            page_number=2,
            raw_text="Total: $1,250.00",
            bounding_box=bbox,
            ocr_confidence=0.97,
            extraction_method=ExtractionMethod.KEY_VALUE_HEURISTIC,
        )
        field_obj = ExtractedField(
            name="total",
            value="$1,250.00",
            normalized_value=1250.00,
            field_type=FieldType.CURRENCY,
            provenance=prov,
            extraction_confidence=0.88,
            is_required=True,
            is_ambiguous=False,
            matched_label="total:",
        )

        # Verify convenience properties
        self.assertEqual(field_obj.page_number, 2)
        self.assertEqual(field_obj.bounding_box, bbox)
        self.assertEqual(field_obj.source_text, "Total: $1,250.00")
        self.assertEqual(field_obj.source_ocr_confidence, 0.97)
        self.assertEqual(field_obj.extraction_confidence, 0.88)
        self.assertNotEqual(field_obj.extraction_confidence, field_obj.source_ocr_confidence)

        # Serialization round-trip
        data = field_obj.to_dict()
        self.assertEqual(data["name"], "total")
        self.assertEqual(data["page_number"], 2)
        self.assertEqual(data["source_ocr_confidence"], 0.97)

        reconstructed = ExtractedField.from_dict(data)
        self.assertEqual(reconstructed.name, field_obj.name)
        self.assertEqual(reconstructed.normalized_value, 1250.00)
        self.assertEqual(reconstructed.provenance.ocr_confidence, 0.97)

    def test_extraction_result_queries_and_serialization(self) -> None:
        prov = Provenance(document_id="doc_res_01", page_number=1, raw_text="INV-1001", ocr_confidence=0.99)
        field_inv = ExtractedField(name="invoice_number", value="INV-1001", normalized_value="INV-1001", provenance=prov, extraction_confidence=0.95)
        entity_date = ExtractedEntity(entity_type=EntityType.DATE, value="2026-10-01", normalized_value="2026-10-01", confidence=0.90)

        result = ExtractionResult(
            document_id="doc_res_01",
            document_type="invoice",  # type: ignore
            fields={"invoice_number": field_inv},
            entities=[entity_date],
            unresolved_fields=["due_date"],
            warnings=["Ambiguity detected in subtotal"],
        )

        self.assertEqual(result.get_field("invoice_number").value, "INV-1001")
        self.assertIsNone(result.get_field("non_existent"))
        self.assertEqual(len(result.get_entities_by_type(EntityType.DATE)), 1)
        self.assertEqual(len(result.get_entities_by_type("date")), 1)
        self.assertEqual(len(result.get_entities_by_type(EntityType.EMAIL)), 0)

        # JSON Serialization
        json_str = result.to_json()
        self.assertIn("doc_res_01", json_str)
        self.assertIn("INV-1001", json_str)

        data = result.to_dict()
        reconstructed = ExtractionResult.from_dict(data)
        self.assertEqual(reconstructed.document_id, "doc_res_01")
        self.assertEqual(len(reconstructed.fields), 1)
        self.assertEqual(len(reconstructed.entities), 1)


if __name__ == "__main__":
    unittest.main()
