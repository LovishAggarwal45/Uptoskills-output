"""Unit tests for classification domain models and serialization contracts."""

import unittest
from src.classification.models import (
    ClassificationCandidate,
    ClassificationEvidence,
    ClassificationResult,
    SignalType,
)
from src.core.models import BoundingBox
from src.core.types import DocumentType


class TestClassificationModels(unittest.TestCase):
    """Test classification data structures, validation constraints, and serialization."""

    def test_classification_evidence_creation_and_serialization(self) -> None:
        bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=200.0, ymax=140.0)
        evidence = ClassificationEvidence(
            signal_name="invoice:tax invoice",
            signal_type=SignalType.PHRASE,
            matched_text="Tax Invoice",
            weight=4.0,
            target_type=DocumentType.INVOICE,
            page_number=1,
            bounding_box=bbox,
            ocr_confidence=0.975,
            character_span=(0, 11),
            details={"is_header": True},
        )

        self.assertEqual(evidence.signal_name, "invoice:tax invoice")
        self.assertEqual(evidence.target_type, DocumentType.INVOICE)
        self.assertEqual(evidence.weight, 4.0)
        self.assertEqual(evidence.page_number, 1)

        # Serialization round-trip
        data = evidence.to_dict()
        self.assertEqual(data["signal_name"], "invoice:tax invoice")
        self.assertEqual(data["signal_type"], "phrase")
        self.assertEqual(data["target_type"], "invoice")
        self.assertIsNotNone(data["bounding_box"])

        reconstructed = ClassificationEvidence.from_dict(data)
        self.assertEqual(reconstructed.signal_name, evidence.signal_name)
        self.assertEqual(reconstructed.weight, evidence.weight)
        self.assertEqual(reconstructed.ocr_confidence, evidence.ocr_confidence)
        self.assertEqual(reconstructed.page_number, evidence.page_number)
        self.assertEqual(reconstructed.bounding_box.xmin, bbox.xmin)

    def test_classification_evidence_invalid_constraints(self) -> None:
        with self.assertRaises(ValueError):
            ClassificationEvidence(
                signal_name="test",
                signal_type=SignalType.KEYWORD,
                matched_text="test",
                weight=1.0,
                target_type=DocumentType.INVOICE,
                page_number=0,  # Invalid: must be >= 1
            )

        with self.assertRaises(ValueError):
            ClassificationEvidence(
                signal_name="test",
                signal_type=SignalType.KEYWORD,
                matched_text="test",
                weight=1.0,
                target_type=DocumentType.INVOICE,
                page_number=1,
                ocr_confidence=1.5,  # Invalid: > 1.0
            )

    def test_classification_candidate_serialization(self) -> None:
        evidence = ClassificationEvidence(
            signal_name="invoice:bill to",
            signal_type=SignalType.PHRASE,
            matched_text="Bill To",
            weight=3.0,
            target_type=DocumentType.INVOICE,
            page_number=1,
        )
        candidate = ClassificationCandidate(
            document_type=DocumentType.INVOICE,
            raw_score=7.0,
            normalized_score=0.85,
            evidence_count=2,
            evidence=[evidence],
        )

        data = candidate.to_dict()
        self.assertEqual(data["document_type"], "invoice")
        self.assertEqual(data["raw_score"], 7.0)
        self.assertEqual(len(data["evidence"]), 1)

        reconstructed = ClassificationCandidate.from_dict(data)
        self.assertEqual(reconstructed.document_type, DocumentType.INVOICE)
        self.assertEqual(reconstructed.raw_score, 7.0)
        self.assertEqual(len(reconstructed.evidence), 1)

    def test_classification_result_evidence_queries_and_serialization(self) -> None:
        ev1 = ClassificationEvidence(
            signal_name="invoice:tax invoice",
            signal_type=SignalType.PHRASE,
            matched_text="Tax Invoice",
            weight=4.0,
            target_type=DocumentType.INVOICE,
            page_number=1,
        )
        ev2 = ClassificationEvidence(
            signal_name="invoice:subtotal",
            signal_type=SignalType.KEYWORD,
            matched_text="Subtotal",
            weight=1.5,
            target_type=DocumentType.INVOICE,
            page_number=2,
        )
        ev3 = ClassificationEvidence(
            signal_name="receipt:cashier",
            signal_type=SignalType.KEYWORD,
            matched_text="Cashier",
            weight=3.5,
            target_type=DocumentType.RECEIPT,
            page_number=1,
        )

        result = ClassificationResult(
            document_type=DocumentType.INVOICE,
            confidence=0.82,
            method="rule_based",
            candidate_scores={"invoice": 0.82, "receipt": 0.18},
            evidence=[ev1, ev2, ev3],
        )

        # Top evidence for winning class (INVOICE)
        top_ev = result.get_top_evidence(limit=2)
        self.assertEqual(len(top_ev), 2)
        self.assertEqual(top_ev[0].signal_name, "invoice:tax invoice")
        self.assertEqual(top_ev[1].signal_name, "invoice:subtotal")

        # Page evidence
        page1_ev = result.get_page_evidence(page_number=1)
        self.assertEqual(len(page1_ev), 2)
        page2_ev = result.get_page_evidence(page_number=2)
        self.assertEqual(len(page2_ev), 1)

        # Serialization
        res_dict = result.to_dict()
        self.assertEqual(res_dict["document_type"], "invoice")
        self.assertEqual(res_dict["confidence"], 0.82)
        self.assertEqual(len(res_dict["evidence"]), 3)

        reconstructed = ClassificationResult.from_dict(res_dict)
        self.assertEqual(reconstructed.document_type, DocumentType.INVOICE)
        self.assertEqual(reconstructed.confidence, 0.82)
        self.assertEqual(len(reconstructed.evidence), 3)


if __name__ == "__main__":
    unittest.main()
