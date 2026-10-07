"""Domain models for document classification and explainable evidence tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.models import BoundingBox
from src.core.types import DocumentType


class SignalType(str, Enum):
    """Categorization of evidence signal used during classification."""
    KEYWORD = "keyword"
    PHRASE = "phrase"
    REGEX = "regex"
    HEADER = "header"
    LAYOUT = "layout"
    NEGATIVE_SIGNAL = "negative_signal"
    CUSTOM = "custom"


@dataclass
class ClassificationEvidence:
    """Individual piece of matched evidence supporting or penalizing a classification decision."""
    signal_name: str
    signal_type: SignalType
    matched_text: str
    weight: float
    target_type: DocumentType
    page_number: int
    bounding_box: Optional[BoundingBox] = None
    ocr_confidence: Optional[float] = None
    character_span: Optional[Tuple[int, int]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"Page number {self.page_number} must be >= 1 (1-indexed)")
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"OCR confidence {self.ocr_confidence} must be in range [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence to dictionary."""
        return {
            "signal_name": self.signal_name,
            "signal_type": self.signal_type.value,
            "matched_text": self.matched_text,
            "weight": round(self.weight, 4),
            "target_type": self.target_type.value,
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "character_span": list(self.character_span) if self.character_span else None,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassificationEvidence:
        """Construct ClassificationEvidence from dictionary."""
        bbox_data = data.get("bounding_box")
        span_data = data.get("character_span")
        return cls(
            signal_name=data["signal_name"],
            signal_type=SignalType(data.get("signal_type", SignalType.KEYWORD.value)),
            matched_text=data.get("matched_text", ""),
            weight=float(data.get("weight", 0.0)),
            target_type=DocumentType(data["target_type"]),
            page_number=int(data.get("page_number", 1)),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            character_span=tuple(span_data) if span_data else None,
            details=data.get("details", {}),
        )


@dataclass
class ClassificationCandidate:
    """Detailed score and evidence breakdown for a single candidate document type."""
    document_type: DocumentType
    raw_score: float
    normalized_score: float
    evidence_count: int
    evidence: List[ClassificationEvidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate breakdown to dictionary."""
        return {
            "document_type": self.document_type.value,
            "raw_score": round(self.raw_score, 4),
            "normalized_score": round(self.normalized_score, 4),
            "evidence_count": self.evidence_count,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassificationCandidate:
        """Construct ClassificationCandidate from dictionary."""
        return cls(
            document_type=DocumentType(data["document_type"]),
            raw_score=float(data.get("raw_score", 0.0)),
            normalized_score=float(data.get("normalized_score", 0.0)),
            evidence_count=int(data.get("evidence_count", 0)),
            evidence=[ClassificationEvidence.from_dict(e) for e in data.get("evidence", [])],
        )


@dataclass
class ClassificationResult:
    """Outcome of document type classification with complete explainability and provenance."""
    document_type: DocumentType
    confidence: float
    method: str = "rule_based"
    version: str = "1.0"
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    candidates: Dict[str, ClassificationCandidate] = field(default_factory=dict)
    evidence: List[ClassificationEvidence] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Classification confidence {self.confidence} must be in range [0.0, 1.0]")
        # Sync backward-compatible fields if empty
        if not self.matched_keywords and self.evidence:
            self.matched_keywords = list(dict.fromkeys(e.matched_text for e in self.evidence if e.matched_text))
        if not self.signals and self.candidate_scores:
            self.signals = {"candidate_scores": self.candidate_scores, "evidence_count": len(self.evidence)}

    def get_top_evidence(self, limit: int = 5) -> List[ClassificationEvidence]:
        """Return the highest-weight positive evidence items for the winning class."""
        winning_evidence = [e for e in self.evidence if e.target_type == self.document_type and e.weight > 0]
        return sorted(winning_evidence, key=lambda e: e.weight, reverse=True)[:limit]

    def get_page_evidence(self, page_number: int) -> List[ClassificationEvidence]:
        """Return all evidence items discovered on a specific page."""
        return [e for e in self.evidence if e.page_number == page_number]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ClassificationResult to dictionary."""
        return {
            "document_type": self.document_type.value,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "version": self.version,
            "candidate_scores": {k: round(v, 4) for k, v in self.candidate_scores.items()},
            "candidates": {k: v.to_dict() for k, v in self.candidates.items()},
            "evidence": [e.to_dict() for e in self.evidence],
            "matched_keywords": self.matched_keywords,
            "signals": self.signals,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassificationResult:
        """Construct ClassificationResult from dictionary."""
        candidates_data = data.get("candidates", {})
        evidence_data = data.get("evidence", [])
        return cls(
            document_type=DocumentType(data["document_type"]),
            confidence=float(data.get("confidence", 0.0)),
            method=data.get("method", "rule_based"),
            version=data.get("version", "1.0"),
            candidate_scores={k: float(v) for k, v in data.get("candidate_scores", {}).items()},
            candidates={k: ClassificationCandidate.from_dict(v) for k, v in candidates_data.items()},
            evidence=[ClassificationEvidence.from_dict(e) for e in evidence_data],
            matched_keywords=data.get("matched_keywords", []),
            signals=data.get("signals", {}),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            ambiguity_reason=data.get("ambiguity_reason"),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
            metadata=data.get("metadata", {}),
        )
