"""Typed domain models for structured field, entity, and candidate extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.models import BoundingBox, ExtractedField, Provenance
from src.core.types import (
    ConfidenceSource,
    DocumentType,
    ExtractionMethod,
    FieldType,
    ValidationStatus,
)


class EntityType(str, Enum):
    """Semantic category for generic extracted entities."""
    DATE = "date"
    MONEY = "money"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    IDENTIFIER = "identifier"
    ORGANIZATION = "organization"
    ADDRESS = "address"
    PERSON = "person"
    SKILL = "skill"
    CUSTOM = "custom"


@dataclass
class ExtractionCandidate:
    """A competing extracted candidate value for an entity or field."""
    value: Any
    raw_value: str
    normalized_value: Optional[Any] = None
    confidence: float = 0.0
    source_text: str = ""
    page_number: int = 1
    bounding_box: Optional[BoundingBox] = None
    ocr_confidence: Optional[float] = None
    extraction_method: ExtractionMethod = ExtractionMethod.KEY_VALUE_HEURISTIC
    matched_label: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"Page number {self.page_number} must be >= 1")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Candidate confidence {self.confidence} must be in [0.0, 1.0]")
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"OCR confidence {self.ocr_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize extraction candidate to dictionary."""
        return {
            "value": self.value,
            "raw_value": self.raw_value,
            "normalized_value": (
                self.normalized_value.isoformat()
                if isinstance(self.normalized_value, datetime)
                else self.normalized_value
            ),
            "confidence": round(self.confidence, 4),
            "source_text": self.source_text,
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "extraction_method": self.extraction_method.value,
            "matched_label": self.matched_label,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionCandidate:
        """Construct ExtractionCandidate from dictionary."""
        bbox_data = data.get("bounding_box")
        return cls(
            value=data["value"],
            raw_value=data.get("raw_value", str(data["value"])),
            normalized_value=data.get("normalized_value"),
            confidence=float(data.get("confidence", 0.0)),
            source_text=data.get("source_text", ""),
            page_number=int(data.get("page_number", 1)),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            extraction_method=ExtractionMethod(data.get("extraction_method", ExtractionMethod.KEY_VALUE_HEURISTIC.value)),
            matched_label=data.get("matched_label"),
            details=data.get("details", {}),
        )


@dataclass
class ExtractedEntity:
    """A generic named or typed entity extracted from document text."""
    entity_type: EntityType
    value: str
    normalized_value: Optional[Any] = None
    confidence: float = 0.0
    page_number: int = 1
    bounding_box: Optional[BoundingBox] = None
    ocr_confidence: Optional[float] = None
    source_text: str = ""
    character_span: Optional[Tuple[int, int]] = None
    extraction_method: ExtractionMethod = ExtractionMethod.REGEX_PATTERN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(f"Page number {self.page_number} must be >= 1")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Entity confidence {self.confidence} must be in [0.0, 1.0]")
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"OCR confidence {self.ocr_confidence} must be in [0.0, 1.0]")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        return {
            "entity_type": self.entity_type.value,
            "value": self.value,
            "normalized_value": (
                self.normalized_value.isoformat()
                if isinstance(self.normalized_value, datetime)
                else self.normalized_value
            ),
            "confidence": round(self.confidence, 4),
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "source_text": self.source_text,
            "character_span": list(self.character_span) if self.character_span else None,
            "extraction_method": self.extraction_method.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedEntity:
        """Construct ExtractedEntity from dictionary."""
        bbox_data = data.get("bounding_box")
        span_data = data.get("character_span")
        return cls(
            entity_type=EntityType(data["entity_type"]),
            value=data["value"],
            normalized_value=data.get("normalized_value"),
            confidence=float(data.get("confidence", 0.0)),
            page_number=int(data.get("page_number", 1)),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            source_text=data.get("source_text", ""),
            character_span=tuple(span_data) if span_data else None,
            extraction_method=ExtractionMethod(data.get("extraction_method", ExtractionMethod.REGEX_PATTERN.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionResult:
    """Unified payload of all structured extracted fields, generic entities, and metadata."""
    document_id: str
    document_type: DocumentType
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    entities: List[ExtractedEntity] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_statistics: Dict[str, Any] = field(default_factory=dict)
    extractor_name: str = "rule_based_extractor"
    extractor_version: str = "1.0.0"
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str) -> Optional[ExtractedField]:
        """Retrieve extracted field by canonical name."""
        return self.fields.get(name)

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List[ExtractedEntity]:
        """Retrieve all extracted generic entities matching a given type."""
        target_type = entity_type.value if isinstance(entity_type, EntityType) else str(entity_type).lower()
        return [e for e in self.entities if e.entity_type.value == target_type]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete extraction result to dictionary."""
        doc_type_val = self.document_type.value if hasattr(self.document_type, "value") else str(self.document_type)
        return {
            "document_id": self.document_id,
            "document_type": doc_type_val,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "entities": [e.to_dict() for e in self.entities],
            "unresolved_fields": self.unresolved_fields,
            "warnings": self.warnings,
            "extraction_statistics": self.extraction_statistics,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize complete extraction result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractionResult:
        """Construct ExtractionResult from dictionary."""
        fields_data = data.get("fields", {})
        entities_data = data.get("entities", [])
        raw_doc_type = data.get("document_type", DocumentType.UNKNOWN.value)
        try:
            doc_type = DocumentType(raw_doc_type)
        except (ValueError, TypeError):
            doc_type = DocumentType.UNKNOWN

        return cls(
            document_id=data["document_id"],
            document_type=doc_type,
            fields={k: ExtractedField.from_dict(v) for k, v in fields_data.items()},
            entities=[ExtractedEntity.from_dict(e) for e in entities_data],
            unresolved_fields=data.get("unresolved_fields", []),
            warnings=data.get("warnings", []),
            extraction_statistics=data.get("extraction_statistics", {}),
            extractor_name=data.get("extractor_name", "rule_based_extractor"),
            extractor_version=data.get("extractor_version", "1.0.0"),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "EntityType",
    "ExtractionCandidate",
    "ExtractedEntity",
    "ExtractedField",
    "ExtractionResult",
]
