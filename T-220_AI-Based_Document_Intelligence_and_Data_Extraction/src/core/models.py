"""Typed domain models and data structures for DocuMind AI.

All models provide clean data encapsulation, provenance preservation,
and bidirectional serialization to standard Python dictionaries / JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.types import (
    ConfidenceSource,
    DocumentType,
    ExtractionMethod,
    FieldType,
    ReviewTriggerType,
    SeverityLevel,
    ValidationStatus,
)


@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding region in a document page.

    Coordinates can be absolute (pixel-based) or normalized ([0.0, 1.0]).
    Origin (0, 0) is standard top-left.
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    is_normalized: bool = False

    def __post_init__(self) -> None:
        if self.xmin > self.xmax:
            raise ValueError(f"Invalid BoundingBox: xmin ({self.xmin}) > xmax ({self.xmax})")
        if self.ymin > self.ymax:
            raise ValueError(f"Invalid BoundingBox: ymin ({self.ymin}) > ymax ({self.ymax})")
        if self.is_normalized:
            for val, name in [(self.xmin, "xmin"), (self.ymin, "ymin"), (self.xmax, "xmax"), (self.ymax, "ymax")]:
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"Normalized BoundingBox coordinate {name}={val} outside [0.0, 1.0]")

    @property
    def width(self) -> float:
        """Width of the bounding region."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Height of the bounding region."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Area of the bounding region."""
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bounding box to dictionary."""
        return {
            "xmin": round(self.xmin, 4),
            "ymin": round(self.ymin, 4),
            "xmax": round(self.xmax, 4),
            "ymax": round(self.ymax, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "is_normalized": self.is_normalized,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoundingBox:
        """Construct BoundingBox from dictionary."""
        return cls(
            xmin=float(data["xmin"]),
            ymin=float(data["ymin"]),
            xmax=float(data["xmax"]),
            ymax=float(data["ymax"]),
            is_normalized=bool(data.get("is_normalized", False)),
        )

    def normalize(self, page_width: float, page_height: float) -> BoundingBox:
        """Return a normalized coordinate representation [0.0, 1.0]."""
        if self.is_normalized:
            return self
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Page width and height must be positive to normalize coordinates.")
        return BoundingBox(
            xmin=max(0.0, min(1.0, self.xmin / page_width)),
            ymin=max(0.0, min(1.0, self.ymin / page_height)),
            xmax=max(0.0, min(1.0, self.xmax / page_width)),
            ymax=max(0.0, min(1.0, self.ymax / page_height)),
            is_normalized=True,
        )

    def denormalize(self, page_width: float, page_height: float) -> BoundingBox:
        """Return pixel coordinates based on given page dimensions."""
        if not self.is_normalized:
            return self
        return BoundingBox(
            xmin=self.xmin * page_width,
            ymin=self.ymin * page_height,
            xmax=self.xmax * page_width,
            ymax=self.ymax * page_height,
            is_normalized=False,
        )

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Compute IoU overlap metric with another bounding box."""
        if self.is_normalized != other.is_normalized:
            raise ValueError("Cannot calculate IoU between normalized and unnormalized bounding boxes.")

        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)

        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        intersection = iw * ih

        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union


@dataclass
class Provenance:
    """Complete provenance tracking linking extracted data to its physical source."""
    document_id: str
    page_number: int
    raw_text: str
    bounding_box: Optional[BoundingBox] = None
    ocr_confidence: Optional[float] = None
    extraction_method: ExtractionMethod = ExtractionMethod.REGEX_PATTERN
    character_span: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.ocr_confidence is not None and not (0.0 <= self.ocr_confidence <= 1.0):
            raise ValueError(f"OCR confidence {self.ocr_confidence} must be in range [0.0, 1.0]")
        if self.page_number < 1:
            raise ValueError(f"Page number {self.page_number} must be >= 1 (1-indexed)")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provenance to dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else None,
            "extraction_method": self.extraction_method.value,
            "character_span": list(self.character_span) if self.character_span else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Provenance:
        """Construct Provenance from dictionary."""
        bbox_data = data.get("bounding_box")
        span_data = data.get("character_span")
        return cls(
            document_id=data["document_id"],
            page_number=int(data["page_number"]),
            raw_text=data.get("raw_text", ""),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
            extraction_method=ExtractionMethod(data.get("extraction_method", ExtractionMethod.REGEX_PATTERN.value)),
            character_span=tuple(span_data) if span_data else None,
        )


@dataclass
class OCRTextRegion:
    """A discrete unit of text detected during optical character recognition."""
    text: str
    page_number: int
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None
    level: str = "word"  # word, line, paragraph, block
    sub_regions: List[OCRTextRegion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize OCR text region to dictionary."""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "level": self.level,
            "sub_regions": [sr.to_dict() for sr in self.sub_regions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OCRTextRegion:
        """Construct OCRTextRegion from dictionary."""
        bbox_data = data.get("bounding_box")
        return cls(
            text=data.get("text", ""),
            page_number=int(data.get("page_number", 1)),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            level=data.get("level", "word"),
            sub_regions=[cls.from_dict(sr) for sr in data.get("sub_regions", [])],
        )


@dataclass
class DocumentMetadata:
    """High-level metadata regarding an ingested document."""
    document_id: str
    filename: str
    file_path: Union[str, Path]
    file_type: str
    file_size_bytes: int
    checksum_sha256: str
    page_count: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_scanned: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_path": str(self.file_path),
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "page_count": self.page_count,
            "created_at": self.created_at.isoformat(),
            "is_scanned": self.is_scanned,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentMetadata:
        """Construct DocumentMetadata from dictionary."""
        created_str = data.get("created_at")
        created_dt = datetime.fromisoformat(created_str) if created_str else datetime.now(timezone.utc)
        return cls(
            document_id=data["document_id"],
            filename=data["filename"],
            file_path=Path(data["file_path"]),
            file_type=data["file_type"],
            file_size_bytes=int(data["file_size_bytes"]),
            checksum_sha256=data["checksum_sha256"],
            page_count=int(data["page_count"]),
            created_at=created_dt,
            is_scanned=data.get("is_scanned"),
        )


@dataclass
class DocumentPage:
    """In-memory representation of a single page within a document."""
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    dpi: Optional[int] = None
    image_path: Optional[Union[str, Path]] = None
    ocr_text_regions: List[OCRTextRegion] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DocumentPage to dictionary."""
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "image_path": str(self.image_path) if self.image_path else None,
            "ocr_text_regions": [tr.to_dict() for tr in self.ocr_text_regions],
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentPage:
        """Construct DocumentPage from dictionary."""
        return cls(
            page_number=int(data["page_number"]),
            width=float(data["width"]) if data.get("width") is not None else None,
            height=float(data["height"]) if data.get("height") is not None else None,
            dpi=int(data["dpi"]) if data.get("dpi") is not None else None,
            image_path=Path(data["image_path"]) if data.get("image_path") else None,
            ocr_text_regions=[OCRTextRegion.from_dict(tr) for tr in data.get("ocr_text_regions", [])],
            raw_text=data.get("raw_text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document:
    """Primary document aggregate containing metadata, pages, and classification."""
    metadata: DocumentMetadata
    pages: List[DocumentPage] = field(default_factory=list)
    classified_type: DocumentType = DocumentType.UNKNOWN
    classification_confidence: Optional[float] = None
    classification_method: str = "unclassified"

    @property
    def id(self) -> str:
        """Convenience property for document ID."""
        return self.metadata.document_id

    @id.setter
    def id(self, value: str) -> None:
        """Setter for convenience property."""
        self.metadata.document_id = value

    @property
    def full_text(self) -> str:
        """Concatenated text across all pages."""
        return "\n\n--- Page Break ---\n\n".join(
            p.raw_text for p in sorted(self.pages, key=lambda x: x.page_number)
        )

    def get_page(self, page_number: int) -> Optional[DocumentPage]:
        """Retrieve a specific page by 1-indexed page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Document to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "pages": [p.to_dict() for p in self.pages],
            "classified_type": self.classified_type.value,
            "classification_confidence": (
                round(self.classification_confidence, 4)
                if self.classification_confidence is not None
                else None
            ),
            "classification_method": self.classification_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        """Construct Document from dictionary."""
        return cls(
            metadata=DocumentMetadata.from_dict(data["metadata"]),
            pages=[DocumentPage.from_dict(p) for p in data.get("pages", [])],
            classified_type=DocumentType(data.get("classified_type", DocumentType.UNKNOWN.value)),
            classification_confidence=(
                float(data["classification_confidence"])
                if data.get("classification_confidence") is not None
                else None
            ),
            classification_method=data.get("classification_method", "unclassified"),
        )


@dataclass
class ExtractedField:
    """An individual field/entity extracted from a document with full provenance."""
    name: str
    value: Any
    normalized_value: Optional[Any] = None
    field_type: FieldType = FieldType.STRING
    provenance: Optional[Provenance] = None
    extraction_confidence: float = 0.0
    confidence_source: ConfidenceSource = ConfidenceSource.HEURISTIC
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    validation_messages: List[str] = field(default_factory=list)
    is_required: bool = False
    is_ambiguous: bool = False
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    matched_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.extraction_confidence <= 1.0):
            raise ValueError(
                f"Extraction confidence {self.extraction_confidence} for field '{self.name}' must be in [0.0, 1.0]"
            )

    @property
    def page_number(self) -> int:
        """1-indexed source page number from provenance."""
        return self.provenance.page_number if self.provenance else 1

    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """Physical or normalized bounding box from provenance."""
        return self.provenance.bounding_box if self.provenance else None

    @property
    def source_text(self) -> str:
        """Raw source text string from provenance."""
        return self.provenance.raw_text if self.provenance else ""

    @property
    def source_ocr_confidence(self) -> Optional[float]:
        """Native optical recognition confidence from provenance."""
        return self.provenance.ocr_confidence if self.provenance else None

    @property
    def extraction_method(self) -> ExtractionMethod:
        """Technical mechanism used during extraction."""
        return self.provenance.extraction_method if self.provenance else ExtractionMethod.KEY_VALUE_HEURISTIC

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ExtractedField to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "normalized_value": (
                self.normalized_value.isoformat()
                if isinstance(self.normalized_value, datetime)
                else self.normalized_value
            ),
            "field_type": self.field_type.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "extraction_confidence": round(self.extraction_confidence, 4),
            "confidence_source": self.confidence_source.value,
            "validation_status": self.validation_status.value,
            "validation_messages": self.validation_messages,
            "is_required": self.is_required,
            "is_ambiguous": self.is_ambiguous,
            "candidates": self.candidates,
            "matched_label": self.matched_label,
            "metadata": self.metadata,
            "source_text": self.source_text,
            "source_ocr_confidence": round(self.source_ocr_confidence, 4) if self.source_ocr_confidence is not None else None,
            "page_number": self.page_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedField:
        """Construct ExtractedField from dictionary."""
        prov_data = data.get("provenance")
        return cls(
            name=data["name"],
            value=data.get("value"),
            normalized_value=data.get("normalized_value"),
            field_type=FieldType(data.get("field_type", FieldType.STRING.value)),
            provenance=Provenance.from_dict(prov_data) if prov_data else None,
            extraction_confidence=float(data.get("extraction_confidence", 0.0)),
            confidence_source=ConfidenceSource(data.get("confidence_source", ConfidenceSource.HEURISTIC.value)),
            validation_status=ValidationStatus(data.get("validation_status", ValidationStatus.UNVALIDATED.value)),
            validation_messages=data.get("validation_messages", []),
            is_required=bool(data.get("is_required", False)),
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            candidates=data.get("candidates", []),
            matched_label=data.get("matched_label"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TableCell:
    """Individual cell representation in an extracted table."""
    row_index: int
    col_index: int
    text: str
    normalized_value: Optional[Any] = None
    bounding_box: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    page_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableCell to dictionary."""
        return {
            "row_index": self.row_index,
            "col_index": self.col_index,
            "text": self.text,
            "normalized_value": self.normalized_value,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "page_number": self.page_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableCell:
        """Construct TableCell from dictionary."""
        bbox_data = data.get("bounding_box")
        return cls(
            row_index=int(data["row_index"]),
            col_index=int(data["col_index"]),
            text=data.get("text", ""),
            normalized_value=data.get("normalized_value"),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            page_number=int(data.get("page_number", 1)),
        )


@dataclass
class ExtractedTable:
    """Structured table extracted from document page."""
    table_id: str
    page_number: int
    headers: List[str]
    rows: List[List[TableCell]]
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    confidence_source: ConfidenceSource = ConfidenceSource.HEURISTIC
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED

    def to_records(self) -> List[Dict[str, Any]]:
        """Convert rows to a list of header-mapped dictionary records for CSV/dataframes."""
        records: List[Dict[str, Any]] = []
        for row in self.rows:
            row_dict: Dict[str, Any] = {}
            for col_idx, header in enumerate(self.headers):
                cell_val = row[col_idx].normalized_value if col_idx < len(row) and row[col_idx].normalized_value is not None else (
                    row[col_idx].text if col_idx < len(row) else None
                )
                row_dict[header] = cell_val
            records.append(row_dict)
        return records

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ExtractedTable to dictionary."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "headers": self.headers,
            "rows": [[cell.to_dict() for cell in row] for row in self.rows],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": round(self.confidence, 4),
            "confidence_source": self.confidence_source.value,
            "validation_status": self.validation_status.value,
            "records": self.to_records(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExtractedTable:
        """Construct ExtractedTable from dictionary."""
        bbox_data = data.get("bounding_box")
        rows_data = data.get("rows", [])
        parsed_rows = [
            [TableCell.from_dict(cell) for cell in row]
            for row in rows_data
        ]
        return cls(
            table_id=data["table_id"],
            page_number=int(data["page_number"]),
            headers=data.get("headers", []),
            rows=parsed_rows,
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            confidence=float(data.get("confidence", 0.0)),
            confidence_source=ConfidenceSource(data.get("confidence_source", ConfidenceSource.HEURISTIC.value)),
            validation_status=ValidationStatus(data.get("validation_status", ValidationStatus.UNVALIDATED.value)),
        )


@dataclass
class ValidationResult:
    """Outcome of a single deterministic or rule-based validation check."""
    rule_id: str
    rule_name: str
    status: ValidationStatus
    severity: SeverityLevel
    message: str
    affected_fields: List[str] = field(default_factory=list)
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ValidationResult to dictionary."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "affected_fields": self.affected_fields,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationResult:
        """Construct ValidationResult from dictionary."""
        ts_str = data.get("timestamp")
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
        return cls(
            rule_id=data["rule_id"],
            rule_name=data["rule_name"],
            status=ValidationStatus(data["status"]),
            severity=SeverityLevel(data["severity"]),
            message=data["message"],
            affected_fields=data.get("affected_fields", []),
            expected_value=data.get("expected_value"),
            actual_value=data.get("actual_value"),
            timestamp=ts,
        )


@dataclass
class ReviewFlag:
    """Human review notification highlighting low-confidence or validation warning."""
    reason: str
    severity: SeverityLevel
    affected_field: Optional[str] = None
    message: str = ""
    trigger_type: ReviewTriggerType = ReviewTriggerType.VALIDATION_FAILURE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReviewFlag to dictionary."""
        return {
            "reason": self.reason,
            "severity": self.severity.value,
            "affected_field": self.affected_field,
            "message": self.message,
            "trigger_type": self.trigger_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewFlag:
        """Construct ReviewFlag from dictionary."""
        return cls(
            reason=data["reason"],
            severity=SeverityLevel(data["severity"]),
            affected_field=data.get("affected_field"),
            message=data.get("message", ""),
            trigger_type=ReviewTriggerType(data.get("trigger_type", ReviewTriggerType.VALIDATION_FAILURE.value)),
        )


@dataclass
class ProcessingSummary:
    """Condensed diagnostic summary of the entire document processing run."""
    document_id: str
    document_type: DocumentType
    total_pages: int
    total_fields_extracted: int
    total_tables_extracted: int
    validation_passed_count: int
    validation_warning_count: int
    validation_error_count: int
    overall_confidence_score: float
    review_required: bool
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ProcessingSummary to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "total_pages": self.total_pages,
            "total_fields_extracted": self.total_fields_extracted,
            "total_tables_extracted": self.total_tables_extracted,
            "validation_passed_count": self.validation_passed_count,
            "validation_warning_count": self.validation_warning_count,
            "validation_error_count": self.validation_error_count,
            "overall_confidence_score": round(self.overall_confidence_score, 4),
            "review_required": self.review_required,
            "processing_time_seconds": round(self.processing_time_seconds, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessingSummary:
        """Construct ProcessingSummary from dictionary."""
        return cls(
            document_id=data["document_id"],
            document_type=DocumentType(data["document_type"]),
            total_pages=int(data["total_pages"]),
            total_fields_extracted=int(data["total_fields_extracted"]),
            total_tables_extracted=int(data["total_tables_extracted"]),
            validation_passed_count=int(data["validation_passed_count"]),
            validation_warning_count=int(data["validation_warning_count"]),
            validation_error_count=int(data["validation_error_count"]),
            overall_confidence_score=float(data["overall_confidence_score"]),
            review_required=bool(data["review_required"]),
            processing_time_seconds=float(data.get("processing_time_seconds", 0.0)),
        )


@dataclass
class ProcessingResult:
    """Complete, end-to-end outcome of executing the document intelligence pipeline."""
    document: Document
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    tables: List[ExtractedTable] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    review_flags: List[ReviewFlag] = field(default_factory=list)
    summary: Optional[ProcessingSummary] = None
    confidence: Optional[Any] = None
    review: Optional[Any] = None
    visualization: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete processing result to structured JSON dictionary."""
        conf_dict = None
        if self.confidence is not None:
            conf_dict = self.confidence.to_dict() if hasattr(self.confidence, "to_dict") else self.confidence

        rev_dict = None
        if self.review is not None:
            rev_dict = self.review.to_dict() if hasattr(self.review, "to_dict") else self.review

        vis_dict = None
        if self.visualization is not None:
            vis_dict = self.visualization.to_dict() if hasattr(self.visualization, "to_dict") else self.visualization

        return {
            "document": self.document.to_dict(),
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "tables": [t.to_dict() for t in self.tables],
            "validation_results": [v.to_dict() for v in self.validation_results],
            "review_flags": [rf.to_dict() for rf in self.review_flags],
            "summary": self.summary.to_dict() if hasattr(self.summary, "to_dict") else self.summary,
            "confidence": conf_dict,
            "review": rev_dict,
            "visualization": vis_dict,
            "errors": self.errors,
            "execution_metadata": self.execution_metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize complete processing result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessingResult:
        """Construct ProcessingResult from dictionary."""
        summary_data = data.get("summary")
        return cls(
            document=Document.from_dict(data["document"]),
            fields={k: ExtractedField.from_dict(v) for k, v in data.get("fields", {}).items()},
            tables=[ExtractedTable.from_dict(t) for t in data.get("tables", [])],
            validation_results=[ValidationResult.from_dict(v) for v in data.get("validation_results", [])],
            review_flags=[ReviewFlag.from_dict(rf) for rf in data.get("review_flags", [])],
            summary=ProcessingSummary.from_dict(summary_data) if summary_data else None,
            confidence=data.get("confidence"),
            review=data.get("review"),
            visualization=data.get("visualization"),
            errors=data.get("errors", []),
            execution_metadata=data.get("execution_metadata", {}),
        )
