"""Typed domain models and data structures for table intelligence and line-item extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.models import (
    BoundingBox,
    ExtractedTable as CoreExtractedTable,
    Provenance,
    TableCell as CoreTableCell,
)
from src.core.types import (
    ConfidenceSource,
    ExtractionMethod,
    ValidationStatus,
)
from src.ocr.models import OCRWord


class TableValueType(str, Enum):
    """Inferred semantic or physical data type for table cells and columns."""
    TEXT = "text"
    INTEGER = "integer"
    DECIMAL = "decimal"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    EMPTY = "empty"


@dataclass
class TableCell:
    """Individual cell representation in a reconstructed 2D table grid."""
    row_index: int
    col_index: int
    text: str = ""
    normalized_value: Optional[Any] = None
    bounding_box: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    page_number: int = 1
    value_type: TableValueType = TableValueType.TEXT
    is_header: bool = False
    words: List[OCRWord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"TableCell confidence {self.confidence} must be in [0.0, 1.0]")
        if self.page_number < 1:
            raise ValueError(f"TableCell page_number {self.page_number} must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableCell to dictionary."""
        return {
            "row_index": self.row_index,
            "col_index": self.col_index,
            "text": self.text,
            "normalized_value": (
                self.normalized_value.isoformat()
                if isinstance(self.normalized_value, datetime)
                else self.normalized_value
            ),
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "page_number": self.page_number,
            "value_type": self.value_type.value,
            "is_header": self.is_header,
            "word_count": len(self.words),
            "metadata": self.metadata,
        }

    def to_core_cell(self) -> CoreTableCell:
        """Convert to core domain TableCell model."""
        return CoreTableCell(
            row_index=self.row_index,
            col_index=self.col_index,
            text=self.text,
            normalized_value=self.normalized_value,
            bounding_box=self.bounding_box,
            confidence=self.confidence,
            page_number=self.page_number,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableCell:
        """Construct TableCell from dictionary."""
        bbox_data = data.get("bounding_box")
        val_type_raw = data.get("value_type", TableValueType.TEXT.value)
        try:
            val_type = TableValueType(val_type_raw)
        except ValueError:
            val_type = TableValueType.TEXT

        return cls(
            row_index=int(data["row_index"]),
            col_index=int(data["col_index"]),
            text=data.get("text", ""),
            normalized_value=data.get("normalized_value"),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            page_number=int(data.get("page_number", 1)),
            value_type=val_type,
            is_header=bool(data.get("is_header", False)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TableColumn:
    """Column definition representing vertical layout, header identity, and type bounds."""
    index: int
    name: str
    canonical_field: Optional[str] = None
    x_start: float = 0.0
    x_end: float = 0.0
    alignment: str = "left"  # left, right, center
    inferred_type: TableValueType = TableValueType.TEXT
    confidence: float = 1.0

    @property
    def width(self) -> float:
        """Width span of the column."""
        return max(0.0, self.x_end - self.x_start)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableColumn to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "canonical_field": self.canonical_field,
            "x_start": round(self.x_start, 2),
            "x_end": round(self.x_end, 2),
            "width": round(self.width, 2),
            "alignment": self.alignment,
            "inferred_type": self.inferred_type.value,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableColumn:
        """Construct TableColumn from dictionary."""
        val_type_raw = data.get("inferred_type", TableValueType.TEXT.value)
        try:
            val_type = TableValueType(val_type_raw)
        except ValueError:
            val_type = TableValueType.TEXT

        return cls(
            index=int(data["index"]),
            name=data.get("name", f"col_{data.get('index', 0)}"),
            canonical_field=data.get("canonical_field"),
            x_start=float(data.get("x_start", 0.0)),
            x_end=float(data.get("x_end", 0.0)),
            alignment=data.get("alignment", "left"),
            inferred_type=val_type,
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class TableHeader:
    """Detected header row containing column labels and spatial bounds."""
    row_index: int
    cells: List[TableCell] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    page_number: int = 1
    column_names: List[str] = field(default_factory=list)
    is_continuation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableHeader to dictionary."""
        return {
            "row_index": self.row_index,
            "column_names": self.column_names,
            "cells": [c.to_dict() for c in self.cells],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": round(self.confidence, 4),
            "page_number": self.page_number,
            "is_continuation": self.is_continuation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableHeader:
        """Construct TableHeader from dictionary."""
        bbox_data = data.get("bounding_box")
        return cls(
            row_index=int(data["row_index"]),
            cells=[TableCell.from_dict(c) for c in data.get("cells", [])],
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            confidence=float(data.get("confidence", 0.0)),
            page_number=int(data.get("page_number", 1)),
            column_names=data.get("column_names", []),
            is_continuation=bool(data.get("is_continuation", False)),
        )


@dataclass
class TableRow:
    """Horizontal row of cells in a reconstructed table grid."""
    row_index: int
    cells: List[TableCell] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    is_header: bool = False
    is_summary: bool = False
    page_number: int = 1
    confidence: float = 0.0

    def get_cell(self, col_index: int) -> Optional[TableCell]:
        """Retrieve cell by 0-indexed column position."""
        if 0 <= col_index < len(self.cells):
            return self.cells[col_index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableRow to dictionary."""
        return {
            "row_index": self.row_index,
            "cells": [c.to_dict() for c in self.cells],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "is_header": self.is_header,
            "is_summary": self.is_summary,
            "page_number": self.page_number,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableRow:
        """Construct TableRow from dictionary."""
        bbox_data = data.get("bounding_box")
        return cls(
            row_index=int(data["row_index"]),
            cells=[TableCell.from_dict(c) for c in data.get("cells", [])],
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            is_header=bool(data.get("is_header", False)),
            is_summary=bool(data.get("is_summary", False)),
            page_number=int(data.get("page_number", 1)),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class TableRegion:
    """Candidate 2D spatial region on a page identified as containing tabular content."""
    page_number: int
    bounding_box: BoundingBox
    header: Optional[TableHeader] = None
    row_boxes: List[BoundingBox] = field(default_factory=list)
    confidence_score: float = 0.0
    detection_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableRegion to dictionary."""
        return {
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict(),
            "header": self.header.to_dict() if self.header else None,
            "row_count": len(self.row_boxes),
            "row_boxes": [rb.to_dict() for rb in self.row_boxes],
            "confidence_score": round(self.confidence_score, 4),
            "detection_signals": self.detection_signals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableRegion:
        """Construct TableRegion from dictionary."""
        header_data = data.get("header")
        return cls(
            page_number=int(data["page_number"]),
            bounding_box=BoundingBox.from_dict(data["bounding_box"]),
            header=TableHeader.from_dict(header_data) if header_data else None,
            row_boxes=[BoundingBox.from_dict(rb) for rb in data.get("row_boxes", [])],
            confidence_score=float(data.get("confidence_score", 0.0)),
            detection_signals=data.get("detection_signals", {}),
        )


@dataclass
class LineItem:
    """Canonical domain representation of an itemized product, service, or charge row."""
    description: str = ""
    item_code: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    tax_rate: Optional[float] = None
    discount: Optional[float] = None
    unit_of_measure: Optional[str] = None
    row_index: int = 0
    page_number: int = 1
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    provenance: Optional[Provenance] = None
    is_valid_arithmetic: Optional[bool] = None
    validation_messages: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"LineItem confidence {self.confidence} must be in [0.0, 1.0]")
        if self.page_number < 1:
            raise ValueError(f"LineItem page_number {self.page_number} must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LineItem to dictionary."""
        return {
            "item_code": self.item_code,
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "amount": self.amount,
            "tax": self.tax,
            "tax_rate": self.tax_rate,
            "discount": self.discount,
            "unit_of_measure": self.unit_of_measure,
            "row_index": self.row_index,
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": round(self.confidence, 4),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "is_valid_arithmetic": self.is_valid_arithmetic,
            "validation_messages": self.validation_messages,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineItem:
        """Construct LineItem from dictionary."""
        bbox_data = data.get("bounding_box")
        prov_data = data.get("provenance")
        return cls(
            description=data.get("description", ""),
            item_code=data.get("item_code"),
            quantity=float(data["quantity"]) if data.get("quantity") is not None else None,
            unit_price=float(data["unit_price"]) if data.get("unit_price") is not None else None,
            amount=float(data["amount"]) if data.get("amount") is not None else None,
            tax=float(data["tax"]) if data.get("tax") is not None else None,
            tax_rate=float(data["tax_rate"]) if data.get("tax_rate") is not None else None,
            discount=float(data["discount"]) if data.get("discount") is not None else None,
            unit_of_measure=data.get("unit_of_measure"),
            row_index=int(data.get("row_index", 0)),
            page_number=int(data.get("page_number", 1)),
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            confidence=float(data.get("confidence", 0.0)),
            provenance=Provenance.from_dict(prov_data) if prov_data else None,
            is_valid_arithmetic=data.get("is_valid_arithmetic"),
            validation_messages=data.get("validation_messages", []),
            raw_data=data.get("raw_data", {}),
        )


@dataclass
class TableValidationResult:
    """Outcome of mathematical and structural table integrity validation."""
    is_valid: bool = True
    row_checks_passed: int = 0
    row_checks_failed: int = 0
    subtotal_valid: Optional[bool] = None
    calculated_subtotal: Optional[float] = None
    reported_subtotal: Optional[float] = None
    tax_valid: Optional[bool] = None
    calculated_tax: Optional[float] = None
    reported_tax: Optional[float] = None
    grand_total_valid: Optional[bool] = None
    calculated_total: Optional[float] = None
    reported_total: Optional[float] = None
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableValidationResult to dictionary."""
        return {
            "is_valid": self.is_valid,
            "row_checks_passed": self.row_checks_passed,
            "row_checks_failed": self.row_checks_failed,
            "subtotal_valid": self.subtotal_valid,
            "calculated_subtotal": self.calculated_subtotal,
            "reported_subtotal": self.reported_subtotal,
            "tax_valid": self.tax_valid,
            "calculated_tax": self.calculated_tax,
            "reported_tax": self.reported_tax,
            "grand_total_valid": self.grand_total_valid,
            "calculated_total": self.calculated_total,
            "reported_total": self.reported_total,
            "discrepancies": self.discrepancies,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableValidationResult:
        """Construct TableValidationResult from dictionary."""
        return cls(
            is_valid=bool(data.get("is_valid", True)),
            row_checks_passed=int(data.get("row_checks_passed", 0)),
            row_checks_failed=int(data.get("row_checks_failed", 0)),
            subtotal_valid=data.get("subtotal_valid"),
            calculated_subtotal=float(data["calculated_subtotal"]) if data.get("calculated_subtotal") is not None else None,
            reported_subtotal=float(data["reported_subtotal"]) if data.get("reported_subtotal") is not None else None,
            tax_valid=data.get("tax_valid"),
            calculated_tax=float(data["calculated_tax"]) if data.get("calculated_tax") is not None else None,
            reported_tax=float(data["reported_tax"]) if data.get("reported_tax") is not None else None,
            grand_total_valid=data.get("grand_total_valid"),
            calculated_total=float(data["calculated_total"]) if data.get("calculated_total") is not None else None,
            reported_total=float(data["reported_total"]) if data.get("reported_total") is not None else None,
            discrepancies=data.get("discrepancies", []),
            messages=data.get("messages", []),
        )


@dataclass
class Table:
    """Complete reconstructed 2D table aggregate with column metadata, rows, and line items."""
    table_id: str
    page_number: int
    headers: List[str] = field(default_factory=list)
    columns: List[TableColumn] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)
    line_items: List[LineItem] = field(default_factory=list)
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    confidence_source: ConfidenceSource = ConfidenceSource.HEURISTIC
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    validation_result: Optional[TableValidationResult] = None
    is_continuation: bool = False
    continuation_of_table_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Table confidence {self.confidence} must be in [0.0, 1.0]")
        if self.page_number < 1:
            raise ValueError(f"Table page_number {self.page_number} must be >= 1")

    @property
    def row_count(self) -> int:
        """Total number of data/summary rows in table."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Total number of columns."""
        return len(self.columns) if self.columns else len(self.headers)

    def to_records(self) -> List[Dict[str, Any]]:
        """Convert rows to a list of header-mapped dictionary records for CSV/dataframes."""
        records: List[Dict[str, Any]] = []
        for row in self.rows:
            if row.is_header:
                continue
            row_dict: Dict[str, Any] = {}
            for col_idx, header in enumerate(self.headers):
                cell = row.get_cell(col_idx)
                if cell:
                    row_dict[header] = cell.normalized_value if cell.normalized_value is not None else cell.text
                else:
                    row_dict[header] = None
            records.append(row_dict)
        return records

    def to_extracted_table(self) -> CoreExtractedTable:
        """Convert to Core ExtractedTable domain representation."""
        core_rows: List[List[CoreTableCell]] = [
            [c.to_core_cell() for c in row.cells]
            for row in self.rows
        ]
        return CoreExtractedTable(
            table_id=self.table_id,
            page_number=self.page_number,
            headers=list(self.headers),
            rows=core_rows,
            bounding_box=self.bounding_box,
            confidence=self.confidence,
            confidence_source=self.confidence_source,
            validation_status=self.validation_status,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Table to dictionary."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "headers": self.headers,
            "columns": [col.to_dict() for col in self.columns],
            "rows": [row.to_dict() for row in self.rows],
            "line_items": [li.to_dict() for li in self.line_items],
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "confidence": round(self.confidence, 4),
            "confidence_breakdown": {k: round(v, 4) for k, v in self.confidence_breakdown.items()},
            "confidence_source": self.confidence_source.value,
            "validation_status": self.validation_status.value,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "is_continuation": self.is_continuation,
            "continuation_of_table_id": self.continuation_of_table_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "records": self.to_records(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Table:
        """Construct Table from dictionary."""
        bbox_data = data.get("bounding_box")
        val_res_data = data.get("validation_result")
        return cls(
            table_id=data["table_id"],
            page_number=int(data.get("page_number", 1)),
            headers=data.get("headers", []),
            columns=[TableColumn.from_dict(c) for c in data.get("columns", [])],
            rows=[TableRow.from_dict(r) for r in data.get("rows", [])],
            line_items=[LineItem.from_dict(li) for li in data.get("line_items", [])],
            bounding_box=BoundingBox.from_dict(bbox_data) if bbox_data else None,
            confidence=float(data.get("confidence", 0.0)),
            confidence_breakdown=data.get("confidence_breakdown", {}),
            confidence_source=ConfidenceSource(data.get("confidence_source", ConfidenceSource.HEURISTIC.value)),
            validation_status=ValidationStatus(data.get("validation_status", ValidationStatus.UNVALIDATED.value)),
            validation_result=TableValidationResult.from_dict(val_res_data) if val_res_data else None,
            is_continuation=bool(data.get("is_continuation", False)),
            continuation_of_table_id=data.get("continuation_of_table_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TableExtractionResult:
    """Unified document-level table extraction outcome across all pages."""
    document_id: str
    tables: List[Table] = field(default_factory=list)
    line_items: List[LineItem] = field(default_factory=list)
    unresolved_tables: List[str] = field(default_factory=list)
    continuation_pairs: List[Tuple[str, str]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TableExtractionResult to dictionary."""
        return {
            "document_id": self.document_id,
            "tables": [t.to_dict() for t in self.tables],
            "line_items": [li.to_dict() for li in self.line_items],
            "unresolved_tables": self.unresolved_tables,
            "continuation_pairs": [list(pair) for pair in self.continuation_pairs],
            "statistics": self.statistics,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize TableExtractionResult to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TableExtractionResult:
        """Construct TableExtractionResult from dictionary."""
        return cls(
            document_id=data["document_id"],
            tables=[Table.from_dict(t) for t in data.get("tables", [])],
            line_items=[LineItem.from_dict(li) for li in data.get("line_items", [])],
            unresolved_tables=data.get("unresolved_tables", []),
            continuation_pairs=[tuple(p) for p in data.get("continuation_pairs", [])],
            statistics=data.get("statistics", {}),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
            warnings=data.get("warnings", []),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "TableValueType",
    "TableCell",
    "TableColumn",
    "TableHeader",
    "TableRow",
    "TableRegion",
    "LineItem",
    "TableValidationResult",
    "Table",
    "TableExtractionResult",
]
