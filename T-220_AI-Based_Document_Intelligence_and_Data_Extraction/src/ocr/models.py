"""Structured OCR data models preserving word, line, and block hierarchies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.models import BoundingBox, OCRTextRegion


class OCRLevel(str, Enum):
    """Hierarchical granularity level for detected OCR text components."""
    PAGE = "page"
    BLOCK = "block"
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"


@dataclass
class OCRWord:
    """Represents an individual recognized word with coordinates and recognition confidence."""
    text: str
    confidence: float  # Normalized [0.0, 1.0]
    raw_confidence: float  # Native engine score (e.g. 0.0 - 100.0 from Tesseract)
    bounding_box: BoundingBox
    page_number: int
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0
    word_num: int = 0
    reading_order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize word data to dictionary."""
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 2),
            "bounding_box": self.bounding_box.to_dict(),
            "page_number": self.page_number,
            "level": OCRLevel.WORD.value,
            "block_num": self.block_num,
            "par_num": self.par_num,
            "line_num": self.line_num,
            "word_num": self.word_num,
            "reading_order": self.reading_order,
        }

    def to_text_region(self) -> OCRTextRegion:
        """Convert to standard Core OCRTextRegion."""
        return OCRTextRegion(
            text=self.text,
            page_number=self.page_number,
            confidence=self.confidence,
            bounding_box=self.bounding_box,
            level=OCRLevel.WORD.value,
        )


@dataclass
class OCRLine:
    """Represents a single recognized horizontal text line comprising multiple words."""
    text: str
    words: List[OCRWord]
    bounding_box: BoundingBox
    confidence: float
    page_number: int
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize text line data to dictionary."""
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box.to_dict(),
            "page_number": self.page_number,
            "level": OCRLevel.LINE.value,
            "block_num": self.block_num,
            "par_num": self.par_num,
            "line_num": self.line_num,
            "words": [w.to_dict() for w in self.words],
        }

    def to_text_region(self) -> OCRTextRegion:
        """Convert to standard Core OCRTextRegion."""
        return OCRTextRegion(
            text=self.text,
            page_number=self.page_number,
            confidence=self.confidence,
            bounding_box=self.bounding_box,
            level=OCRLevel.LINE.value,
            sub_regions=[w.to_text_region() for w in self.words],
        )


@dataclass
class OCRBlock:
    """Represents a coherent spatial paragraph/block of text lines on a page."""
    text: str
    lines: List[OCRLine]
    bounding_box: BoundingBox
    confidence: float
    page_number: int
    block_num: int = 0
    block_type: str = "text"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize block data to dictionary."""
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box.to_dict(),
            "page_number": self.page_number,
            "level": OCRLevel.BLOCK.value,
            "block_num": self.block_num,
            "block_type": self.block_type,
            "lines": [l.to_dict() for l in self.lines],
        }


@dataclass
class PageOCRResult:
    """Standardized OCR recognition payload for a single page."""
    page_number: int
    raw_text: str
    blocks: List[OCRBlock] = field(default_factory=list)
    lines: List[OCRLine] = field(default_factory=list)
    words: List[OCRWord] = field(default_factory=list)
    text_regions: List[OCRTextRegion] = field(default_factory=list)
    engine_name: str = "unknown"
    engine_version: Optional[str] = None
    language: str = "eng"
    mean_confidence: Optional[float] = None
    execution_time_seconds: float = 0.0
    provenance_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PageOCRResult to dictionary."""
        return {
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "language": self.language,
            "mean_confidence": round(self.mean_confidence, 4) if self.mean_confidence is not None else None,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "word_count": len(self.words),
            "line_count": len(self.lines),
            "block_count": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
            "lines": [l.to_dict() for l in self.lines],
            "words": [w.to_dict() for w in self.words],
            "text_regions": [tr.to_dict() for tr in self.text_regions],
            "provenance_metadata": self.provenance_metadata,
        }


@dataclass
class DocumentOCRResult:
    """Aggregated OCR output spanning all pages of a processed document."""
    document_id: str
    pages: List[PageOCRResult] = field(default_factory=list)
    full_text: str = ""
    total_words: int = 0
    total_lines: int = 0
    total_blocks: int = 0
    mean_confidence: Optional[float] = None
    engine_name: str = "unknown"
    total_duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_page(self, page_number: int) -> Optional[PageOCRResult]:
        """Retrieve OCR results for a specific 1-indexed page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DocumentOCRResult to structured dictionary."""
        return {
            "document_id": self.document_id,
            "full_text": self.full_text,
            "total_pages": len(self.pages),
            "total_words": self.total_words,
            "total_lines": self.total_lines,
            "total_blocks": self.total_blocks,
            "mean_confidence": round(self.mean_confidence, 4) if self.mean_confidence is not None else None,
            "engine_name": self.engine_name,
            "total_duration_seconds": round(self.total_duration_seconds, 4),
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
        }

