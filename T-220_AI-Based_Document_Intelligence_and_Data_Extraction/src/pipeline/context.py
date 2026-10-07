"""Execution context flowing through pipeline stages."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.config import DocuMindConfig
from src.core.models import (
    Document,
    ExtractedField,
    ExtractedTable,
    ProcessingResult,
    ProcessingSummary,
    ReviewFlag,
    ValidationResult,
)


@dataclass
class PipelineExecutionContext:
    """State and artifacts accumulated as a document transitions through pipeline steps."""
    source_file_path: Union[str, Path]
    config: DocuMindConfig = field(default_factory=DocuMindConfig)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    document: Optional[Document] = None
    fields: Dict[str, ExtractedField] = field(default_factory=dict)
    tables: List[ExtractedTable] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    review_flags: List[ReviewFlag] = field(default_factory=list)
    confidence: Optional[Any] = None
    review: Optional[Any] = None
    visualization: Optional[Any] = None
    step_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_processing_result(self) -> ProcessingResult:
        """Synthesize accumulated context into an immutable ProcessingResult."""
        if self.document is None:
            raise ValueError("Cannot build ProcessingResult without an ingested Document.")

        now = datetime.now(timezone.utc)
        total_time = (now - self.start_time).total_seconds()

        passed_count = sum(1 for v in self.validation_results if v.status.value == "valid")
        warning_count = sum(1 for v in self.validation_results if v.status.value == "warning")
        error_count = sum(1 for v in self.validation_results if v.status.value == "invalid")

        # Compute summary
        mean_field_conf = (
            sum(f.extraction_confidence for f in self.fields.values()) / len(self.fields)
            if self.fields
            else 0.0
        )
        overall_conf = max(0.0, mean_field_conf - (error_count * 0.1) - (warning_count * 0.02))

        summary = ProcessingSummary(
            document_id=self.document.id,
            document_type=self.document.classified_type,
            total_pages=len(self.document.pages),
            total_fields_extracted=len(self.fields),
            total_tables_extracted=len(self.tables),
            validation_passed_count=passed_count,
            validation_warning_count=warning_count,
            validation_error_count=error_count,
            overall_confidence_score=round(overall_conf, 4),
            review_required=len(self.review_flags) > 0 or error_count > 0,
            processing_time_seconds=round(total_time, 4),
        )

        return ProcessingResult(
            document=self.document,
            fields=self.fields,
            tables=self.tables,
            validation_results=self.validation_results,
            review_flags=self.review_flags,
            summary=summary,
            confidence=self.confidence,
            review=self.review,
            visualization=self.visualization,
            errors=self.errors,
            execution_metadata={
                "step_timings_seconds": self.step_timings,
                "environment": self.config.environment,
                **self.metadata,
            },
        )
