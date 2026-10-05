"""Candidate conflict detector for identifying competing or ambiguous field values."""

from typing import Any, Dict, List, Optional, Set

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.validation.base import BaseValidationRule
from src.validation.models import ValidationCategory, ValidationIssue


class CandidateConflictDetector(BaseValidationRule):
    """Identifies and flags ambiguous or contradictory extracted candidates for a single semantic field."""

    def __init__(self, confidence_gap_threshold: float = 0.15) -> None:
        self.confidence_gap_threshold = confidence_gap_threshold

    @property
    def rule_id(self) -> str:
        return "VAL_CONF_CANDIDATES_001"

    @property
    def rule_name(self) -> str:
        return "Candidate Conflict & Ambiguity Detection"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CONFLICT

    @property
    def target_fields(self) -> List[str]:
        return ["*"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        for field_name, field_obj in fields.items():
            candidates = field_obj.candidates or []
            if len(candidates) <= 1 and not field_obj.is_ambiguous:
                continue

            # Extract distinct normalized/raw values among candidates
            distinct_values: Dict[str, List[Dict[str, Any]]] = {}
            for cand in candidates:
                cand_val = cand.get("normalized_value") if cand.get("normalized_value") is not None else cand.get("value")
                val_key = str(cand_val).strip().lower()
                if val_key:
                    if val_key not in distinct_values:
                        distinct_values[val_key] = []
                    distinct_values[val_key].append(cand)

            # If multiple distinct semantic values exist
            if len(distinct_values) > 1:
                competing_summary = [
                    f"'{list(c_list)[0].get('value')}' (conf: {list(c_list)[0].get('confidence', 0.0):.2f})"
                    for c_list in distinct_values.values()
                ]

                issues.append(
                    ValidationIssue(
                        rule_id=f"VAL_CONF_{field_name.upper()}",
                        rule_name=f"Field Candidate Conflict: {field_name}",
                        status=ValidationStatus.INVALID,
                        severity=SeverityLevel.WARNING,
                        message=(
                            f"Field '{field_name}' has {len(distinct_values)} distinct competing candidate values: "
                            f"[{', '.join(competing_summary)}]."
                        ),
                        category=self.category,
                        affected_fields=[field_name],
                        expected_value="Single unambiguous candidate value",
                        actual_value=str(field_obj.value),
                        bounding_box=field_obj.bounding_box,
                        page_number=field_obj.page_number,
                        ocr_confidence=field_obj.source_ocr_confidence,
                        extraction_method=field_obj.extraction_method,
                        metadata={
                            "distinct_value_count": len(distinct_values),
                            "candidates": candidates,
                        },
                    )
                )
            elif field_obj.is_ambiguous:
                issues.append(
                    ValidationIssue(
                        rule_id=f"VAL_CONF_{field_name.upper()}",
                        rule_name=f"Field Ambiguity: {field_name}",
                        status=ValidationStatus.WARNING,
                        severity=SeverityLevel.WARNING,
                        message=f"Field '{field_name}' is flagged as ambiguous by extraction heuristics.",
                        category=self.category,
                        affected_fields=[field_name],
                        expected_value="Unambiguous field",
                        actual_value=str(field_obj.value),
                        bounding_box=field_obj.bounding_box,
                        page_number=field_obj.page_number,
                    )
                )
            else:
                # All candidates agree on the same underlying value (e.g. repeated in header)
                issues.append(
                    ValidationIssue(
                        rule_id=f"VAL_CONF_{field_name.upper()}",
                        rule_name=f"Field Candidate Consistency: {field_name}",
                        status=ValidationStatus.VALID,
                        severity=SeverityLevel.INFO,
                        message=f"Field '{field_name}' candidate values are consistent across {len(candidates)} occurrences.",
                        category=self.category,
                        affected_fields=[field_name],
                        expected_value="Consistent candidates",
                        actual_value=str(field_obj.value),
                        bounding_box=field_obj.bounding_box,
                        page_number=field_obj.page_number,
                    )
                )

        return issues


__all__ = ["CandidateConflictDetector"]
