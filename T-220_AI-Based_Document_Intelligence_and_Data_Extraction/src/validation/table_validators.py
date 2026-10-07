"""Table integrity validation rule integrating Phase 6 table verification into document validation."""

from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.tables.models import Table, TableValidationResult
from src.validation.base import BaseValidationRule
from src.validation.models import ValidationCategory, ValidationIssue


class TableIntegrityValidationRule(BaseValidationRule):
    """Bridges Phase 6 table extraction outcomes and mathematical integrity into document validation."""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()

    @property
    def rule_id(self) -> str:
        return "VAL_TBL_INTEGRITY_001"

    @property
    def rule_name(self) -> str:
        return "Table Structural and Mathematical Integrity"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.TABLE

    @property
    def target_fields(self) -> List[str]:
        return ["table", "line_items"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        rich_tables: List[Table] = (context or {}).get("rich_tables", [])

        if not rich_tables and not tables:
            return []

        for tbl in rich_tables:
            val_res = getattr(tbl, "validation_result", None)
            tbl_id = getattr(tbl, "table_id", "table_1")
            tbl_page = getattr(tbl, "page_number", 1)
            tbl_bbox = getattr(tbl, "bounding_box", None)

            if val_res is not None:
                if not val_res.is_valid or val_res.discrepancies:
                    for disc in val_res.discrepancies:
                        issues.append(
                            ValidationIssue(
                                rule_id=f"VAL_TBL_{tbl_id.upper()}",
                                rule_name=f"Table Math Integrity: {tbl_id}",
                                status=ValidationStatus.INVALID,
                                severity=SeverityLevel.ERROR,
                                message=f"Table '{tbl_id}' discrepancy: {disc.get('error', disc.get('type', 'calculation mismatch'))}",
                                category=self.category,
                                affected_fields=["table", "line_items"],
                                expected_value=disc.get("expected"),
                                actual_value=disc.get("actual"),
                                bounding_box=tbl_bbox,
                                page_number=tbl_page,
                                metadata=disc,
                            )
                        )
                else:
                    issues.append(
                        ValidationIssue(
                            rule_id=f"VAL_TBL_{tbl_id.upper()}",
                            rule_name=f"Table Math Integrity: {tbl_id}",
                            status=ValidationStatus.VALID,
                            severity=SeverityLevel.INFO,
                            message=f"Table '{tbl_id}' integrity passed ({val_res.row_checks_passed} row checks passed).",
                            category=self.category,
                            affected_fields=["table"],
                            bounding_box=tbl_bbox,
                            page_number=tbl_page,
                        )
                    )

            # Check if table has headers but zero rows/line_items
            if hasattr(tbl, "headers") and tbl.headers:
                has_rows = bool(getattr(tbl, "rows", []))
                has_items = bool(getattr(tbl, "line_items", []))
                if not has_rows and not has_items:
                    issues.append(
                        ValidationIssue(
                            rule_id=f"VAL_TBL_{tbl_id.upper()}_EMPTY",
                            rule_name=f"Table Content Validation: {tbl_id}",
                            status=ValidationStatus.WARNING,
                            severity=SeverityLevel.WARNING,
                            message=f"Table '{tbl_id}' has declared headers {tbl.headers} but contains 0 data rows.",
                            category=self.category,
                            affected_fields=["table"],
                            bounding_box=tbl_bbox,
                            page_number=tbl_page,
                        )
                    )

        return issues


__all__ = ["TableIntegrityValidationRule"]
