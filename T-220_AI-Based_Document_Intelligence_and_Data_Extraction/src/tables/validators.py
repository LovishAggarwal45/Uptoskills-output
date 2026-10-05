"""Deterministic mathematical validation, row-level arithmetic checks,
and invoice table sum integrity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.logging import get_logger
from src.extraction.normalizers import normalize_money
from src.tables.base import BaseTableValidator
from src.tables.models import Table, TableRow, TableValidationResult

logger = get_logger("tables.validators")


class TableValidator(BaseTableValidator):
    """Validate line-item arithmetic, invoice totals, and table completeness."""

    def __init__(
        self,
        amount_tolerance: float = 0.02,
        total_tolerance: float = 0.05,
    ) -> None:
        self.amount_tolerance = amount_tolerance
        self.total_tolerance = total_tolerance

    def validate_table(
        self,
        table: Table,
        document_totals: Optional[Dict[str, float]] = None,
    ) -> TableValidationResult:
        """Validate line items and independently verify invoice summary totals."""

        row_passed = 0
        row_failed = 0
        discrepancies: List[Dict[str, Any]] = []
        messages: List[str] = []

        # 1. Validate individual line-item arithmetic.
        for item in table.line_items:
            if item.is_valid_arithmetic is True:
                row_passed += 1
            elif item.is_valid_arithmetic is False:
                row_failed += 1
                messages.extend(item.validation_messages)

                discrepancies.append({
                    "type": "row_arithmetic_mismatch",
                    "row_index": item.row_index,
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "amount": item.amount,
                    "messages": item.validation_messages,
                })

        # 2. Sum the extracted line items.
        calculated_subtotal = round(
            sum(
                item.amount
                for item in table.line_items
                if item.amount is not None
            ),
            2,
        )

        # 3. Obtain reported values from document extraction or summary rows.
        summary_values = self._extract_summary_values(table.rows)

        reported_subtotal = (
            document_totals.get("subtotal")
            if document_totals and document_totals.get("subtotal") is not None
            else summary_values.get("subtotal")
        )

        reported_tax = (
            document_totals.get("tax")
            if document_totals and document_totals.get("tax") is not None
            else summary_values.get("tax")
        )

        reported_total = (
            document_totals.get("total")
            if document_totals and document_totals.get("total") is not None
            else summary_values.get("total")
        )

        # 4. Compare line-item sum with the reported subtotal.
        subtotal_valid: Optional[bool] = None

        if reported_subtotal is not None and table.line_items:
            subtotal_diff = round(
                abs(calculated_subtotal - reported_subtotal), 2
            )
            subtotal_valid = subtotal_diff <= self.total_tolerance

            if not subtotal_valid:
                messages.append(
                    f"Table subtotal mismatch: extracted line items total "
                    f"{calculated_subtotal:.2f}, but reported subtotal is "
                    f"{reported_subtotal:.2f}. Difference: "
                    f"{subtotal_diff:.2f}."
                )

                discrepancies.append({
                    "type": "subtotal_mismatch",
                    "calculated": calculated_subtotal,
                    "reported": reported_subtotal,
                    "difference": subtotal_diff,
                })

        # 5. Validate the invoice summary independently.
        # A valid subtotal + tax = total does not prove that all line
        # items were extracted correctly.
        calculated_total: Optional[float] = None
        summary_valid: Optional[bool] = None
        grand_total_valid: Optional[bool] = None

        if reported_subtotal is not None:
            calculated_total = round(
                reported_subtotal + (reported_tax or 0.0), 2
            )
        elif reported_tax is not None and table.line_items:
            calculated_total = round(
                calculated_subtotal + reported_tax, 2
            )

        if reported_total is not None and calculated_total is not None:
            summary_diff = round(
                abs(calculated_total - reported_total), 2
            )
            summary_valid = summary_diff <= self.total_tolerance

            if not summary_valid:
                messages.append(
                    f"Invoice summary arithmetic mismatch: reported "
                    f"subtotal {reported_subtotal if reported_subtotal is not None else calculated_subtotal:.2f} "
                    f"+ tax {(reported_tax or 0.0):.2f} = "
                    f"{calculated_total:.2f}, but reported total is "
                    f"{reported_total:.2f}. Difference: "
                    f"{summary_diff:.2f}."
                )

                discrepancies.append({
                    "type": "summary_total_mismatch",
                    "calculated": calculated_total,
                    "reported": reported_total,
                    "difference": summary_diff,
                })

        # 6. Independently verify whether extracted items account for
        # the reported total after tax. Do not substitute the reported
        # subtotal here: doing so would conceal missing line items.
        line_items_valid: Optional[bool] = None

        if reported_total is not None and table.line_items:
            line_items_total = round(
                calculated_subtotal + (reported_tax or 0.0), 2
            )
            line_items_diff = round(
                abs(line_items_total - reported_total), 2
            )
            line_items_valid = (
                line_items_diff <= self.total_tolerance
            )

            # The subtotal mismatch already reports this discrepancy
            # when both the extracted and reported subtotals are known.
            # Add a separate completeness finding only if it provides
            # additional information.
            if (
                not line_items_valid
                and subtotal_valid is not False
            ):
                messages.append(
                    f"Line-item completeness warning: extracted line "
                    f"items total {calculated_subtotal:.2f}; adding tax "
                    f"{(reported_tax or 0.0):.2f} gives "
                    f"{line_items_total:.2f}, but the reported invoice "
                    f"total is {reported_total:.2f}. Difference: "
                    f"{line_items_diff:.2f}. Check for missing or "
                    f"incorrectly extracted rows."
                )

                discrepancies.append({
                    "type": "grand_total_mismatch",
                    "calculated": line_items_total,
                    "reported": reported_total,
                    "difference": line_items_diff,
                })

        # 7. Overall validity.
        # Any confirmed subtotal, summary, line-item, or row arithmetic
        # discrepancy keeps the table invalid.
        is_valid = (
            row_failed == 0
            and subtotal_valid is not False
            and summary_valid is not False
            and line_items_valid is not False
        )

        logger.info(
            f"Table '{table.table_id}' validation: "
            f"is_valid={is_valid}, rows_passed={row_passed}, "
            f"rows_failed={row_failed}, "
            f"subtotal_valid={subtotal_valid}, "
            f"summary_valid={summary_valid}, "
            f"line_items_valid={line_items_valid}, "
            f"grand_total_valid={is_valid if reported_total is not None else None}"
        )

        return TableValidationResult(
            is_valid=is_valid,
            row_checks_passed=row_passed,
            row_checks_failed=row_failed,
            subtotal_valid=subtotal_valid,
            calculated_subtotal=(
                calculated_subtotal if table.line_items else None
            ),
            reported_subtotal=reported_subtotal,
            tax_valid=None if reported_tax is None else True,
            calculated_tax=reported_tax,
            reported_tax=reported_tax,
            grand_total_valid=(
                is_valid if reported_total is not None else None
            ),
            calculated_total=calculated_total,
            reported_total=reported_total,
            discrepancies=discrepancies,
            messages=messages,
        )

    def _extract_summary_values(
        self,
        rows: List[TableRow],
    ) -> Dict[str, float]:
        """Extract subtotal, tax, and total values from summary rows."""

        summary_values: Dict[str, float] = {}

        for row in rows:
            if not row.is_summary:
                continue

            row_text = " ".join(
                cell.text for cell in row.cells
            ).lower()

            row_numbers = []

            for cell in row.cells:
                value, _ = normalize_money(cell.text)
                if value is not None:
                    row_numbers.append(value)

            if not row_numbers:
                continue

            value = row_numbers[-1]

            if "subtotal" in row_text or "sub-total" in row_text:
                summary_values["subtotal"] = value
            elif any(
                keyword in row_text
                for keyword in ("tax", "vat", "gst")
            ):
                summary_values["tax"] = value
            elif any(
                keyword in row_text
                for keyword in ("total", "balance due", "amount due")
            ):
                summary_values["total"] = value

        return summary_values


__all__ = ["TableValidator"]