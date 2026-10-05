"""Arithmetic consistency rules for document totals, taxes, discounts, and line-item math."""

from typing import Any, Dict, List, Optional

from src.core.config import ValidationConfig
from src.core.models import ExtractedField, ExtractedTable
from src.core.types import DocumentType, SeverityLevel, ValidationStatus
from src.tables.models import Table
from src.validation.base import BaseValidationRule
from src.validation.format_validators import CurrencyFormatValidator
from src.validation.models import ValidationCategory, ValidationIssue


class SubtotalTaxTotalRule(BaseValidationRule):
    """Verifies that subtotal + tax + other fees - discount == grand total within numerical tolerance."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance
        self.currency_parser = CurrencyFormatValidator()

    @property
    def rule_id(self) -> str:
        return "VAL_ARITH_SUBTOTAL_TAX_001"

    @property
    def rule_name(self) -> str:
        return "Subtotal + Tax Grand Total Consistency"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.ARITHMETIC

    @property
    def target_fields(self) -> List[str]:
        return ["subtotal", "tax", "total", "discount", "shipping", "tip"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        subtotal_f = fields.get("subtotal")
        tax_f = fields.get("tax")
        total_f = fields.get("total")

        if not (subtotal_f and total_f):
            return []

        sub_val = self.currency_parser.parse_amount(subtotal_f.normalized_value or subtotal_f.value)
        tot_val = self.currency_parser.parse_amount(total_f.normalized_value or total_f.value)

        if sub_val is None or tot_val is None:
            return []

        tax_val = 0.0
        if tax_f:
            parsed_tax = self.currency_parser.parse_amount(tax_f.normalized_value or tax_f.value)
            if parsed_tax is not None:
                tax_val = parsed_tax

        discount_f = fields.get("discount")
        discount_val = 0.0
        if discount_f:
            parsed_disc = self.currency_parser.parse_amount(discount_f.normalized_value or discount_f.value)
            if parsed_disc is not None:
                discount_val = parsed_disc

        shipping_f = fields.get("shipping")
        shipping_val = 0.0
        if shipping_f:
            parsed_ship = self.currency_parser.parse_amount(shipping_f.normalized_value or shipping_f.value)
            if parsed_ship is not None:
                shipping_val = parsed_ship

        tip_f = fields.get("tip")
        tip_val = 0.0
        if tip_f:
            parsed_tip = self.currency_parser.parse_amount(tip_f.normalized_value or tip_f.value)
            if parsed_tip is not None:
                tip_val = parsed_tip

        expected_total = sub_val + tax_val + shipping_val + tip_val - discount_val
        diff = abs(expected_total - tot_val)

        affected = [f.name for f in [subtotal_f, tax_f, total_f, discount_f, shipping_f, tip_f] if f is not None]

        if diff <= self.tolerance:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.VALID,
                    severity=SeverityLevel.INFO,
                    message=(
                        f"Arithmetic sum verified: subtotal ({sub_val:.2f}) + tax ({tax_val:.2f}) "
                        f"- discount ({discount_val:.2f}) == total ({tot_val:.2f}) [diff: {diff:.2f}]."
                    ),
                    category=self.category,
                    affected_fields=affected,
                    expected_value=round(expected_total, 2),
                    actual_value=round(tot_val, 2),
                    bounding_box=total_f.bounding_box,
                    page_number=total_f.page_number,
                    ocr_confidence=total_f.source_ocr_confidence,
                    extraction_method=total_f.extraction_method,
                )
            ]
        else:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.ERROR,
                    message=(
                        f"Arithmetic mismatch: subtotal ({sub_val:.2f}) + tax ({tax_val:.2f}) "
                        f"- discount ({discount_val:.2f}) = {expected_total:.2f}, but total is {tot_val:.2f} "
                        f"(discrepancy: {diff:.2f} exceeds tolerance {self.tolerance:.2f})."
                    ),
                    category=self.category,
                    affected_fields=affected,
                    expected_value=round(expected_total, 2),
                    actual_value=round(tot_val, 2),
                    bounding_box=total_f.bounding_box or subtotal_f.bounding_box,
                    page_number=total_f.page_number,
                    ocr_confidence=total_f.source_ocr_confidence,
                    extraction_method=total_f.extraction_method,
                )
            ]


class LineItemArithmeticRule(BaseValidationRule):
    """Verifies that line-item quantity * unit_price - discount == amount for each extracted item row."""

    def __init__(self, tolerance: float = 0.02) -> None:
        self.tolerance = tolerance

    @property
    def rule_id(self) -> str:
        return "VAL_ARITH_LINE_ITEM_001"

    @property
    def rule_name(self) -> str:
        return "Line Item Arithmetic Integrity (Qty * Price == Amount)"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.ARITHMETIC

    @property
    def target_fields(self) -> List[str]:
        return ["line_items", "quantity", "unit_price", "amount"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        raw_tables = (context or {}).get("rich_tables", [])
        line_items = (context or {}).get("line_items", [])

        # If rich tables exist in context, inspect their line items
        all_line_items = list(line_items)
        for tbl in raw_tables:
            if hasattr(tbl, "line_items"):
                for li in tbl.line_items:
                    if li not in all_line_items:
                        all_line_items.append(li)

        for idx, item in enumerate(all_line_items):
            qty = getattr(item, "quantity", None)
            price = getattr(item, "unit_price", None)
            amt = getattr(item, "amount", None)
            disc = getattr(item, "discount", 0.0) or 0.0

            if qty is None or price is None or amt is None:
                continue

            expected_amt = (qty * price) - disc
            diff = abs(expected_amt - amt)

            desc = getattr(item, "description", f"Row #{idx + 1}")
            item_bbox = getattr(item, "bounding_box", None)
            item_page = getattr(item, "page_number", 1)

            if diff <= self.tolerance:
                issues.append(
                    ValidationIssue(
                        rule_id=f"VAL_ARITH_LINE_ITEM_{idx + 1}",
                        rule_name=f"Line Item #{idx + 1} Arithmetic",
                        status=ValidationStatus.VALID,
                        severity=SeverityLevel.INFO,
                        message=f"Line #{idx + 1} ('{desc[:30]}'): {qty} * {price:.2f} == {amt:.2f}.",
                        category=self.category,
                        affected_fields=["line_items"],
                        expected_value=round(expected_amt, 2),
                        actual_value=round(amt, 2),
                        bounding_box=item_bbox,
                        page_number=item_page,
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        rule_id=f"VAL_ARITH_LINE_ITEM_{idx + 1}",
                        rule_name=f"Line Item #{idx + 1} Arithmetic",
                        status=ValidationStatus.INVALID,
                        severity=SeverityLevel.ERROR,
                        message=(
                            f"Line #{idx + 1} ('{desc[:30]}') arithmetic mismatch: "
                            f"{qty} * {price:.2f} = {expected_amt:.2f}, but reported amount is {amt:.2f} (diff: {diff:.2f})."
                        ),
                        category=self.category,
                        affected_fields=["line_items"],
                        expected_value=round(expected_amt, 2),
                        actual_value=round(amt, 2),
                        bounding_box=item_bbox,
                        page_number=item_page,
                    )
                )

        return issues


class LineItemsSubtotalSumRule(BaseValidationRule):
    """Verifies that the sum of line item amounts equals the extracted document subtotal (or total)."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance
        self.currency_parser = CurrencyFormatValidator()

    @property
    def rule_id(self) -> str:
        return "VAL_ARITH_LINES_SUBTOTAL_001"

    @property
    def rule_name(self) -> str:
        return "Sum of Line Items vs Subtotal Consistency"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.ARITHMETIC

    @property
    def target_fields(self) -> List[str]:
        return ["subtotal", "total", "line_items"]

    def evaluate(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[ExtractedTable],
        document_type: DocumentType = DocumentType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        raw_tables = (context or {}).get("rich_tables", [])
        line_items = (context or {}).get("line_items", [])

        all_items = list(line_items)
        for tbl in raw_tables:
            if hasattr(tbl, "line_items"):
                for li in tbl.line_items:
                    if li not in all_items:
                        all_items.append(li)

        if not all_items:
            return []

        item_amounts = [li.amount for li in all_items if getattr(li, "amount", None) is not None]
        if not item_amounts:
            return []

        sum_lines = sum(item_amounts)

        subtotal_f = fields.get("subtotal")
        target_f = subtotal_f or fields.get("total")

        if not target_f:
            return []

        target_val = self.currency_parser.parse_amount(target_f.normalized_value or target_f.value)
        if target_val is None:
            return []

        diff = abs(sum_lines - target_val)
        field_label = "subtotal" if subtotal_f else "total"

        if diff <= self.tolerance:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.VALID,
                    severity=SeverityLevel.INFO,
                    message=(
                        f"Line items sum ({sum_lines:.2f}) across {len(item_amounts)} items "
                        f"matches document {field_label} ({target_val:.2f})."
                    ),
                    category=self.category,
                    affected_fields=[target_f.name, "line_items"],
                    expected_value=round(target_val, 2),
                    actual_value=round(sum_lines, 2),
                    bounding_box=target_f.bounding_box,
                    page_number=target_f.page_number,
                    ocr_confidence=target_f.source_ocr_confidence,
                    extraction_method=target_f.extraction_method,
                )
            ]
        else:
            return [
                ValidationIssue(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    status=ValidationStatus.INVALID,
                    severity=SeverityLevel.WARNING,
                    message=(
                        f"Sum of line items ({sum_lines:.2f}) does not match document {field_label} ({target_val:.2f}) "
                        f"[diff: {diff:.2f}]."
                    ),
                    category=self.category,
                    affected_fields=[target_f.name, "line_items"],
                    expected_value=round(target_val, 2),
                    actual_value=round(sum_lines, 2),
                    bounding_box=target_f.bounding_box,
                    page_number=target_f.page_number,
                    ocr_confidence=target_f.source_ocr_confidence,
                    extraction_method=target_f.extraction_method,
                )
            ]


__all__ = [
    "SubtotalTaxTotalRule",
    "LineItemArithmeticRule",
    "LineItemsSubtotalSumRule",
]
