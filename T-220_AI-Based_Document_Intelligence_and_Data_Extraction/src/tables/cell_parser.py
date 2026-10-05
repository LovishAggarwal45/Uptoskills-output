"""Cell value normalization, type classification, bounding box aggregation, and cell-level confidence."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.core.models import BoundingBox
from src.extraction.normalizers import normalize_date, normalize_money, normalize_text
from src.ocr.models import OCRWord
from src.tables.models import TableCell, TableColumn, TableValueType

logger = get_logger("tables.cell_parser")


class CellParser:
    """Parses, normalizes, and types raw cell text and word collections."""

    @staticmethod
    def normalize_cell_value(
        raw_text: str,
        expected_type: Optional[TableValueType] = None,
        canonical_field: Optional[str] = None,
    ) -> Tuple[Optional[Any], TableValueType]:
        """Normalize raw cell string to appropriate Python numeric, date, or clean string representation.

        Args:
            raw_text: Raw string content from cell.
            expected_type: Optional hinted TableValueType.
            canonical_field: Optional semantic field hint (e.g. 'quantity', 'amount', 'unit_price').

        Returns:
            Tuple of (normalized_value, detected_value_type).
        """
        text = raw_text.strip()
        if not text:
            return None, TableValueType.EMPTY

        # 1. Hinted by canonical field
        if canonical_field in ("amount", "unit_price", "tax", "discount"):
            money_val, _ = normalize_money(text)
            if money_val is not None:
                return money_val, TableValueType.CURRENCY

        if canonical_field == "quantity":
            # Attempt integer or float parse
            cleaned_num = re.sub(r"[^\d.-]", "", text)
            if cleaned_num:
                try:
                    if "." in cleaned_num:
                        return float(cleaned_num), TableValueType.DECIMAL
                    return float(int(cleaned_num)), TableValueType.INTEGER
                except ValueError:
                    pass

        # 2. Currency check ($100.00, €50, 1,250.50 with currency symbols)
        if any(sym in text for sym in ["$", "€", "£", "¥", "₹", "USD", "EUR", "GBP"]):
            money_val, _ = normalize_money(text)
            if money_val is not None:
                return money_val, TableValueType.CURRENCY

        # 3. Numeric check (integer, decimal)
        cleaned_digits = re.sub(r"[, ]", "", text)
        if re.match(r"^-?\d+(?:\.\d+)?$", cleaned_digits):
            try:
                if "." in cleaned_digits:
                    return float(cleaned_digits), TableValueType.DECIMAL
                return float(int(cleaned_digits)), TableValueType.INTEGER
            except ValueError:
                pass

        # 4. Percentage check (e.g. 10%, 15.5%)
        pct_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*%$", text)
        if pct_match:
            try:
                pct_val = float(pct_match.group(1))
                return pct_val, TableValueType.PERCENTAGE
            except ValueError:
                pass

        # 5. Date check
        date_val, _ = normalize_date(text)
        if date_val is not None:
            return date_val, TableValueType.DATE

        # 6. Fallback clean text
        cleaned_text = normalize_text(text)
        return cleaned_text, TableValueType.TEXT

    @classmethod
    def populate_cell_values(
        cls,
        cell: TableCell,
        column: Optional[TableColumn] = None,
    ) -> TableCell:
        """Enrich a TableCell with normalized value and validated value type."""
        if not cell.text.strip():
            cell.normalized_value = None
            cell.value_type = TableValueType.EMPTY
            return cell

        expected_type = column.inferred_type if column else None
        canonical = column.canonical_field if column else None

        norm_val, val_type = cls.normalize_cell_value(
            raw_text=cell.text,
            expected_type=expected_type,
            canonical_field=canonical,
        )

        cell.normalized_value = norm_val
        cell.value_type = val_type
        return cell


__all__ = [
    "CellParser",
]
