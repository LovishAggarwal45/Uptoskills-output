"""Canonical line-item extraction, column semantic role mapping, and arithmetic row integrity."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.logging import get_logger
from src.core.models import Provenance
from src.core.types import ExtractionMethod
from src.extraction.normalizers import normalize_money
from src.tables.base import BaseLineItemExtractor
from src.tables.models import LineItem, Table, TableColumn, TableRow, TableValueType
from src.tables.row_parser import is_summary_text

logger = get_logger("tables.line_items")


class LineItemExtractor(BaseLineItemExtractor):
    """Extract typed line items from reconstructed table grids."""

    def __init__(self, amount_tolerance: float = 0.02) -> None:
        self.amount_tolerance = amount_tolerance

    def extract_line_items(
        self,
        table: Table,
        document_id: str = "doc",
    ) -> List[LineItem]:
        """Extract structured line items from a table."""
        if not table.rows:
            return []

        col_roles = self._resolve_column_roles(table.columns)
        line_items: List[LineItem] = []

        for row in table.rows:
            if row.is_header or row.is_summary:
                continue

            row_raw_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            )

            meaningful_text = re.sub(r"[\s|—–\-_:.,]+", "", row_raw_text)
            if not meaningful_text:
                continue

            item = self._extract_single_line_item(
                row=row,
                col_roles=col_roles,
                table=table,
                document_id=document_id,
            )

            if item is not None:
                line_items.append(item)

        logger.info(
            f"Extracted {len(line_items)} line item(s) "
            f"from Table '{table.table_id}'."
        )
        return line_items

    def _resolve_column_roles(
        self,
        columns: List[TableColumn],
    ) -> Dict[int, str]:
        """Map table columns to canonical line-item fields."""
        roles: Dict[int, str] = {}
        assigned: Dict[str, int] = {}

        # 1. Honor explicit mappings supplied by the header detector.
        for index, column in enumerate(columns):
            if column.canonical_field:
                field = column.canonical_field
                roles[index] = field
                assigned[field] = index

        # 2. Handle the observed six-column, headerless invoice layout.
        #
        # col_0 = serial number
        # col_1 = description
        # col_2 = HSN/item code
        # col_3 = quantity
        # col_4 = unit price
        # col_5 = line amount
        #
        # This rule is deliberately restricted to generic column names.
        generic_six_column_invoice = (
            len(columns) == 6
            and all(
                column.name.lower().strip() == f"col_{index}"
                for index, column in enumerate(columns)
            )
        )

        if generic_six_column_invoice:
            return {
                0: "item_serial",
                1: "description",
                2: "item_code",
                3: "quantity",
                4: "unit_price",
                5: "amount",
            }

        # 3. Identify a serial-number column from its header.
        if columns and 0 not in roles and len(columns) > 1:
            first = columns[0]
            first_name = first.name.lower().strip()

            serial_headers = (
                "sl",
                "s.no",
                "s no",
                "sr",
                "serial",
                "position",
                "pos",
                "line",
                "#",
                "no.",
            )

            if any(term in first_name for term in serial_headers):
                roles[0] = "item_serial"
                assigned["item_serial"] = 0

        # 4. Identify the description column.
        if "description" not in assigned:
            text_columns = [
                index
                for index, column in enumerate(columns)
                if index not in roles
                and column.inferred_type == TableValueType.TEXT
            ]

            if text_columns:
                description_index = text_columns[0]
                roles[description_index] = "description"
                assigned["description"] = description_index

        # 5. Infer remaining numeric roles.
        for index, column in enumerate(columns):
            if index in roles:
                continue

            name = column.name.lower().strip()
            inferred_type = column.inferred_type

            if (
                "quantity" not in assigned
                and any(term in name for term in ("quantity", "qty"))
            ):
                roles[index] = "quantity"
                assigned["quantity"] = index

            elif (
                "unit_price" not in assigned
                and any(term in name for term in ("unit price", "rate", "price"))
            ):
                roles[index] = "unit_price"
                assigned["unit_price"] = index

            elif (
                "amount" not in assigned
                and any(term in name for term in ("amount", "line total", "extended"))
            ):
                roles[index] = "amount"
                assigned["amount"] = index

            elif (
                "tax" not in assigned
                and any(term in name for term in ("tax", "vat", "gst"))
            ):
                roles[index] = "tax"
                assigned["tax"] = index

            elif (
                "discount" not in assigned
                and "discount" in name
            ):
                roles[index] = "discount"
                assigned["discount"] = index

            elif (
                "unit_of_measure" not in assigned
                and any(term in name for term in ("unit of measure", "uom"))
            ):
                roles[index] = "unit_of_measure"
                assigned["unit_of_measure"] = index

            elif (
                "item_code" not in assigned
                and any(term in name for term in ("hsn", "item code", "sku", "product code"))
            ):
                roles[index] = "item_code"
                assigned["item_code"] = index

            elif (
                "quantity" not in assigned
                and inferred_type == TableValueType.INTEGER
            ):
                roles[index] = "quantity"
                assigned["quantity"] = index

            elif (
                "unit_price" not in assigned
                and inferred_type == TableValueType.CURRENCY
            ):
                roles[index] = "unit_price"
                assigned["unit_price"] = index

            elif (
                "amount" not in assigned
                and inferred_type == TableValueType.CURRENCY
            ):
                roles[index] = "amount"
                assigned["amount"] = index

            elif (
                "tax" not in assigned
                and inferred_type == TableValueType.CURRENCY
            ):
                roles[index] = "tax"
                assigned["tax"] = index

        return roles

    def _extract_single_line_item(
        self,
        row: TableRow,
        col_roles: Dict[int, str],
        table: Table,
        document_id: str,
    ) -> Optional[LineItem]:
        """Extract and validate a single item from a reconstructed row."""
        raw_data: Dict[str, Any] = {}
        description_parts: List[str] = []
        item_code: Optional[str] = None
        quantity: Optional[float] = None
        unit_price: Optional[float] = None
        amount: Optional[float] = None
        tax: Optional[float] = None
        discount: Optional[float] = None
        uom: Optional[str] = None
        all_row_words = []

        for column_index, cell in enumerate(row.cells):
            column_name = (
                table.columns[column_index].name
                if column_index < len(table.columns)
                else f"col_{column_index}"
            )

            raw_data[column_name] = cell.text
            role = col_roles.get(column_index)
            all_row_words.extend(cell.words)

            text = cell.text.strip()
            if not text:
                continue

            if role == "description":
                description_parts.append(text)

            elif role == "item_code":
                item_code = text

            elif role == "quantity":
                quantity = self._parse_numeric(text)

            elif role in ("unit_price", "amount", "tax", "discount"):
                value, _ = normalize_money(text)
                if value is None:
                    value = self._parse_numeric(text)

                if role == "unit_price":
                    unit_price = value
                elif role == "amount":
                    amount = value
                elif role == "tax":
                    tax = value
                else:
                    discount = value

            elif role == "unit_of_measure":
                uom = text

            elif cell.value_type == TableValueType.TEXT:
                if "description" not in col_roles.values():
                    description_parts.append(text)

        description = " ".join(description_parts).strip()
        row_raw_text = " | ".join(
            cell.text.strip()
            for cell in row.cells
            if cell.text.strip()
        )

        meaningful_text = re.sub(r"[\s|—–\-_:.,]+", "", row_raw_text)
        if not meaningful_text:
            return None

        if (
            not description
            and amount is None
            and unit_price is None
            and quantity is None
        ):
            return None

        # Remove leading serial numbers mistakenly included in descriptions.
        if description:
            serial_match = re.match(
                r"^\s*(\d{1,4})[.)\-:\s]+\s*(\S.*)$",
                description,
            )

            if serial_match and not item_code:
                candidate_description = serial_match.group(2).strip()

                if not re.match(r"^\d+[kKmMgGbB]\b", description):
                    item_code = serial_match.group(1)
                    description = candidate_description

        # Exclude subtotal, tax-summary, and total rows.
        if is_summary_text(description) or is_summary_text(row_raw_text):
            return None

        clean_description = re.sub(r"[^\w]", "", description)
        if (
            not clean_description
            and amount is None
            and unit_price is None
            and quantity is None
        ):
            return None

        # Derive one missing numeric value only when the other two are known.
        if (
            amount is None
            and quantity is not None
            and unit_price is not None
        ):
            amount = round(
                quantity * unit_price - (discount or 0.0),
                2,
            )

        elif (
            quantity is None
            and unit_price is not None
            and amount is not None
            and unit_price > 0
        ):
            derived_quantity = amount / unit_price

            # Avoid inventing fractional quantities for countable products.
            if abs(derived_quantity - round(derived_quantity)) <= 0.000001:
                quantity = float(round(derived_quantity))
            else:
                quantity = round(derived_quantity, 2)

            logger.info(
                "Derived missing quantity for row %s: %s / %s = %s",
                row.row_index + 1,
                amount,
                unit_price,
                quantity,
            )

        elif (
            unit_price is None
            and quantity is not None
            and amount is not None
            and quantity > 0
        ):
            unit_price = round(amount / quantity, 2)

        # Validate arithmetic without changing the reported amount.
        is_valid_arithmetic: Optional[bool] = None
        validation_messages: List[str] = []

        if (
            quantity is not None
            and unit_price is not None
            and amount is not None
        ):
            expected = round(
                quantity * unit_price - (discount or 0.0),
                2,
            )
            difference = abs(expected - amount)

            if difference <= self.amount_tolerance:
                is_valid_arithmetic = True
            else:
                is_valid_arithmetic = False
                validation_messages.append(
                    f"Arithmetic mismatch on Row {row.row_index + 1}: "
                    f"Qty ({quantity}) * Unit Price ({unit_price:.2f}) "
                    f"= {expected:.2f}, but reported Amount is "
                    f"{amount:.2f} (diff: {difference:.2f})"
                )

        # Calculate OCR confidence from words belonging to this row.
        mean_ocr_confidence = (
            sum(word.confidence for word in all_row_words) / len(all_row_words)
            if all_row_words
            else row.confidence
        )

        provenance = Provenance(
            document_id=document_id,
            page_number=row.page_number,
            raw_text=row_raw_text,
            bounding_box=row.bounding_box,
            ocr_confidence=mean_ocr_confidence,
            extraction_method=ExtractionMethod.TABLE_STRUCTURE_PARSER,
        )

        arithmetic_score = (
            0.2
            if is_valid_arithmetic is True
            else 0.0
            if is_valid_arithmetic is False
            else 0.1
        )

        item_confidence = (
            row.confidence * 0.5
            + mean_ocr_confidence * 0.3
            + arithmetic_score
        )

        return LineItem(
            description=description,
            item_code=item_code,
            quantity=quantity,
            unit_price=unit_price,
            amount=amount,
            tax=tax,
            discount=discount,
            unit_of_measure=uom,
            row_index=row.row_index,
            page_number=row.page_number,
            bounding_box=row.bounding_box,
            confidence=min(1.0, max(0.0, item_confidence)),
            provenance=provenance,
            is_valid_arithmetic=is_valid_arithmetic,
            validation_messages=validation_messages,
            raw_data=raw_data,
        )

    @staticmethod
    def _parse_numeric(text: str) -> Optional[float]:
        """Extract a numeric value from OCR text."""
        cleaned = re.sub(r"[^\d.-]", "", text.replace(",", ""))

        if not cleaned or cleaned in {".", "-", "-."}:
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None


__all__ = ["LineItemExtractor"]
