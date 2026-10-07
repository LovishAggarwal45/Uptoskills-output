"""Unit tests for Phase 6 table JSON and dictionary serialization round-trips."""

import json
import unittest

from src.core.models import BoundingBox, Provenance
from src.core.types import ConfidenceSource, ExtractionMethod, ValidationStatus
from src.tables.models import (
    LineItem,
    Table,
    TableCell,
    TableColumn,
    TableExtractionResult,
    TableHeader,
    TableRow,
    TableValidationResult,
    TableValueType,
)


class TestTableSerialization(unittest.TestCase):
    """Test suite for Table models complete JSON round-trip serialization."""

    def test_full_table_json_roundtrip(self) -> None:
        bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=600.0, ymax=500.0)
        prov = Provenance(
            document_id="doc_test_1",
            page_number=1,
            raw_text="Widget Pro | 2 | $50.00 | $100.00",
            bounding_box=bbox,
            ocr_confidence=0.97,
            extraction_method=ExtractionMethod.TABLE_STRUCTURE_PARSER,
        )

        col1 = TableColumn(index=0, name="Description", canonical_field="description", x_start=50.0, x_end=250.0, alignment="left", inferred_type=TableValueType.TEXT)
        col2 = TableColumn(index=1, name="Qty", canonical_field="quantity", x_start=250.0, x_end=350.0, alignment="right", inferred_type=TableValueType.INTEGER)
        col3 = TableColumn(index=2, name="Amount", canonical_field="amount", x_start=350.0, x_end=600.0, alignment="right", inferred_type=TableValueType.CURRENCY)

        cell1 = TableCell(row_index=0, col_index=0, text="Widget Pro", normalized_value="Widget Pro", bounding_box=bbox, confidence=0.98, value_type=TableValueType.TEXT)
        cell2 = TableCell(row_index=0, col_index=1, text="2", normalized_value=2.0, bounding_box=bbox, confidence=0.99, value_type=TableValueType.INTEGER)
        cell3 = TableCell(row_index=0, col_index=2, text="$100.00", normalized_value=100.0, bounding_box=bbox, confidence=0.97, value_type=TableValueType.CURRENCY)

        row = TableRow(row_index=0, cells=[cell1, cell2, cell3], bounding_box=bbox, confidence=0.98)
        item = LineItem(description="Widget Pro", quantity=2.0, unit_price=50.0, amount=100.0, bounding_box=bbox, confidence=0.96, provenance=prov, is_valid_arithmetic=True)

        val_res = TableValidationResult(is_valid=True, row_checks_passed=1, subtotal_valid=True, calculated_subtotal=100.0, reported_subtotal=100.0)

        table = Table(
            table_id="tbl_roundtrip_01",
            page_number=1,
            headers=["Description", "Qty", "Amount"],
            columns=[col1, col2, col3],
            rows=[row],
            line_items=[item],
            bounding_box=bbox,
            confidence=0.94,
            confidence_breakdown={"header": 1.0, "columns": 0.95},
            confidence_source=ConfidenceSource.HEURISTIC,
            validation_status=ValidationStatus.VALID,
            validation_result=val_res,
        )

        result = TableExtractionResult(
            document_id="doc_test_1",
            tables=[table],
            line_items=[item],
            statistics={"total_tables": 1, "total_lines": 1},
        )

        json_str = result.to_json()
        data = json.loads(json_str)

        restored_res = TableExtractionResult.from_dict(data)
        self.assertEqual(restored_res.document_id, "doc_test_1")
        self.assertEqual(len(restored_res.tables), 1)

        restored_table = restored_res.tables[0]
        self.assertEqual(restored_table.table_id, "tbl_roundtrip_01")
        self.assertEqual(len(restored_table.columns), 3)
        self.assertEqual(len(restored_table.rows), 1)
        self.assertEqual(len(restored_table.line_items), 1)
        self.assertTrue(restored_table.validation_result.is_valid)
        self.assertEqual(restored_table.line_items[0].description, "Widget Pro")
        self.assertEqual(restored_table.line_items[0].provenance.ocr_confidence, 0.97)


if __name__ == "__main__":
    unittest.main()
