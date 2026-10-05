"""Unit tests for Phase 6 table data models and serialization contracts."""

import unittest
from datetime import datetime

from src.core.models import BoundingBox, Provenance
from src.core.types import ConfidenceSource, ExtractionMethod, ValidationStatus
from src.ocr.models import OCRWord
from src.tables.models import (
    LineItem,
    Table,
    TableCell,
    TableColumn,
    TableExtractionResult,
    TableHeader,
    TableRegion,
    TableRow,
    TableValidationResult,
    TableValueType,
)


class TestTableModels(unittest.TestCase):
    """Test suite for Table, TableCell, TableColumn, LineItem, and serialization."""

    def setUp(self) -> None:
        self.bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=550.0, ymax=400.0)
        self.cell_bbox = BoundingBox(xmin=50.0, ymin=100.0, xmax=200.0, ymax=130.0)

    def test_table_cell_creation_and_dict_serialization(self) -> None:
        cell = TableCell(
            row_index=0,
            col_index=1,
            text="$1,250.00",
            normalized_value=1250.0,
            bounding_box=self.cell_bbox,
            confidence=0.98,
            page_number=1,
            value_type=TableValueType.CURRENCY,
            is_header=False,
        )

        self.assertEqual(cell.row_index, 0)
        self.assertEqual(cell.col_index, 1)
        self.assertEqual(cell.normalized_value, 1250.0)
        self.assertEqual(cell.value_type, TableValueType.CURRENCY)

        data = cell.to_dict()
        self.assertEqual(data["row_index"], 0)
        self.assertEqual(data["col_index"], 1)
        self.assertEqual(data["normalized_value"], 1250.0)
        self.assertEqual(data["value_type"], "currency")

        # Round trip
        restored = TableCell.from_dict(data)
        self.assertEqual(restored.row_index, cell.row_index)
        self.assertEqual(restored.col_index, cell.col_index)
        self.assertEqual(restored.normalized_value, cell.normalized_value)
        self.assertEqual(restored.value_type, TableValueType.CURRENCY)

    def test_table_cell_invalid_parameters_raise_error(self) -> None:
        with self.assertRaises(ValueError):
            TableCell(row_index=0, col_index=0, confidence=1.5)

        with self.assertRaises(ValueError):
            TableCell(row_index=0, col_index=0, page_number=0)

    def test_table_cell_to_core_cell_conversion(self) -> None:
        cell = TableCell(
            row_index=1,
            col_index=2,
            text="Cloud Server",
            normalized_value="Cloud Server",
            bounding_box=self.cell_bbox,
            confidence=0.95,
            page_number=1,
        )
        core_cell = cell.to_core_cell()
        self.assertEqual(core_cell.row_index, 1)
        self.assertEqual(core_cell.col_index, 2)
        self.assertEqual(core_cell.text, "Cloud Server")
        self.assertEqual(core_cell.confidence, 0.95)

    def test_table_column_properties_and_dict_serialization(self) -> None:
        col = TableColumn(
            index=0,
            name="Description",
            canonical_field="description",
            x_start=50.0,
            x_end=250.0,
            alignment="left",
            inferred_type=TableValueType.TEXT,
            confidence=0.99,
        )

        self.assertEqual(col.width, 200.0)
        data = col.to_dict()
        self.assertEqual(data["name"], "Description")
        self.assertEqual(data["canonical_field"], "description")
        self.assertEqual(data["width"], 200.0)

        restored = TableColumn.from_dict(data)
        self.assertEqual(restored.index, 0)
        self.assertEqual(restored.name, "Description")
        self.assertEqual(restored.inferred_type, TableValueType.TEXT)

    def test_table_header_and_row_serialization(self) -> None:
        cell1 = TableCell(row_index=0, col_index=0, text="Item", is_header=True)
        cell2 = TableCell(row_index=0, col_index=1, text="Price", is_header=True)
        header = TableHeader(
            row_index=0,
            cells=[cell1, cell2],
            bounding_box=self.cell_bbox,
            confidence=0.96,
            column_names=["Item", "Price"],
        )

        header_dict = header.to_dict()
        restored_hdr = TableHeader.from_dict(header_dict)
        self.assertEqual(len(restored_hdr.cells), 2)
        self.assertEqual(restored_hdr.column_names, ["Item", "Price"])

        row_cell1 = TableCell(row_index=1, col_index=0, text="Product A", normalized_value="Product A")
        row_cell2 = TableCell(row_index=1, col_index=1, text="$50.00", normalized_value=50.0)
        row = TableRow(
            row_index=1,
            cells=[row_cell1, row_cell2],
            bounding_box=self.cell_bbox,
            is_header=False,
            is_summary=False,
            confidence=0.94,
        )

        self.assertIsNotNone(row.get_cell(0))
        self.assertIsNone(row.get_cell(5))

        row_dict = row.to_dict()
        restored_row = TableRow.from_dict(row_dict)
        self.assertEqual(restored_row.row_index, 1)
        self.assertEqual(len(restored_row.cells), 2)

    def test_line_item_model_and_provenance(self) -> None:
        prov = Provenance(
            document_id="doc_inv_1",
            page_number=1,
            raw_text="Widget Pro | 2 | $50.00 | $100.00",
            bounding_box=self.cell_bbox,
            ocr_confidence=0.97,
            extraction_method=ExtractionMethod.TABLE_STRUCTURE_PARSER,
        )

        item = LineItem(
            description="Widget Pro",
            item_code="W-100",
            quantity=2.0,
            unit_price=50.0,
            amount=100.0,
            tax=5.0,
            discount=0.0,
            row_index=1,
            page_number=1,
            bounding_box=self.cell_bbox,
            confidence=0.95,
            provenance=prov,
            is_valid_arithmetic=True,
        )

        item_dict = item.to_dict()
        self.assertEqual(item_dict["description"], "Widget Pro")
        self.assertEqual(item_dict["quantity"], 2.0)
        self.assertEqual(item_dict["amount"], 100.0)
        self.assertTrue(item_dict["is_valid_arithmetic"])

        restored_item = LineItem.from_dict(item_dict)
        self.assertEqual(restored_item.description, "Widget Pro")
        self.assertEqual(restored_item.unit_price, 50.0)
        self.assertIsNotNone(restored_item.provenance)
        self.assertEqual(restored_item.provenance.ocr_confidence, 0.97)

    def test_table_aggregate_and_records_export(self) -> None:
        col1 = TableColumn(index=0, name="Description", canonical_field="description")
        col2 = TableColumn(index=1, name="Quantity", canonical_field="quantity")
        col3 = TableColumn(index=2, name="Amount", canonical_field="amount")

        cell1 = TableCell(row_index=0, col_index=0, text="Item A", normalized_value="Item A")
        cell2 = TableCell(row_index=0, col_index=1, text="2", normalized_value=2.0)
        cell3 = TableCell(row_index=0, col_index=2, text="$20.00", normalized_value=20.0)

        row = TableRow(row_index=0, cells=[cell1, cell2, cell3])
        table = Table(
            table_id="table_01",
            page_number=1,
            headers=["Description", "Quantity", "Amount"],
            columns=[col1, col2, col3],
            rows=[row],
            bounding_box=self.bbox,
            confidence=0.92,
        )

        self.assertEqual(table.row_count, 1)
        self.assertEqual(table.column_count, 3)

        records = table.to_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["Description"], "Item A")
        self.assertEqual(records[0]["Quantity"], 2.0)
        self.assertEqual(records[0]["Amount"], 20.0)

        core_table = table.to_extracted_table()
        self.assertEqual(core_table.table_id, "table_01")
        self.assertEqual(len(core_table.rows), 1)
        self.assertEqual(core_table.rows[0][0].text, "Item A")

    def test_table_extraction_result_json_serialization(self) -> None:
        table = Table(
            table_id="table_01",
            page_number=1,
            headers=["Item", "Total"],
            columns=[TableColumn(index=0, name="Item"), TableColumn(index=1, name="Total")],
            rows=[],
            confidence=0.88,
        )

        res = TableExtractionResult(
            document_id="doc_inv_test",
            tables=[table],
            line_items=[],
            statistics={"total_tables": 1},
        )

        json_str = res.to_json()
        self.assertIn("doc_inv_test", json_str)
        self.assertIn("table_01", json_str)

        data = res.to_dict()
        restored = TableExtractionResult.from_dict(data)
        self.assertEqual(restored.document_id, "doc_inv_test")
        self.assertEqual(len(restored.tables), 1)


if __name__ == "__main__":
    unittest.main()
