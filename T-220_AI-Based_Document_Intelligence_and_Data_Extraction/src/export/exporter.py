"""Production implementation of structured result exporters (JSON, CSV, Markdown reports)."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.config import StorageConfig
from src.core.logging import get_logger
from src.core.models import ProcessingResult
from src.export.base import BaseExporter

logger = get_logger("export.exporter")


class DocumentExporter(BaseExporter):
    """Handles structured exporting of processing results to JSON, CSV, and Markdown validation reports."""

    def __init__(self, config: Optional[StorageConfig] = None) -> None:
        super().__init__(config=config)
        self.output_json_dir = Path(getattr(self.config, "output_json_dir", getattr(self.config, "json_dir", "outputs/json")))
        self.output_csv_dir = Path(getattr(self.config, "output_csv_dir", getattr(self.config, "csv_dir", "outputs/csv")))
        self.output_reports_dir = Path(getattr(self.config, "output_reports_dir", getattr(self.config, "reports_dir", "outputs/reports")))

        # Ensure base directories exist
        self.output_json_dir.mkdir(parents=True, exist_ok=True)
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)
        self.output_reports_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        result: Optional[ProcessingResult] = None,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        classification: Optional[Any] = None,
        extraction: Optional[Any] = None,
        tables: Optional[Any] = None,
        validation: Optional[Any] = None,
        confidence: Optional[Any] = None,
        review_queue: Optional[Any] = None,
    ) -> Union[Path, Dict[str, Any]]:
        """Export full processing result to formatted JSON.

        Supports both ProcessingResult object or direct parameter components.
        """
        if result is not None:
            doc_id = result.document.id
            dest = Path(output_path) if output_path else self.output_json_dir / f"{doc_id}_result.json"
            dest.parent.mkdir(parents=True, exist_ok=True)

            data = result.to_dict()
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)

            logger.info(f"Exported processing result JSON for '{doc_id}' -> {dest}")
            return dest

        # Component parameter mode
        doc_id = document_id or "doc_unknown"
        clf_dict = classification.to_dict() if hasattr(classification, "to_dict") else {}
        ext_dict = extraction.to_dict() if hasattr(extraction, "to_dict") else {}
        tbl_dict = tables.to_dict() if hasattr(tables, "to_dict") else {}
        val_dict = validation.to_dict() if hasattr(validation, "to_dict") else {}
        cnf_dict = confidence.to_dict() if hasattr(confidence, "to_dict") else {}
        rev_dict = review_queue.to_dict() if hasattr(review_queue, "to_dict") else {}

        data = {
            "document_id": doc_id,
            "filename": filename or "document.pdf",
            "document_type": getattr(classification, "document_type", "unknown"),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "classification": clf_dict,
            "fields": ext_dict.get("fields", {}),
            "entities": ext_dict.get("entities", []),
            "tables": tbl_dict,
            "validation": val_dict,
            "confidence": cnf_dict,
            "review_queue": rev_dict,
        }

        if output_path:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)
            return dest

        return data

    def export_reviewed_json(
        self,
        result: Optional[ProcessingResult] = None,
        corrections: Optional[List[Any]] = None,
        audit_trail: Optional[List[Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        classification: Optional[Any] = None,
        extraction: Optional[Any] = None,
        tables: Optional[Any] = None,
        validation: Optional[Any] = None,
        confidence: Optional[Any] = None,
        decisions: Optional[List[Any]] = None,
    ) -> Union[Path, Dict[str, Any]]:
        """Export reviewed result with human corrections applied and audit trail attached."""
        if result is not None:
            doc_id = result.document.id
            dest = Path(output_path) if output_path else self.output_json_dir / f"{doc_id}_reviewed.json"
            dest.parent.mkdir(parents=True, exist_ok=True)

            data = result.to_dict()
            data["is_reviewed"] = True
            data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            data["corrections_applied"] = corrections or []
            data["audit_trail"] = audit_trail or []

            # Apply corrections to the fields dictionary in export
            if corrections:
                for corr in corrections:
                    corr_dict = corr.to_dict() if hasattr(corr, "to_dict") else (corr if isinstance(corr, dict) else {})
                    fname = corr_dict.get("field_name")
                    if fname and fname in data.get("fields", {}):
                        data["fields"][fname]["original_extracted_value"] = data["fields"][fname].get("value")
                        data["fields"][fname]["value"] = corr_dict.get("corrected_value")
                        data["fields"][fname]["normalized_value"] = corr_dict.get("corrected_value")
                        data["fields"][fname]["is_human_corrected"] = True
                        data["fields"][fname]["correction_reason"] = corr_dict.get("reason")
                        data["fields"][fname]["corrected_by"] = corr_dict.get("reviewer_id")

            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)

            logger.info(f"Exported reviewed result JSON for '{doc_id}' -> {dest}")
            return dest

        # Component parameter mode
        base_data = self.export_json(
            document_id=document_id,
            filename=filename,
            classification=classification,
            extraction=extraction,
            tables=tables,
            validation=validation,
            confidence=confidence,
        )
        data = dict(base_data)
        data["is_reviewed"] = True
        data["reviewed_at"] = datetime.now(timezone.utc).isoformat()

        corr_dicts = [c.to_dict() if hasattr(c, "to_dict") else (c if isinstance(c, dict) else {}) for c in (corrections or [])]
        dec_dicts = [d.to_dict() if hasattr(d, "to_dict") else (d if isinstance(d, dict) else {}) for d in (decisions or [])]
        audit_dicts = [a.to_dict() if hasattr(a, "to_dict") else (a if isinstance(a, dict) else {}) for a in (audit_trail or [])]

        data["corrections_applied"] = corr_dicts
        data["review_decisions"] = dec_dicts
        data["audit_trail"] = audit_dicts

        if corr_dicts:
            for corr in corr_dicts:
                fname = corr.get("field_name")
                if fname and fname in data.get("fields", {}):
                    data["fields"][fname]["original_extracted_value"] = data["fields"][fname].get("value")
                    data["fields"][fname]["value"] = corr.get("corrected_value")
                    data["fields"][fname]["normalized_value"] = corr.get("corrected_value")
                    data["fields"][fname]["is_human_corrected"] = True
                    data["fields"][fname]["correction_reason"] = corr.get("reason")
                    data["fields"][fname]["corrected_by"] = corr.get("reviewer_id")

        if output_path:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)
            return dest

        return data

    def export_csv(
        self,
        result: Optional[Union[ProcessingResult, str]] = None,
        output_dir: Optional[Union[str, Path, Any]] = None,
        extraction: Optional[Any] = None,
    ) -> Union[List[Path], Tuple[bytes, str, str]]:
        """Export extracted tables and structured key-values/entities to CSV.

        If called with (result: ProcessingResult), writes to disk and returns List[Path].
        If called with (document_id: str, tables, extraction), returns (csv_bytes, mime_type, filename).
        """
        if isinstance(result, ProcessingResult):
            doc_id = result.document.id
            target_dir = Path(output_dir) if output_dir else self.output_csv_dir / doc_id
            target_dir.mkdir(parents=True, exist_ok=True)

            csv_paths: List[Path] = []
            tables = result.tables or []

            # 1. Export Tables (if any)
            for idx, table in enumerate(tables):
                tbl_id = getattr(table, "table_id", f"table_{idx + 1}")
                csv_path = target_dir / f"{doc_id}_{tbl_id}.csv"

                headers: List[str] = getattr(table, "headers", [])
                rows_data: List[List[str]] = []

                rows = getattr(table, "rows", [])
                for r in rows:
                    cells = getattr(r, "cells", [])
                    row_vals = [getattr(c, "text", str(c)) for c in cells]
                    if row_vals:
                        rows_data.append(row_vals)

                if not headers and rows_data:
                    max_cols = max(len(r) for r in rows_data)
                    headers = [f"Column_{i + 1}" for i in range(max_cols)]

                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if headers:
                        writer.writerow(headers)
                    for r in rows_data:
                        writer.writerow(r)

                csv_paths.append(csv_path)
            
            # 2. Export Normalized Line Items
            line_items = [
                item
                for table in tables
                for item in (getattr(table, "line_items", []) or [])
            ]

            if line_items:
                line_items_csv_path = (
                    target_dir / f"{doc_id}_line_items.csv"
                )

                with open(
                    line_items_csv_path,
                    "w",
                    newline="",
                    encoding="utf-8-sig",
                ) as f:
                    writer = csv.writer(f)

                    writer.writerow([
                        "Description",
                        "HSN / Item Code",
                        "Quantity",
                        "Unit Price",
                        "Amount",
                        "Tax",
                        "Tax Rate",
                        "Discount",
                        "Unit",
                        "Page",
                        "Confidence",
                        "Arithmetic Valid",
                        "Validation Messages",
                        "Quantity Source",
                    ])

                    for item in line_items:
                        raw_data = getattr(item, "raw_data", {}) or {}
                        raw_quantity = str(
                            raw_data.get("col_3", "")
                        ).strip()

                        quantity_source = (
                            "Inferred from amount / unit price"
                            if not raw_quantity
                            else "OCR extracted"
                        )

                        writer.writerow([
                            getattr(item, "description", ""),
                            getattr(item, "item_code", ""),
                            getattr(item, "quantity", ""),
                            getattr(item, "unit_price", ""),
                            getattr(item, "amount", ""),
                            getattr(item, "tax", ""),
                            getattr(item, "tax_rate", ""),
                            getattr(item, "discount", ""),
                            getattr(item, "unit_of_measure", ""),
                            getattr(item, "page_number", ""),
                            f"{getattr(item, 'confidence', 0.0):.4f}",
                            getattr(item, "is_valid_arithmetic", ""),
                            "; ".join(
                                getattr(item, "validation_messages", []) or []
                            ),
                            quantity_source,
                        ])

                csv_paths.append(line_items_csv_path)

            # 3. Export Extracted Fields & Entities

            # 2. Export Extracted Fields & Entities
            fields = result.fields or {}
            if fields:
                fields_csv_path = target_dir / f"{doc_id}_fields.csv"
                with open(fields_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Field Name", "Value", "Normalized Value", "Type", "Confidence", "Page", "Source Line"])
                    for fname, f_obj in fields.items():
                        val = getattr(f_obj, "value", str(f_obj))
                        norm = getattr(f_obj, "normalized_value", "")
                        ftype = getattr(f_obj, "field_type", "string")
                        ftype_str = ftype.value if hasattr(ftype, "value") else str(ftype)
                        conf = getattr(f_obj, "extraction_confidence", 1.0)
                        page_num = getattr(f_obj, "page_number", 1)
                        src_line = getattr(f_obj, "source_text", "")
                        writer.writerow([fname, val, norm, ftype_str, f"{conf:.4f}", page_num, src_line])
                csv_paths.append(fields_csv_path)

            logger.info(f"Exported {len(csv_paths)} CSV files for '{doc_id}' -> {target_dir}")
            return csv_paths

        # Component in-memory mode: (document_id, tables, extraction)
        doc_id = str(result) if result else "document"
        tables = output_dir  # In this signature, output_dir passed tables
        ext = extraction

        table_list = getattr(tables, "tables", []) if hasattr(tables, "tables") else (tables if isinstance(tables, list) else [])

        # If no tables but extraction has fields, export fields CSV
        if not table_list and ext:
            fields_map = getattr(ext, "fields", {}) if hasattr(ext, "fields") else (ext if isinstance(ext, dict) else {})
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["Field Name", "Value", "Normalized Value", "Confidence"])
            for fname, f_obj in fields_map.items():
                val = getattr(f_obj, "value", str(f_obj))
                norm = getattr(f_obj, "normalized_value", "")
                conf = getattr(f_obj, "extraction_confidence", 1.0)
                writer.writerow([fname, val, norm, f"{conf:.4f}"])

            content = out.getvalue().encode("utf-8")
            return content, "text/csv; charset=utf-8", f"{doc_id}_fields.csv"

        # If exactly 1 table and no other multiple tables, return clean single CSV
        if len(table_list) <= 1:
            table = table_list[0] if table_list else None
            headers = getattr(table, "headers", []) if table else []
            rows = getattr(table, "rows", []) if table else []

            out = io.StringIO()
            writer = csv.writer(out)
            if headers:
                writer.writerow(headers)

            for r in rows:
                cells = getattr(r, "cells", []) if hasattr(r, "cells") else r
                row_vals = [getattr(c, "text", str(c)) for c in cells]
                writer.writerow(row_vals)

            content = out.getvalue().encode("utf-8")
            return content, "text/csv; charset=utf-8", f"{doc_id}_table.csv"

        # Multiple tables -> ZIP archive of CSVs
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, tbl in enumerate(table_list):
                t_id = getattr(tbl, "table_id", f"table_{idx+1}")
                headers = getattr(tbl, "headers", [])
                rows = getattr(tbl, "rows", [])
                t_out = io.StringIO()
                t_writer = csv.writer(t_out)
                if headers:
                    t_writer.writerow(headers)
                for r in rows:
                    cells = getattr(r, "cells", []) if hasattr(r, "cells") else r
                    t_writer.writerow([getattr(c, "text", str(c)) for c in cells])
                zip_file.writestr(f"{doc_id}_{t_id}.csv", t_out.getvalue())

        zip_buffer.seek(0)
        return zip_buffer.getvalue(), "application/zip", f"{doc_id}_tables.zip"

    def export_validation_report(
        self,
        result: Optional[ProcessingResult] = None,
        output_path: Optional[Union[str, Path]] = None,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        validation: Optional[Any] = None,
        confidence: Optional[Any] = None,
        review_queue: Optional[Any] = None,
    ) -> Union[Path, str]:
        """Export comprehensive Markdown validation report."""
        if result is not None:
            doc_id = result.document.id
            dest = Path(output_path) if output_path else self.output_reports_dir / f"{doc_id}_validation_report.md"
            dest.parent.mkdir(parents=True, exist_ok=True)

            summary = result.summary
            doc = result.document
            meta = doc.metadata

            val_err_cnt = getattr(summary, "validation_error_count", None)
            if val_err_cnt is None and isinstance(summary, dict):
                val_err_cnt = summary.get("validation_error_count", 0)
            val_err_cnt = val_err_cnt or 0

            overall_conf = getattr(summary, "overall_confidence_score", None)
            if overall_conf is None and isinstance(summary, dict):
                overall_conf = summary.get("overall_confidence_score", 1.0)
            overall_conf = overall_conf if overall_conf is not None else 1.0

            val_pass_cnt = getattr(summary, "validation_passed_count", None)
            if val_pass_cnt is None and isinstance(summary, dict):
                val_pass_cnt = summary.get("validation_passed_count", 0)
            val_pass_cnt = val_pass_cnt or 0

            val_warn_cnt = getattr(summary, "validation_warning_count", None)
            if val_warn_cnt is None and isinstance(summary, dict):
                val_warn_cnt = summary.get("validation_warning_count", 0)
            val_warn_cnt = val_warn_cnt or 0

            lines: List[str] = [
                f"# DocuMind AI — Document Validation & Integrity Report",
                f"",
                f"**Document ID:** `{doc_id}`  ",
                f"**Filename:** `{meta.filename if meta else 'N/A'}`  ",
                f"**Generated At:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  ",
                f"",
                f"---",
                f"",
                f"## 1. Executive Summary",
                f"",
                f"| Metric | Value |",
                f"| :--- | :--- |",
                f"| Overall Status | `{'VALID' if val_err_cnt == 0 else 'WARNING'}` |",
                f"| Overall Confidence Score | `{overall_conf:.2%}` |",
                f"| Validation Passed Rules | `{val_pass_cnt}` |",
                f"| Validation Warnings | `{val_warn_cnt}` |",
                f"| Validation Errors | `{val_err_cnt}` |",
                f"",
                f"---",
                f"",
                f"## 2. Validation Findings & Consistency Checks",
                f"",
            ]

            if not result.validation_results:
                lines.append("✓ **No validation issues detected. All rules evaluated successfully.**\n")
            else:
                lines.append("| Rule ID | Severity | Status | Affected Fields | Message |")
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for vr in result.validation_results:
                    sev = getattr(vr.severity, "value", str(vr.severity)).upper()
                    stat = getattr(vr.status, "value", str(vr.status)).upper()
                    aff = ", ".join(vr.affected_fields) if vr.affected_fields else "N/A"
                    lines.append(f"| `{vr.rule_id}` | `{sev}` | `{stat}` | `{aff}` | {vr.message} |")
                lines.append("")

            report_content = "\n".join(lines)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(report_content)

            return dest

        # Component parameter mode
        doc_id = document_id or "document"
        fn = filename or "document.pdf"
        val = validation
        issues = getattr(val, "issues", []) if val else []
        status_val = getattr(val, "overall_status", "VALID")
        status_str = getattr(status_val, "value", str(status_val)).upper()

        lines = [
            f"# DocuMind AI — Document Validation & Integrity Report",
            f"",
            f"**Document ID:** `{doc_id}`  ",
            f"**Filename:** `{fn}`  ",
            f"**Generated At:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  ",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| Overall Status | `{status_str}` |",
            f"| Total Issues | `{len(issues)}` |",
            f"",
            f"---",
            f"",
            f"## 2. Validation Findings",
            f"",
        ]

        if not issues:
            lines.append("✓ **No validation issues detected. All rules passed.**\n")
        else:
            lines.append("| Rule ID | Severity | Message |")
            lines.append("| :--- | :--- | :--- |")
            for issue in issues:
                rule_id = getattr(issue, "rule_id", "RULE-01")
                sev = getattr(issue, "severity", "INFO")
                sev_str = getattr(sev, "value", str(sev)).upper()
                msg = getattr(issue, "message", "")
                lines.append(f"| `{rule_id}` | `{sev_str}` | {msg} |")
            lines.append("")

        report_content = "\n".join(lines)
        if output_path:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(report_content)
            return dest

        return report_content


__all__ = ["DocumentExporter"]
