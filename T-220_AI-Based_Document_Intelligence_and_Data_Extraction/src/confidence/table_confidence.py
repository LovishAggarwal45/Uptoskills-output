"""Table and line-item confidence calculation and review trigger evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.config import TableConfidenceConfig
from src.core.types import ValidationStatus
from src.confidence.aggregator import (
    aggregate_weighted_signals,
    compute_confidence_band,
    map_validation_status_to_signal,
)
from src.confidence.models import (
    ConfidenceBand,
    LineItemConfidence,
    TableConfidence,
)
from src.confidence.signals import extract_table_signals
from src.tables.models import LineItem, Table
from src.validation.models import ValidationReport


class TableConfidenceCalculator:
    """Evaluates multi-dimensional confidence scores for reconstructed tables and itemized rows."""

    def __init__(self, config: Optional[TableConfidenceConfig] = None) -> None:
        self.config = config or TableConfidenceConfig()

    def calculate_line_item(self, line_item: LineItem) -> LineItemConfidence:
        """Evaluate confidence and arithmetic consistency for an individual line item."""
        orig_conf = float(line_item.confidence)
        ocr_conf = line_item.provenance.ocr_confidence if line_item.provenance else None

        # Build arithmetic penalty if explicitly invalid
        arith_penalty = 0.25 if line_item.is_valid_arithmetic is False else 0.0

        signals = [
            (orig_conf, 0.70),
            (ocr_conf, 0.30 if ocr_conf is not None else 0.0),
        ]
        agg_score, _ = aggregate_weighted_signals(signals, penalty=arith_penalty)

        band = compute_confidence_band(
            score=agg_score,
            high_threshold=self.config.high_threshold,
            medium_threshold=self.config.medium_threshold,
            low_threshold=self.config.low_threshold,
        )

        review_required = False
        reasons: List[str] = []

        if line_item.is_valid_arithmetic is False:
            review_required = True
            reasons.append(
                f"Row {line_item.row_index + 1} arithmetic mismatch (Qty={line_item.quantity} * UnitPrice={line_item.unit_price} != Amount={line_item.amount})"
            )

        if agg_score < self.config.min_line_item_threshold:
            review_required = True
            reasons.append(
                f"Row {line_item.row_index + 1} confidence ({agg_score:.2f}) below threshold ({self.config.min_line_item_threshold:.2f})"
            )

        if line_item.validation_messages:
            review_required = True
            reasons.extend(line_item.validation_messages)

        return LineItemConfidence(
            row_index=line_item.row_index,
            description=line_item.description,
            aggregated_confidence=agg_score,
            confidence_band=band,
            original_confidence=orig_conf,
            ocr_confidence=ocr_conf,
            is_valid_arithmetic=line_item.is_valid_arithmetic,
            review_required=review_required,
            review_reasons=reasons,
            cell_confidences={"line_item": orig_conf},
            provenance=line_item.provenance,
        )

    def calculate_table(
        self,
        table: Table,
        validation_report: Optional[ValidationReport] = None,
    ) -> TableConfidence:
        """Evaluate structural, row, cell, and validation confidence across an entire table."""
        signals = extract_table_signals(table, validation_report)

        struct_conf = float(signals.metadata.get("structure_confidence", table.confidence))
        row_conf = float(signals.metadata.get("row_confidence", table.confidence))
        cell_conf = float(signals.metadata.get("cell_confidence", table.confidence))

        val_numeric = map_validation_status_to_signal(signals.validation_status)

        signal_pairs = [
            (struct_conf, self.config.structure_weight),
            (row_conf, self.config.row_weight),
            (cell_conf, self.config.cell_weight),
            (val_numeric, self.config.validation_weight),
        ]

        agg_score, _ = aggregate_weighted_signals(signal_pairs)

        band = compute_confidence_band(
            score=agg_score,
            high_threshold=self.config.high_threshold,
            medium_threshold=self.config.medium_threshold,
            low_threshold=self.config.low_threshold,
        )

        # Evaluate individual line items
        line_item_confs: List[LineItemConfidence] = [
            self.calculate_line_item(li) for li in table.line_items
        ]

        # Check review triggers
        review_required = False
        reasons: List[str] = []

        if table.validation_result:
            vr = table.validation_result
            if not vr.is_valid:
                review_required = True
                reasons.append("Table arithmetic integrity validation failed")
            if vr.subtotal_valid is False:
                review_required = True
                reasons.append(
                    f"Table subtotal mismatch: calculated ${vr.calculated_subtotal} vs reported ${vr.reported_subtotal}"
                )
            if vr.grand_total_valid is False:
                review_required = True
                reasons.append(
                    f"Table grand total mismatch: calculated ${vr.calculated_total} vs reported ${vr.reported_total}"
                )
            if vr.row_checks_failed > 0:
                review_required = True
                reasons.append(f"{vr.row_checks_failed} table row(s) failed arithmetic verification")

        if signals.validation_status == ValidationStatus.INVALID:
            review_required = True
            reasons.append(f"Table '{table.table_id}' marked INVALID by validation suite")

        if any(li.review_required for li in line_item_confs):
            review_required = True
            failed_rows = [str(li.row_index + 1) for li in line_item_confs if li.review_required]
            reasons.append(f"Line items requiring review at row(s): {', '.join(failed_rows)}")

        if agg_score < self.config.min_line_item_threshold:
            review_required = True
            reasons.append(
                f"Overall table confidence ({agg_score:.2f}) below threshold ({self.config.min_line_item_threshold:.2f})"
            )

        return TableConfidence(
            table_id=table.table_id,
            page_number=table.page_number,
            structure_confidence=struct_conf,
            row_confidence=row_conf,
            cell_confidence=cell_conf,
            validation_status=signals.validation_status or ValidationStatus.UNVALIDATED,
            aggregated_confidence=agg_score,
            confidence_band=band,
            review_required=review_required,
            review_reasons=reasons,
            line_item_confidences=line_item_confs,
            bounding_box=table.bounding_box,
        )
