"""Field-level confidence calculation and review trigger evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.config import FieldConfidenceConfig
from src.core.models import ExtractedField
from src.core.types import ValidationStatus
from src.confidence.aggregator import (
    aggregate_weighted_signals,
    compute_confidence_band,
    map_validation_status_to_signal,
)
from src.confidence.models import ConfidenceBand, FieldConfidence
from src.confidence.signals import extract_field_signals
from src.validation.models import ValidationReport


class FieldConfidenceCalculator:
    """Calculates explainable, deterministic confidence scores and review triggers for fields."""

    def __init__(self, config: Optional[FieldConfidenceConfig] = None) -> None:
        self.config = config or FieldConfidenceConfig()

    def calculate(
        self,
        field: ExtractedField,
        validation_report: Optional[ValidationReport] = None,
    ) -> FieldConfidence:
        """Evaluate field confidence without overwriting original extraction or OCR scores.

        Args:
            field: The extracted field to evaluate.
            validation_report: Optional validation report for cross-field consistency.

        Returns:
            FieldConfidence instance containing the aggregated score, band, and review reasons.
        """
        signals = extract_field_signals(field, validation_report)

        # 1. Map validation status to numeric signal
        val_numeric = map_validation_status_to_signal(signals.validation_status)

        # 2. Prepare weighted signal list
        signal_pairs = [
            (signals.extraction_confidence, self.config.extraction_weight),
            (signals.ocr_confidence, self.config.ocr_weight),
            (val_numeric, self.config.validation_weight),
        ]

        # 3. Apply conflict penalty if multiple competing candidate values exist
        conflict_penalty = self.config.conflict_penalty if signals.conflict_count > 0 else 0.0

        # 4. Compute aggregated bounded confidence
        agg_score, breakdown = aggregate_weighted_signals(signal_pairs, penalty=conflict_penalty)

        # 5. Determine categorical confidence band
        band = compute_confidence_band(
            score=agg_score,
            high_threshold=self.config.high_threshold,
            medium_threshold=self.config.medium_threshold,
            low_threshold=self.config.low_threshold,
        )

        # 6. Evaluate review triggers and build human-readable explanations
        review_required = False
        reasons: List[str] = []

        if signals.required_field_missing:
            review_required = True
            reasons.append(f"Required field '{field.name}' is missing or has empty value")

        if signals.validation_status == ValidationStatus.INVALID:
            review_required = True
            val_msg = f"Validation failed for '{field.name}'"
            if field.validation_messages:
                val_msg += f": {'; '.join(field.validation_messages)}"
            reasons.append(val_msg)

        if agg_score < self.config.min_confidence_threshold:
            review_required = True
            reasons.append(
                f"Aggregated confidence ({agg_score:.2f}) below threshold ({self.config.min_confidence_threshold:.2f})"
            )

        if signals.ocr_confidence is not None and signals.ocr_confidence < 0.60:
            review_required = True
            reasons.append(f"Low OCR recognition confidence ({signals.ocr_confidence:.2f})")

        if signals.conflict_count > 0:
            review_required = True
            reasons.append(f"Multiple conflicting extraction candidates detected ({signals.conflict_count} alternatives)")

        if field.is_ambiguous:
            review_required = True
            reasons.append(f"Field extraction marked ambiguous by extractor")

        # 7. Build narrative explanation
        ocr_desc = f"{signals.ocr_confidence:.2f}" if signals.ocr_confidence is not None else "unavailable"
        val_desc = signals.validation_status.value if signals.validation_status else "unvalidated"
        explanation = (
            f"Extraction Conf: {signals.extraction_confidence:.2f} (wt: {self.config.extraction_weight:.2f}), "
            f"OCR Conf: {ocr_desc} (wt: {self.config.ocr_weight:.2f}), "
            f"Validation: {val_desc} (wt: {self.config.validation_weight:.2f}), "
            f"Conflict Penalty: -{conflict_penalty:.2f} -> Final Aggregated: {agg_score:.2f} [{band.value.upper()}]"
        )

        return FieldConfidence(
            field_name=field.name,
            value=field.value,
            normalized_value=field.normalized_value,
            original_extraction_confidence=field.extraction_confidence,
            ocr_confidence=signals.ocr_confidence,
            validation_status=signals.validation_status or ValidationStatus.UNVALIDATED,
            aggregated_confidence=agg_score,
            confidence_band=band,
            review_required=review_required,
            review_reasons=reasons,
            signals=signals,
            provenance=field.provenance,
            explanation=explanation,
            metadata={"breakdown": breakdown, "is_required": field.is_required},
        )
