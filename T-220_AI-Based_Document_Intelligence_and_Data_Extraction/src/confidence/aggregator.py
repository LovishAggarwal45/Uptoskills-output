"""Deterministic mathematical aggregation and scoring engine for multi-tier confidence."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.core.types import ValidationStatus
from src.confidence.models import ConfidenceBand


def map_validation_status_to_signal(status: Optional[ValidationStatus]) -> float:
    """Map a categorical validation status to a normalized numeric signal [0.0, 1.0].

    Deterministic mapping rules:
    - VALID: 1.0 (full positive validation contribution)
    - WARNING: 0.60 (mild deduction for non-blocking warnings)
    - INVALID: 0.0 (maximum penalty for failing validation rules)
    - UNVALIDATED / SKIPPED: 0.80 (neutral default when validation was not executed)
    - None: 0.80 (neutral default)
    """
    if status is None:
        return 0.80

    mapping = {
        ValidationStatus.VALID: 1.0,
        ValidationStatus.WARNING: 0.60,
        ValidationStatus.INVALID: 0.0,
        ValidationStatus.UNVALIDATED: 0.80,
        ValidationStatus.SKIPPED: 0.80,
    }
    return mapping.get(status, 0.80)


def compute_confidence_band(
    score: float,
    high_threshold: float = 0.85,
    medium_threshold: float = 0.65,
    low_threshold: float = 0.40,
) -> ConfidenceBand:
    """Classify a continuous confidence score [0.0, 1.0] into an explainable descriptive band.

    NOTE: Confidence bands represent deterministic system states derived from
    OCR, extraction, and validation heuristics. They are NOT calibrated Bayesian probabilities.
    """
    if score >= high_threshold:
        return ConfidenceBand.HIGH
    elif score >= medium_threshold:
        return ConfidenceBand.MEDIUM
    elif score >= low_threshold:
        return ConfidenceBand.LOW
    else:
        return ConfidenceBand.VERY_LOW


def aggregate_weighted_signals(
    signals_and_weights: List[Tuple[Optional[float], float]],
    penalty: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute normalized weighted average across available (non-None) signals.

    If a signal is None, its weight is dynamically omitted and remaining weights
    are renormalized to sum to 1.0. The final result is bounded strictly to [0.0, 1.0].

    Args:
        signals_and_weights: List of tuples (signal_value, weight).
        penalty: Additive penalty subtracted from the score (e.g., conflict penalty).

    Returns:
        Tuple of (aggregated_score_bounded, component_breakdown_dict).
    """
    active_signals: List[Tuple[float, float]] = []
    total_active_weight = 0.0

    for val, w in signals_and_weights:
        if val is not None and w > 0.0:
            clamped_val = max(0.0, min(1.0, float(val)))
            active_signals.append((clamped_val, float(w)))
            total_active_weight += float(w)

    if total_active_weight <= 0.0:
        return (0.0, {})

    raw_sum = sum(val * (w / total_active_weight) for val, w in active_signals)
    penalized = max(0.0, min(1.0, raw_sum - penalty))

    breakdown = {
        f"signal_{idx}": round(val * (w / total_active_weight), 4)
        for idx, (val, w) in enumerate(active_signals)
    }
    if penalty > 0.0:
        breakdown["penalty"] = round(penalty, 4)

    return (round(penalized, 4), breakdown)
