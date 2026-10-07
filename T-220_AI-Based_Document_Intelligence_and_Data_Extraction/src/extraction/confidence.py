"""Explainable extraction confidence derivation."""

from typing import Optional


def calculate_extraction_confidence(
    label_match_strength: float = 1.0,
    pattern_validity_score: float = 1.0,
    spatial_score: float = 1.0,
    candidate_count: int = 1,
    ocr_confidence: Optional[float] = None,
) -> float:
    """Calculate an explainable extraction confidence score in [0.0, 1.0].

    Distinctly separates OCR character recognition confidence from extraction/semantic confidence.

    Args:
        label_match_strength: [0.0, 1.0] indicating precision of matched label.
        pattern_validity_score: [0.0, 1.0] indicating syntactic regex/format conformity.
        spatial_score: [0.0, 1.0] indicating 2D layout and proximity quality.
        candidate_count: Total number of competing candidates detected for this field.
        ocr_confidence: Optional optical confidence reported by OCR engine.

    Returns:
        Confidence score bounded in [0.0, 1.0].
    """
    # Candidate uniqueness score
    if candidate_count <= 1:
        uniqueness_score = 1.0
    elif candidate_count == 2:
        uniqueness_score = 0.75
    else:
        uniqueness_score = max(0.40, 1.0 - (candidate_count * 0.15))

    # OCR quality score (default to 0.85 if OCR engine does not provide confidence)
    ocr_score = ocr_confidence if ocr_confidence is not None else 0.85
    ocr_score = max(0.0, min(1.0, ocr_score))

    # Weighted explainable synthesis
    # Label match (35%), Pattern validity (30%), Spatial proximity (15%), Uniqueness (10%), OCR (10%)
    raw_confidence = (
        0.35 * label_match_strength
        + 0.30 * pattern_validity_score
        + 0.15 * spatial_score
        + 0.10 * uniqueness_score
        + 0.10 * ocr_score
    )

    return round(max(0.0, min(1.0, raw_confidence)), 4)
