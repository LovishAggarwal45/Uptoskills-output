
import re
from collections import Counter, defaultdict


def normalize_plate(text):
    """Remove spaces and symbols commonly introduced by OCR."""
    if not text:
        return ""

    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def validate_indian_plate(text):
    """Basic format check; this does not verify official registration."""
    plate = normalize_plate(text)

    # Standard private-vehicle format
    standard = re.fullmatch(
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",
        plate,
    )

    # Bharat Series format, e.g. 22BH1234AA
    bharat_series = re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{1,2}",
        plate,
    )

    return bool(standard or bharat_series)


def get_final_plate(observations):
    """
    observations: list of (plate_text, OCR_confidence)
    Returns the most frequently observed plate and its vote count.
    """
    votes = defaultdict(float)
    counts = Counter()

    for text, confidence in observations:
        plate = normalize_plate(text)

        if not plate:
            continue

        confidence = max(0.0, min(float(confidence), 1.0))
        votes[plate] += confidence
        counts[plate] += 1

    if not votes:
        return {
            "final_voted_plate": "UNREADABLE",
            "vote_count": 0,
            "final_confidence": 0.0,
            "format_valid": False,
        }

    # Highest accumulated OCR confidence wins.
    final_plate = max(votes, key=votes.get)

    return {
        "final_voted_plate": final_plate,
        "vote_count": counts[final_plate],
        "final_confidence": votes[final_plate] / counts[final_plate],
        "format_valid": validate_indian_plate(final_plate),
    }
