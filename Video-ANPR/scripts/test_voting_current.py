
import runpy

# Load the helper definitions without running the full video pipeline.
# main.py currently runs its pipeline at import time, so we test the
# voting logic independently by reproducing the expected vote structure.

from collections import Counter, defaultdict
import re

def normalize_plate(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())

def validate_indian_plate(text):
    text = normalize_plate(text)
    return bool(
        re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", text)
        or re.fullmatch(r"\d{2}BH\d{4}[A-Z]{1,2}", text)
    )

def choose_final_plate(votes):
    normalized = []
    for text, score, timestamp in votes:
        text = normalize_plate(text)
        if 4 <= len(text) <= 12:
            normalized.append((text, float(score or 0), timestamp))

    if not normalized:
        return {"plate": "UNREADABLE", "vote_count": 0}

    valid = [v for v in normalized if validate_indian_plate(v[0])]
    pool = valid or normalized
    counts = Counter(v[0] for v in pool)
    winner = max(counts, key=lambda t: (counts[t], len(t)))
    return {"plate": winner, "vote_count": counts[winner]}

samples = [
    ("L605HZ", 0.79, 14.0),
    ("L605HZ", 0.81, 14.1),
    ("L60SHZ", 0.74, 14.2),
    ("R783JF", 0.33, 14.3),
]

print("Voting test:", choose_final_plate(samples))
print("No-votes test:", choose_final_plate([]))
