
import re

import easyocr


# Initialize EasyOCR once.
reader = easyocr.Reader(["en"], gpu=False)


def normalize_plate(text):
    """Convert OCR output to uppercase letters and digits."""
    if text is None:
        return ""

    if not isinstance(text, str):
        return ""

    text = text.strip().upper()

    # Reject missing values and numeric placeholders.
    if text in {"", "NONE", "NAN", "UNREADABLE", "1.0", "-1"}:
        return ""

    return re.sub(r"[^A-Z0-9]", "", text)


def check_license_plate_format(text):
    """
    Check common Indian registration formats.

    This checks the format only, not whether a registration is genuine.
    """
    text = normalize_plate(text)

    if not text:
        return False

    # Examples: TN09AB1234, KL07BA5252
    state_format = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"

    # Example: 22BH1234AA
    bharat_format = r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"

    return bool(
        re.fullmatch(state_format, text)
        or re.fullmatch(bharat_format, text)
    )


def format_license_number(text):
    """Return normalized OCR text."""
    return normalize_plate(text)


def read_license_plate(img):
    """
    Return (recognized_text, confidence).

    Return (None, 0.0) if no usable OCR text is detected.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return None, 0.0

    try:
        detections = reader.readtext(img)
    except Exception as exc:
        print("EasyOCR warning:", exc)
        return None, 0.0

    best_text = None
    best_score = 0.0

    for detection in detections:
        if not isinstance(detection, (tuple, list)) or len(detection) < 3:
            continue

        _, raw_text, raw_score = detection

        if not isinstance(raw_text, str):
            continue

        text = normalize_plate(raw_text)
        if not text:
            continue

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue

        if not 0.0 <= score <= 1.0:
            continue

        # Prefer higher confidence, with a small tie-breaker for
        # candidates matching an Indian registration format.
        rank = score + (
            0.10 if check_license_plate_format(text) else 0.0
        )

        best_rank = best_score + (
            0.10 if check_license_plate_format(best_text) else 0.0
        ) if best_text else -1.0

        if rank > best_rank:
            best_text = text
            best_score = score

    return best_text, best_score


def map_car(plate, tracking_ids):
    """
    Return the tracked vehicle containing the plate.

    Returns a five-element tuple of -1 values if no vehicle matches.
    """
    try:
        if plate is None or len(plate) < 4:
            return (-1, -1, -1, -1, -1)

        x1, y1, x2, y2 = plate[:4]

        for vehicle in tracking_ids:
            if len(vehicle) < 5:
                continue

            vx1, vy1, vx2, vy2, car_id = vehicle[:5]

            if (
                x1 >= vx1
                and y1 >= vy1
                and x2 <= vx2
                and y2 <= vy2
            ):
                return vehicle

    except (TypeError, ValueError, IndexError) as exc:
        print("Plate-to-vehicle mapping warning:", exc)

    return (-1, -1, -1, -1, -1)
