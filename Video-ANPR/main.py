import os
import re
import csv
import sys
import types
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch


import easyocr

from ultralytics import YOLO
from sort.sort import Sort
from utils import map_car

# Real-ESRGAN compatibility
import torchvision.transforms.functional as functional

module_name = "torchvision.transforms.functional_tensor"
if module_name not in sys.modules:
    compatibility_module = types.ModuleType(module_name)
    compatibility_module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module_name] = compatibility_module

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = os.environ.get("ANPR_VIDEO_PATH", "./uploads/ANPR.mp4")

OUTPUT_DIR = Path(os.environ.get("ANPR_OUTPUT_DIR", "./outputs"))
PLATE_DIR = OUTPUT_DIR / "plate_crops"
ENHANCED_DIR = OUTPUT_DIR / "enhanced_plates"

OUTPUT_VIDEO = OUTPUT_DIR / "anpr_realesrgan_output.mp4"
OUTPUT_CSV = OUTPUT_DIR / "anpr_realesrgan_results.csv"

WEIGHTS = Path("./weights/RealESRGAN_x4plus.pth")
PLATE_MODEL_PATH = Path("./models/license_plate_detector.pt")

FRAME_SKIP = 3
VEHICLE_CLASSES = {2, 3, 5, 7}

MIN_PLATE_LENGTH = 5
MAX_PLATE_LENGTH = 12

# Expand detected plate crop before OCR.
CROP_PADDING_X = 0.30
CROP_PADDING_Y = 0.45

for folder in (OUTPUT_DIR, PLATE_DIR, ENHANCED_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# OCR INITIALIZATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("OCR device:", DEVICE)

reader = easyocr.Reader(
    ["en"],
    gpu=(DEVICE == "cuda")
)

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ============================================================
# INDIAN PLATE VALIDATION
# ============================================================

PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"),
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),
]


def normalize_plate(value):
    if value is None:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper()
    )


def validate_indian_plate(value):
    text = normalize_plate(value)

    return bool(text) and any(
        pattern.fullmatch(text)
        for pattern in PLATE_PATTERNS
    )


def plausible_plate(value):
    text = normalize_plate(value)

    return MIN_PLATE_LENGTH <= len(text) <= MAX_PLATE_LENGTH


# ============================================================
# PLATE CROP WITH PADDING
# ============================================================

def extract_plate_crop(frame, box):
    x1, y1, x2, y2 = map(float, box[:4])

    height, width = frame.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = max(8, int(box_width * CROP_PADDING_X))
    pad_y = max(8, int(box_height * CROP_PADDING_Y))

    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(width, int(x2) + pad_x)
    y2 = min(height, int(y2) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# OCR PREPROCESSING
# ============================================================

def create_ocr_variants(image):
    if image is None or image.size == 0:
        return []

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    variants = [image, gray]

    # Enlarge small plate images.
    enlarged = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(enlarged)

    # Contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(enlarged)

    variants.append(clahe)

    # Otsu threshold.
    _, otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))

    # Adaptive threshold.
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    variants.append(adaptive)
    variants.append(cv2.bitwise_not(adaptive))

    return variants


def read_best_ocr(image):
    """
    Return the best OCR text and its EasyOCR confidence.
    Numeric placeholders and implausibly short strings are rejected.
    """

    best_text = ""
    best_score = 0.0
    best_rank = -1.0

    variants = create_ocr_variants(image)

    for variant in variants:
        try:
            detections = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=OCR_ALLOWLIST
            )

        except Exception as exc:
            print("OCR warning:", exc)
            continue

        for detection in detections:
            if len(detection) < 3:
                continue

            _, raw_text, raw_score = detection

            text = normalize_plate(raw_text)

            if not plausible_plate(text):
                continue

            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            if not 0 <= score <= 1:
                continue

            # Prefer valid Indian plate formats when confidence
            # values are reasonably comparable.
            rank = score + (
                0.15 if validate_indian_plate(text) else 0
            )

            if rank > best_rank:
                best_text = text
                best_score = score
                best_rank = rank

    return best_text, best_score


# ============================================================
# VEHICLE MAPPING
# ============================================================

def get_car_id(plate_box, tracking_ids):
    try:
        result = map_car(plate_box, tracking_ids)

        if result is None or len(result) < 5:
            return -1

        return int(result[4])

    except Exception as exc:
        print("Vehicle mapping warning:", exc)
        return -1


# ============================================================
# MULTI-FRAME VOTING
# ============================================================

def choose_final_plate(votes):
    if not votes:
        return {
            "plate": "UNREADABLE",
            "vote_count": 0,
            "confidence": 0.0,
            "format_valid": False,
            "status": "UNREADABLE"
        }

    valid_votes = [
        item for item in votes
        if validate_indian_plate(item[0])
    ]

    candidates = valid_votes if valid_votes else votes

    counts = Counter(
        text for text, score, timestamp in candidates
    )

    scores = defaultdict(list)

    for text, score, timestamp in candidates:
        scores[text].append(float(score))

    winner = max(
        counts,
        key=lambda text: (
            counts[text],
            sum(scores[text]) / len(scores[text])
        )
    )

    confidence = sum(scores[winner]) / len(scores[winner])
    valid = validate_indian_plate(winner)

    return {
        "plate": winner,
        "vote_count": counts[winner],
        "confidence": round(confidence, 4),
        "format_valid": valid,
        "status": (
            "INDIAN_FORMAT_VALID"
            if valid
            else "UNVERIFIED_FORMAT"
        )
    }


# ============================================================
# INPUT CHECKS
# ============================================================

if not Path(VIDEO_PATH).is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

if not WEIGHTS.is_file():
    raise FileNotFoundError(f"Real-ESRGAN weights not found: {WEIGHTS}")

if not PLATE_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Plate detector not found: {PLATE_MODEL_PATH}"
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading vehicle detector...")
vehicle_model = YOLO("yolov8n.pt")

print("Loading plate detector...")
plate_model = YOLO(str(PLATE_MODEL_PATH))

device = torch.device(DEVICE)

esrgan_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=str(WEIGHTS),
    model=esrgan_model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=(device.type == "cuda"),
    device=device
)

tracker = Sort()

print("All models loaded.")


# ============================================================
# VIDEO METADATA
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

if fps <= 0:
    fps = 25.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0 or height <= 0:
    cap.release()
    raise ValueError("Invalid video dimensions.")


# ============================================================
# FIRST PASS: DETECTION AND OCR
# ============================================================

frame_results = []
plate_votes = defaultdict(list)
annotations = []
seen_cars = set()

frame_number = 0

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_number / fps
        frame_for_detection = frame.copy()

        if frame_number % FRAME_SKIP == 0:

            vehicle_result = vehicle_model(
                frame_for_detection,
                verbose=False
            )[0]

            vehicle_detections = []

            for box in vehicle_result.boxes.data.tolist():

                x1, y1, x2, y2, score, class_id = box

                if int(class_id) in VEHICLE_CLASSES:
                    vehicle_detections.append(
                        [x1, y1, x2, y2, score]
                    )

            if vehicle_detections:
                tracking_ids = tracker.update(
                    np.asarray(
                        vehicle_detections,
                        dtype=np.float32
                    )
                )
            else:
                tracking_ids = tracker.update(
                    np.empty((0, 5), dtype=np.float32)
                )

            for vehicle in tracking_ids:
                seen_cars.add(int(vehicle[4]))

            plate_result = plate_model(
                frame_for_detection,
                verbose=False
            )[0]

            for plate in plate_result.boxes.data.tolist():

                x1, y1, x2, y2, detector_score, _ = plate

                car_id = get_car_id(plate, tracking_ids)

                if car_id < 0:
                    continue

                crop = extract_plate_crop(
                    frame_for_detection,
                    plate
                )

                if crop is None:
                    continue

                original_path = (
                    PLATE_DIR /
                    f"frame_{frame_number:06d}_car_{car_id}.jpg"
                )

                cv2.imwrite(str(original_path), crop)

                # OCR before enhancement.
                original_text, original_score = read_best_ocr(crop)

                # Real-ESRGAN enhancement.
                try:
                    enhanced_image, _ = upsampler.enhance(
                        crop,
                        outscale=4
                    )
                except Exception as exc:
                    print("Enhancement warning:", exc)
                    enhanced_image = crop.copy()

                enhanced_path = (
                    ENHANCED_DIR /
                    f"enhanced_frame_{frame_number:06d}_car_{car_id}.jpg"
                )

                cv2.imwrite(str(enhanced_path), enhanced_image)

                # OCR after enhancement.
                enhanced_text, enhanced_score = read_best_ocr(
                    enhanced_image
                )

                # Vote using the strongest usable prediction.
                if enhanced_text and enhanced_score >= original_score:
                    vote_text = enhanced_text
                    vote_score = enhanced_score

                elif original_text:
                    vote_text = original_text
                    vote_score = original_score

                else:
                    vote_text = ""
                    vote_score = 0.0

                if vote_text:
                    plate_votes[car_id].append(
                        (vote_text, vote_score, timestamp)
                    )

                frame_results.append({
                    "frame": frame_number,
                    "time_sec": round(timestamp, 3),
                    "car_id": car_id,
                    "plate_detection_confidence": round(
                        float(detector_score), 4
                    ),

                    "original_ocr": original_text,
                    "original_ocr_confidence": round(
                        float(original_score), 4
                    ),

                    "enhanced_ocr": enhanced_text,
                    "enhanced_ocr_confidence": round(
                        float(enhanced_score), 4
                    ),

                    # Dashboard compatibility columns.
                    "original_prediction": original_text,
                    "enhanced_prediction": enhanced_text,

                    "original_valid": validate_indian_plate(
                        original_text
                    ),
                    "enhanced_valid": validate_indian_plate(
                        enhanced_text
                    ),

                    "original_crop": str(original_path),
                    "enhanced_crop": str(enhanced_path)
                })

                annotations.append({
                    "frame": frame_number,
                    "bbox": tuple(map(int, plate[:4])),
                    "car_id": car_id,
                    "detector_confidence": float(detector_score)
                })

        frame_number += 1

        if frame_number % 50 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

finally:
    cap.release()


# ============================================================
# FINAL VOTING
# ============================================================

final_votes = {
    car_id: choose_final_plate(plate_votes[car_id])
    for car_id in seen_cars
}

print("\nFINAL PLATE RESULTS")

for car_id, result in sorted(final_votes.items()):
    print(
        f"Vehicle {car_id}: {result['plate']} | "
        f"votes={result['vote_count']} | "
        f"confidence={result['confidence']:.3f} | "
        f"status={result['status']}"
    )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

fieldnames = [
    "frame",
    "time_sec",
    "car_id",
    "plate_detection_confidence",

    "original_ocr",
    "original_ocr_confidence",
    "original_prediction",
    "original_valid",

    "enhanced_ocr",
    "enhanced_ocr_confidence",
    "enhanced_prediction",
    "enhanced_valid",

    "final_voted_plate",
    "vote_count",
    "final_confidence",
    "format_valid",
    "final_status",

    "original_crop",
    "enhanced_crop"
]

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in frame_results:

        final = final_votes.get(
            row["car_id"],
            choose_final_plate([])
        )

        row["final_voted_plate"] = final["plate"]
        row["vote_count"] = final["vote_count"]
        row["final_confidence"] = final["confidence"]
        row["format_valid"] = final["format_valid"]
        row["final_status"] = final["status"]

        writer.writerow(row)


# ============================================================
# SECOND PASS: ANNOTATED OUTPUT VIDEO
# ============================================================

annotations_by_frame = defaultdict(list)

for item in annotations:
    annotations_by_frame[item["frame"]].append(item)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError("Cannot reopen source video.")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise ValueError("Cannot create output video.")

frame_number = 0

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_number / fps

        for item in annotations_by_frame.get(frame_number, []):

            x1, y1, x2, y2 = item["bbox"]
            car_id = item["car_id"]

            final = final_votes.get(
                car_id,
                choose_final_plate([])
            )

            label = (
                f"{final['plate']} | "
                f"OCR:{final['confidence']:.2f} | "
                f"{timestamp:.2f}s"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_number} | Time: {timestamp:.2f}s",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)
        frame_number += 1

finally:
    cap.release()
    writer.release()


# ============================================================
# COMPLETION
# ============================================================

print("\n========================================")
print("ANPR + REAL-ESRGAN PIPELINE COMPLETE")
print("========================================")
print("Input video:", VIDEO_PATH)
print("Output video:", OUTPUT_VIDEO)
print("Output CSV:", OUTPUT_CSV)
print("Original crops:", PLATE_DIR)
print("Enhanced crops:", ENHANCED_DIR)
print("Processed frames:", frame_number)
print("========================================")
