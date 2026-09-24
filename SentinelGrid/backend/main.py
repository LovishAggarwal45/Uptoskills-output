"""Geometry-aware multi-camera crowd counting API.

Frames from calibrated cameras are detected independently, projected to a shared
ground plane, then fused.  Each output event is persisted as CSV-compatible JSON.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from ultralytics import YOLO
except ImportError:  # Lets the API start before the model is installed.
    YOLO = None

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DATA = ROOT / "data"
EVENTS = DATA / "events.csv"
SAVED_CALIBRATIONS = ROOT / "config" / "calibrations.json"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="Occlusion-Resistant Crowd Surveillance API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/frontend", StaticFiles(directory=str(FRONTEND)), name="frontend")


class Calibration(BaseModel):
    camera_id: str
    # 3x3 inverse homography: image pixels -> ground-plane metres.
    image_to_ground: list[list[float]] = Field(min_length=3, max_length=3)


class CameraPose(BaseModel):
    """Pinhole calibration, with world-to-camera R and t."""
    camera_id: str
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]


class Settings(BaseModel):
    capacity_limit: int = Field(default=50, ge=1)
    cluster_distance_m: float = Field(default=0.8, gt=0, le=5)
    detection_conf_thresh: float = Field(default=0.30, ge=0.05, le=1.0)
    location: str = "Demo venue"


settings = Settings()
calibrations: dict[str, np.ndarray] = {}
model: Any = None

# When the WILDTRACK calibration file is placed in config/, demo cameras are
# ready as soon as the server starts. Manual API registration remains available.
if SAVED_CALIBRATIONS.exists():
    try:
        for saved in json.loads(SAVED_CALIBRATIONS.read_text(encoding="utf-8")):
            calibrations[saved["camera_id"]] = np.asarray(saved["image_to_ground"], dtype=np.float32)
    except (ValueError, KeyError, TypeError):
        pass


def get_model():
    global model
    if model is None:
        if YOLO is None:
            raise HTTPException(503, "Model package missing. Run: pip install -r requirements.txt")
        weights = ROOT / "models" / "best.pt"
        # Falls back to pretrained COCO person detector until Colab-trained weights are copied here.
        model = YOLO(str(weights) if weights.exists() else "yolo11n.pt")
    return model


def detect_and_project(
    image: np.ndarray,
    camera_id: str,
    homography: np.ndarray,
    conf_thresh: float = 0.30,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    """Detect people with YOLO, project foot-points to ground plane, and annotate frame."""
    result = get_model()(image, classes=[0], conf=conf_thresh, verbose=False)[0]
    annotated = image.copy()
    projected: list[dict[str, Any]] = []
    boxes_info: list[dict[str, Any]] = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0])
        foot_x = (x1 + x2) / 2
        foot_y = y2

        pixel = np.array([[foot_x, foot_y]], dtype=np.float32).reshape(-1, 1, 2)
        gx, gy = cv2.perspectiveTransform(pixel, homography)[0, 0]

        point_record = {
            "x": round(float(gx), 2),
            "y": round(float(gy), 2),
            "confidence": round(conf, 3),
            "camera_id": camera_id,
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        }
        projected.append(point_record)

        boxes_info.append({
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
            "confidence": round(conf, 3),
            "class": "person",
            "ground_x": round(float(gx), 2),
            "ground_y": round(float(gy), 2),
        })

        # Render visible bounding box and label on the camera frame
        ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (0, 240, 255), 2)
        # Foot contact point indicator
        cv2.circle(annotated, (int(round(foot_x)), int(round(foot_y))), 4, (0, 255, 128), -1)

        label = f"person {int(round(conf * 100))}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, iy1 - th - 8)
        tag_y2 = iy1
        cv2.rectangle(annotated, (ix1, tag_y1), (ix1 + tw + 8, tag_y2), (16, 35, 45), -1)
        cv2.rectangle(annotated, (ix1, tag_y1), (ix1 + tw + 8, tag_y2), (0, 240, 255), 1)
        cv2.putText(
            annotated,
            label,
            (ix1 + 4, tag_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (229, 255, 114),
            1,
            cv2.LINE_AA,
        )

    return projected, annotated, boxes_info


def ground_points(image: np.ndarray, homography: np.ndarray) -> list[dict[str, float]]:
    """Detect people and map each bbox foot-point to the venue ground plane."""
    pts, _, _ = detect_and_project(image, "C1", homography, settings.detection_conf_thresh)
    return [{"x": p["x"], "y": p["y"], "confidence": p["confidence"]} for p in pts]


def fuse(points: list[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    """Multi-camera aware spatial fusion.
    
    Associates nearby ground-plane detections across different cameras into a single
    unique person entity. Detections from the SAME camera are never merged together
    (a single camera view cannot observe the same physical person twice simultaneously).
    """
    clusters: list[list[dict[str, Any]]] = []
    for point in sorted(points, key=lambda item: item["confidence"], reverse=True):
        best_cluster = None
        min_dist = float("inf")
        for cluster in clusters:
            # Enforce camera exclusivity: same camera cannot have multiple detections in one person cluster
            if any(item.get("camera_id") == point.get("camera_id") for item in cluster):
                continue
            cx = sum(p["x"] for p in cluster) / len(cluster)
            cy = sum(p["y"] for p in cluster) / len(cluster)
            dist = float(np.hypot(point["x"] - cx, point["y"] - cy))
            if dist <= radius and dist < min_dist:
                min_dist = dist
                best_cluster = cluster
        if best_cluster is not None:
            best_cluster.append(point)
        else:
            clusters.append([point])

    fused_people = []
    for idx, c in enumerate(clusters, start=1):
        fused_people.append({
            "id": idx,
            "x": round(float(np.mean([p["x"] for p in c])), 2),
            "y": round(float(np.mean([p["y"] for p in c])), 2),
            "confidence": round(float(np.mean([p["confidence"] for p in c])), 3),
            "views": len(c),
            "cameras": sorted(list(set(p["camera_id"] for p in c if "camera_id" in p))),
        })
    return fused_people


def append_event(event: dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "event_id",
        "location",
        "cameras_analyzed",
        "raw_detections_total",
        "per_camera_counts",
        "unified_count",
        "capacity",
        "occupancy_percentage",
        "average_confidence",
        "risk",
        "condition",
    ]
    write_header = not EVENTS.exists()
    if not write_header:
        try:
            with EVENTS.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
                if "unified_count" not in first_line:
                    write_header = True
        except Exception:
            write_header = True

    mode = "w" if (write_header and EVENTS.exists()) else ("a" if not write_header else "w")
    with EVENTS.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(event)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


@app.get("/")
def dashboard():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "calibrated_cameras": list(calibrations)}


@app.put("/api/settings")
def update_settings(value: Settings):
    global settings
    settings = value
    return settings


@app.post("/api/calibrations")
def set_calibrations(items: list[Calibration]):
    for item in items:
        matrix = np.asarray(item.image_to_ground, dtype=np.float32)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise HTTPException(422, "Each image_to_ground value must be a finite 3x3 matrix")
        calibrations[item.camera_id] = matrix
    return {"saved": list(calibrations)}


@app.post("/api/geometry")
def geometry(first: CameraPose, second: CameraPose):
    """Return epipolar F and both image-to-ground homographies from two calibrated cameras."""
    K1, R1, t1 = np.asarray(first.K, float), np.asarray(first.R, float), np.asarray(first.t, float).reshape(3, 1)
    K2, R2, t2 = np.asarray(second.K, float), np.asarray(second.R, float), np.asarray(second.t, float).reshape(3, 1)
    if any(matrix.shape != (3, 3) for matrix in (K1, R1, K2, R2)):
        raise HTTPException(422, "K and R must each be 3x3")
    # x2.T F x1 = 0 for corresponding pixels. Both poses are world -> camera.
    relative_R = R2 @ R1.T
    relative_t = t2 - relative_R @ t1
    F = np.linalg.inv(K2).T @ skew(relative_t) @ relative_R @ np.linalg.inv(K1)
    F /= max(np.linalg.norm(F), 1e-12)
    homographies = {}
    for item, K, R, t in ((first, K1, R1, t1), (second, K2, R2, t2)):
        ground_to_image = K @ np.column_stack((R[:, 0], R[:, 1], t.reshape(3)))
        homographies[item.camera_id] = np.linalg.inv(ground_to_image).tolist()
    return {"fundamental_matrix": F.tolist(), "image_to_ground_homographies": homographies}


@app.post("/api/analyze")
async def analyze(camera_ids: list[str] = Form(...), frames: list[UploadFile] = File(...)):
    if len(camera_ids) != len(frames):
        raise HTTPException(422, "camera_ids and frames must have the same length")
    all_points, views = [], []
    for camera_id, frame in zip(camera_ids, frames):
        if camera_id not in calibrations:
            raise HTTPException(422, f"No calibration registered for {camera_id}")
        raw = await frame.read()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(422, f"{frame.filename} is not a readable image")
        detections, annotated_image, boxes = detect_and_project(
            image, camera_id, calibrations[camera_id], settings.detection_conf_thresh
        )
        all_points.extend(detections)
        ok, encoded = cv2.imencode(".jpg", annotated_image)
        views.append({
            "camera_id": camera_id,
            "detections": len(detections),
            "boxes": boxes,
            "image": base64.b64encode(encoded).decode() if ok else ""
        })
    people = fuse(all_points, settings.cluster_distance_m)
    count = len(people)
    raw_total = len(all_points)
    average = round(float(np.mean([p["confidence"] for p in people])) if people else 0.0, 3)
    occupancy_pct = round((count / settings.capacity_limit) * 100, 1)
    risk_ratio = round(count / settings.capacity_limit, 3)
    condition = "CAPACITY EXCEEDED" if count > settings.capacity_limit else "SAFE"
    per_cam_summary = {v["camera_id"]: v["detections"] for v in views}

    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_id": str(uuid.uuid4())[:8],
        "location": settings.location,
        "cameras_analyzed": ",".join(sorted(list(set(camera_ids)))),
        "raw_detections_total": raw_total,
        "per_camera_counts": json.dumps(per_cam_summary),
        "unified_count": count,
        "count": count,
        "capacity": settings.capacity_limit,
        "occupancy_percentage": occupancy_pct,
        "average_confidence": average,
        "risk": risk_ratio,
        "condition": condition,
    }
    append_event(event)
    return {
        "event": event,
        "ground_plane_people": people,
        "raw_projected_points": all_points,
        "views": views,
        "stats": {
            "raw_detections_total": raw_total,
            "unified_count": count,
            "multi_camera_fused_count": sum(1 for p in people if p["views"] > 1),
            "single_camera_count": sum(1 for p in people if p["views"] == 1),
            "occupancy_percentage": occupancy_pct,
            "condition": condition,
            "capacity": settings.capacity_limit,
            "average_confidence": average,
        }
    }


@app.get("/api/reports/events.csv")
def download_report():
    if not EVENTS.exists():
        raise HTTPException(404, "No analysis events yet")
    return FileResponse(EVENTS, media_type="text/csv", filename="crowd_events.csv")
