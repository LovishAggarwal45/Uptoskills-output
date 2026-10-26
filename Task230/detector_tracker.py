"""
Vehicle Detection & Tracking Module
Leverages YOLOv8 with persistent tracking (ByteTrack) to detect and track
vehicles across video frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

@dataclass
class TrackedVehicle:
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    centroid: Tuple[float, float]            # cx, cy
    footprint: Tuple[float, float]           # cx, y2 (ground contact)
    confidence: float
    class_id: int
    class_name: str

class VehicleDetectorTracker:
    # COCO Class mapping for vehicles
    VEHICLE_CLASS_MAP = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        target_classes: Optional[List[int]] = None
    ):
        """
        Initializes the YOLOv8 vehicle detector and tracker.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or [2, 3, 5, 7]
        self._next_fallback_id = 1
        self._previous_tracks: Dict[int, Tuple[float, float]] = {}

    def process_frame(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Processes a single video frame, performing vehicle detection and tracking.
        :param frame: BGR image from OpenCV (H, W, 3)
        :return: List of TrackedVehicle objects
        """
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False
        )

        tracked_vehicles: List[TrackedVehicle] = []
        if not results or len(results) == 0:
            return tracked_vehicles

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return tracked_vehicles

        for box in boxes:
            # Coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            
            # Confidence & Class
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.5
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 2
            cls_name = self.VEHICLE_CLASS_MAP.get(cls_id, result.names.get(cls_id, f"class_{cls_id}"))

            # Track ID
            if box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
            else:
                # Fallback simple nearest centroid matching if tracker did not assign id
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track_id = self._assign_fallback_id(cx, cy)

            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            footprint = ((x1 + x2) / 2.0, y2)

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    centroid=centroid,
                    footprint=footprint,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
            )

        return tracked_vehicles

    def _assign_fallback_id(self, cx: float, cy: float, max_distance: float = 60.0) -> int:
        best_id = None
        best_dist = float("inf")
        for tid, (px, py) in list(self._previous_tracks.items()):
            dist = np.hypot(cx - px, cy - py)
            if dist < best_dist and dist < max_distance:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = self._next_fallback_id
            self._next_fallback_id += 1

        self._previous_tracks[best_id] = (cx, cy)
        # Purge excess fallback entries
        if len(self._previous_tracks) > 100:
            self._previous_tracks.clear()

        return best_id
