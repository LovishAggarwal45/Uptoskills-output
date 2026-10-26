"""
Visualizer & HUD Overlay Module
Renders lane polygons, traffic-flow arrows, vehicle bounding boxes,
trajectory motion trails, velocity vectors, and real-time alert banners.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from config import LaneDefinition
from detector_tracker import TrackedVehicle
from lane_manager import LaneManager
from trajectory_estimator import TrajectoryEstimator
from violation_detector import ViolationDetector

class Visualizer:
    # Color palette (BGR)
    COLOR_LANE_NORMAL = (220, 220, 220)
    COLOR_LANE_FILL = (60, 60, 60)
    COLOR_BOX_NORMAL = (0, 220, 0)         # Green
    COLOR_BOX_WARNING = (0, 165, 255)       # Orange (Straddling)
    COLOR_BOX_DANGER = (0, 0, 255)          # Red (Wrong Way)
    COLOR_TRAIL = (255, 200, 0)             # Cyan/Yellow
    COLOR_VECTOR = (0, 255, 255)            # Yellow arrow

    def __init__(self, lane_manager: LaneManager):
        self.lane_manager = lane_manager

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draws semi-transparent lane overlays and flow arrows."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        for lane_id, lane in self.lane_manager.lanes.items():
            pts = np.array(lane.polygon, np.int32).reshape((-1, 1, 2))
            
            # Semi-transparent fill
            cv2.fillPoly(overlay, [pts], (40, 40, 50))
            
            # Boundary lines
            cv2.polylines(frame, [pts], isClosed=True, color=(120, 120, 120), thickness=2)

            # Draw lane label & flow arrow at center of polygon
            poly_np = np.array(lane.polygon)
            center_x = int(np.mean(poly_np[:, 0]))
            center_y = int(np.mean(poly_np[:, 1]))

            flow_deg = self.lane_manager.get_flow_heading(lane_id)
            if flow_deg is not None:
                rad = math.radians(flow_deg)
                arrow_len = 35
                arrow_tip = (
                    int(center_x + arrow_len * math.cos(rad)),
                    int(center_y + arrow_len * math.sin(rad))
                )
                arrow_tail = (
                    int(center_x - arrow_len * math.cos(rad)),
                    int(center_y - arrow_len * math.sin(rad))
                )
                cv2.arrowedLine(frame, arrow_tail, arrow_tip, (200, 200, 200), 2, tipLength=0.35)

                # Label
                label = f"{lane.name} [{int(flow_deg)}deg]"
                cv2.putText(
                    frame, label, (center_x - 70, max(30, center_y - 45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
                )

        # Draw Adaptive Spatial Flow Field vectors if available
        if hasattr(self.lane_manager, "flow_field"):
            ff = self.lane_manager.flow_field
            for (r, c), (angle_deg, conf) in ff.grid_consensus.items():
                if conf >= 0.4:
                    cx = int((c + 0.5) * ff.cell_size)
                    cy = int((r + 0.5) * ff.cell_size)
                    rad = math.radians(angle_deg)
                    arrow_len = 16
                    tip = (int(cx + arrow_len * math.cos(rad)), int(cy + arrow_len * math.sin(rad)))
                    tail = (int(cx - arrow_len * math.cos(rad)), int(cy - arrow_len * math.sin(rad)))
                    cv2.arrowedLine(frame, tail, tip, (0, 220, 120), 1, tipLength=0.35)

        # Blend overlay
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame

    def draw_vehicle(
        self,
        frame: np.ndarray,
        vehicle: TrackedVehicle,
        lane_name: str,
        heading_deg: Optional[float],
        is_moving: bool,
        is_wrong_way: bool,
        is_straddling: bool,
        trail: List[Tuple[int, int]]
    ):
        """Draws bounding box, label, trajectory trail, and heading vector for a vehicle."""
        x1, y1, x2, y2 = [int(v) for v in vehicle.bbox]
        cx, cy = int(vehicle.centroid[0]), int(vehicle.centroid[1])

        # Determine color based on violation severity
        if is_wrong_way:
            box_color = self.COLOR_BOX_DANGER
            status_text = "WRONG-WAY!"
        elif is_straddling:
            box_color = self.COLOR_BOX_WARNING
            status_text = "STRADDLING"
        else:
            box_color = self.COLOR_BOX_NORMAL
            status_text = "OK"

        # Bounding box
        thickness = 3 if (is_wrong_way or is_straddling) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Draw Trajectory Trail
        if len(trail) > 1:
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                color = (
                    int(self.COLOR_TRAIL[0] * alpha),
                    int(self.COLOR_TRAIL[1] * alpha),
                    int(self.COLOR_TRAIL[2] * alpha)
                )
                cv2.line(frame, trail[i - 1], trail[i], color, 2)

        # Draw Heading Arrow
        if heading_deg is not None and is_moving:
            rad = math.radians(heading_deg)
            arrow_len = 30
            tip_x = int(cx + arrow_len * math.cos(rad))
            tip_y = int(cy + arrow_len * math.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y), self.COLOR_VECTOR, 2, tipLength=0.3)

        # Header tag above box
        heading_str = f"{int(heading_deg)}deg" if heading_deg is not None else "N/A"
        tag = f"#{vehicle.track_id} {vehicle.class_name} ({vehicle.confidence:.2f}) | {status_text}"
        sub_tag = f"Lane: {lane_name} | Dir: {heading_str}"

        # Background badge for tag
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(th + 5, y1 - 8)
        cv2.rectangle(frame, (x1, tag_y - th - 5), (x1 + tw + 6, tag_y + 4), box_color, -1)
        cv2.putText(frame, tag, (x1 + 3, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Sub-badge
        (stw, sth), _ = cv2.getTextSize(sub_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (x1, tag_y + 6), (x1 + stw + 6, tag_y + sth + 10), (30, 30, 30), -1)
        cv2.putText(frame, sub_tag, (x1 + 3, tag_y + sth + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        active_vehicles_count: int,
        wrong_way_count: int,
        straddle_count: int,
        active_alerts: List[str]
    ):
        """Draws flashing alert banner only when an active violation is present (heading title removed)."""
        h, w = frame.shape[:2]

        # Active Alert Banners (Flashing effect) - displayed only when violation occurs
        if active_alerts:
            is_flash_on = (frame_idx // 8) % 2 == 0
            if is_flash_on:
                alert_y = 18
                for alert in active_alerts:
                    bg_color = (0, 0, 220) if "WRONG" in alert else (0, 140, 255)
                    (atw, ath), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    alert_box_w = atw + 30
                    alert_box_x = (w - alert_box_w) // 2
                    
                    cv2.rectangle(
                        frame,
                        (alert_box_x, alert_y),
                        (alert_box_x + alert_box_w, alert_y + ath + 16),
                        bg_color,
                        -1
                    )
                    cv2.rectangle(
                        frame,
                        (alert_box_x, alert_y),
                        (alert_box_x + alert_box_w, alert_y + ath + 16),
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"[!] {alert}",
                        (alert_box_x + 15, alert_y + ath + 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    alert_y += ath + 24
