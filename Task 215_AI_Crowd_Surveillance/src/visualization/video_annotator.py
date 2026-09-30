"""
Video Annotation Engine for CCTV Surveillance.
Overlays bounding boxes, action status badges, confidence meters,
live timestamps, and temporal alert indicators onto surveillance video.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

class VideoAnnotator:
    def __init__(
        self,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
        font=cv2.FONT_HERSHEY_SIMPLEX
    ):
        self.color_map = color_map or {
            "Normal": (0, 200, 0),         # Green (BGR)
            "Fighting": (0, 0, 230),       # Red (BGR)
            "Panic_Dispersal": (0, 140, 255) # Orange (BGR)
        }
        self.font = font

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        current_action: str,
        confidence: float,
        is_alert: bool,
        active_event: Optional[Dict[str, Any]] = None,
        bboxes: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Draws professional surveillance HUD overlay onto a video frame.
        """
        vis = frame_bgr.copy()
        H, W = vis.shape[:2]

        # 1. Top HUD Header Bar
        cv2.rectangle(vis, (0, 0), (W, 40), (20, 20, 20), -1)
        
        # Timecode
        mins = int(timestamp // 60)
        secs = timestamp % 60
        ts_str = f"CAM-01 [LIVE] {mins:02d}:{secs:05.2f}"
        cv2.putText(vis, ts_str, (10, 26), self.font, 0.55, (220, 220, 220), 1)

        # Action Badge & Alert Indicator
        badge_color = self.color_map.get(current_action, (0, 200, 0))
        if is_alert:
            # Flashing or bold red alert badge
            status_text = f"! ALERT: {current_action.upper()} ({confidence*100:.1f}%) !"
            cv2.rectangle(vis, (W - 320, 5), (W - 10, 35), (0, 0, 200), -1)
            cv2.putText(vis, status_text, (W - 310, 26), self.font, 0.52, (255, 255, 255), 2)
            # Red border around the entire CCTV frame to signal active alert
            cv2.rectangle(vis, (0, 0), (W-1, H-1), (0, 0, 220), 3)
        else:
            status_text = f"Status: {current_action} ({confidence*100:.1f}%)"
            cv2.rectangle(vis, (W - 260, 5), (W - 10, 35), (40, 40, 40), -1)
            cv2.putText(vis, status_text, (W - 250, 25), self.font, 0.5, (0, 255, 0), 1)

        # 2. Draw Bounding Boxes if anomalous
        if is_alert and bboxes:
            for b in bboxes:
                bx, by, bw, bh = b["bbox_pixel"]
                # Box rectangle
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 240), 2)
                # Label badge with pseudo-localization disclaimer
                label_tag = f"{current_action} [Motion ROI]"
                cv2.rectangle(vis, (bx, max(0, by - 20)), (bx + len(label_tag)*9, by), (0, 0, 240), -1)
                cv2.putText(vis, label_tag, (bx + 3, max(14, by - 5)), self.font, 0.38, (255, 255, 255), 1)

        # 3. Active Event Interval Details
        if active_event:
            ev_info = f"Event: {active_event['start_time_str']} -> {active_event['end_time_str']} (Peak Conf: {active_event.get('peak_confidence', confidence)*100:.1f}%)"
            cv2.rectangle(vis, (10, H - 32), (W - 10, H - 8), (20, 20, 20), -1)
            cv2.putText(vis, ev_info, (15, H - 15), self.font, 0.45, (0, 220, 255), 1)

        return vis
