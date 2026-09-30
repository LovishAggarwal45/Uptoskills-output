"""
Event-Level Alert Management System.
Aggregates temporal surveillance events, prevents per-frame alert spam,
and records alerts to persistent log files and standard streams.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class AlertManager:
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        log_file: str = "outputs/alerts/surveillance_alerts.log",
        enable_console: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.enable_console = enable_console
        self.triggered_alerts = []

    def process_events(
        self,
        video_id: str,
        events: List[Dict[str, Any]],
        representative_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates temporal events against the alert threshold.
        De-duplicates and yields event-level alert notifications.
        """
        active_alerts = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for ev in events:
            conf = ev["confidence"]
            if conf >= self.confidence_threshold:
                ev_id = ev["event_id"]
                bbox = representative_bboxes.get(ev_id) if representative_bboxes else None
                
                alert_payload = {
                    "alert_id": f"ALT-{video_id}-{ev_id:03d}",
                    "timestamp": now_str,
                    "video_id": video_id,
                    "condition": ev["action"],
                    "confidence": conf,
                    "peak_confidence": ev.get("peak_confidence", conf),
                    "start_time": ev["start_time"],
                    "end_time": ev["end_time"],
                    "duration_seconds": ev["duration"],
                    "start_time_str": ev["start_time_str"],
                    "end_time_str": ev["end_time_str"],
                    "bbox_norm": bbox,
                    "severity": "CRITICAL" if conf > 0.85 else "WARNING"
                }

                active_alerts.append(alert_payload)
                self.triggered_alerts.append(alert_payload)
                self._dispatch_alert(alert_payload)

        return active_alerts

    def _dispatch_alert(self, alert: Dict[str, Any]):
        """Logs alert to console and persistent alert log file."""
        log_line = (
            f"[{alert['timestamp']}] [ALERT: {alert['severity']}] "
            f"Video: {alert['video_id']} | Action: {alert['condition']} | "
            f"Conf: {alert['confidence']*100:.1f}% | "
            f"Interval: {alert['start_time_str']} -> {alert['end_time_str']} ({alert['duration_seconds']}s)"
        )
        if alert["bbox_norm"]:
            bx, by, bw, bh = alert["bbox_norm"]
            log_line += f" | Box: [{bx:.2f}, {by:.2f}, {bw:.2f}, {bh:.2f}]"

        if self.enable_console:
            print("\n" + "!" * 70)
            print(f"SURVEILLANCE ALERT TRIGGERED:")
            print(f"  {log_line}")
            print("!" * 70 + "\n")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
