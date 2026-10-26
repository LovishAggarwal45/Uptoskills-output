"""
CSV Logging Module
Records violations in a structured CSV containing vehicle type,
lane, timestamp, confidence, and violation metadata.
"""

import csv
import os
from typing import List
from violation_detector import ViolationEvent

class CSVLogger:
    FIELDNAMES = [
        "violation_id",
        "vehicle_id",
        "vehicle_type",
        "violation_type",
        "lane",
        "timestamp",
        "confidence",
        "heading_deg",
        "deviation_deg",
        "duration_seconds",
        "clip_path"
    ]

    def __init__(self, output_csv_path: str):
        self.output_csv_path = output_csv_path
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        
        # Initialize file with header if not exists
        if not os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def log_violation(self, event: ViolationEvent):
        """Appends a completed or triggered violation event to the CSV log."""
        with open(self.output_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({
                "violation_id": event.violation_id,
                "vehicle_id": event.vehicle_id,
                "vehicle_type": event.vehicle_type,
                "violation_type": event.violation_type.value,
                "lane": event.lane_name or (f"Lane {event.lane_id}" if event.lane_id else "Unknown"),
                "timestamp": event.start_timestamp,
                "confidence": f"{event.confidence:.2f}",
                "heading_deg": f"{event.heading_deg:.1f}",
                "deviation_deg": f"{event.deviation_deg:.1f}",
                "duration_seconds": f"{event.duration_seconds:.2f}",
                "clip_path": event.clip_path
            })

    def log_all(self, events: List[ViolationEvent]):
        """Logs a list of violation events."""
        for ev in events:
            self.log_violation(ev)
