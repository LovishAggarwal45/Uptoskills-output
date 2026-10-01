"""
CSV Event Report Generator.
Formats and writes surveillance detection events into standardized CSV format
with complete spatial and temporal audit metadata.
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

REPORT_COLUMNS = [
    "video_id",
    "timestamp",
    "start_time",
    "end_time",
    "location",
    "condition",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height"
]

class ReportGenerator:
    def __init__(self, output_csv: str = "outputs/reports/event_report.csv"):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not self.output_csv.exists() or self.output_csv.stat().st_size == 0:
            with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(REPORT_COLUMNS)

    def write_events(
        self,
        video_id: str,
        events: List[Dict[str, Any]],
        event_bboxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        location: str = "CAM_SURVEILLANCE_MAIN"
    ):
        """
        Appends event records to the CSV report.
        """
        rows = []
        for ev in events:
            ev_id = ev.get("event_id", 0)
            bbox = event_bboxes.get(ev_id) if event_bboxes else None
            
            bx = f"{bbox[0]:.4f}" if bbox else ""
            by = f"{bbox[1]:.4f}" if bbox else ""
            bw = f"{bbox[2]:.4f}" if bbox else ""
            bh = f"{bbox[3]:.4f}" if bbox else ""

            row = [
                video_id,
                ev.get("start_time_str", ""),
                f"{ev['start_time']:.2f}",
                f"{ev['end_time']:.2f}",
                location,
                ev["action"],
                f"{ev['confidence']:.4f}",
                bx,
                by,
                bw,
                bh
            ]
            rows.append(row)

        if rows:
            with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
