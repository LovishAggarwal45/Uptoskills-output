"""
Traffic Monitoring System Configuration
Configures parameters for detection, tracking, adaptive spatial flow fields,
persistence thresholds, and output directories.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class LaneDefinition:
    lane_id: int
    name: str
    polygon: List[Tuple[int, int]]  # Ordered (x, y) coordinates
    expected_heading_deg: float     # Heading angle in degrees
    allowed_angle_deviation_deg: float = 110.0

@dataclass
class SystemConfig:
    # Model settings
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50
    # COCO classes for vehicles: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # Trajectory & Direction estimation
    trajectory_history_length: int = 40       # Frames to store in trajectory
    direction_window_frames: int = 12         # Window over which to calculate heading vector
    min_displacement_px: float = 6.0          # Minimum px displacement over window to compute heading
    min_total_displacement_px: float = 35.0   # Minimum total net movement to confirm violation

    # Violation Persistence Thresholds
    # A violation must persist for this many frames before triggering an alert
    wrong_way_persistence_frames: int = 15     # 15 frames (~0.5 - 0.75s)
    wrong_way_angle_threshold: float = 125.0   # Angle difference in deg to consider opposite direction (125-180°)
    
    straddle_persistence_frames: int = 25      # 25 frames (~1.0s)
    straddle_overlap_min_ratio: float = 0.20   # Vehicle footprint must overlap at least 20% in second lane

    # Clip recorder settings
    pre_roll_frames: int = 45                  # Frames to buffer before violation confirmation (~1.5s)
    post_roll_frames: int = 45                 # Frames to record after violation confirmation (~1.5s)
    
    # Adaptive Flow Field Grid
    flow_grid_cell_size: int = 80              # Pixels per spatial cell for consensus direction learning
    
    # Output paths
    output_dir: str = "output"
    annotated_video_filename: str = "annotated_traffic.mp4"
    csv_filename: str = "violations.csv"
    clips_dirname: str = "violation_clips"

    def get_output_paths(self) -> Dict[str, str]:
        os.makedirs(self.output_dir, exist_ok=True)
        clips_dir = os.path.join(self.output_dir, self.clips_dirname)
        os.makedirs(clips_dir, exist_ok=True)
        return {
            "output_dir": self.output_dir,
            "annotated_video": os.path.join(self.output_dir, self.annotated_video_filename),
            "csv_path": os.path.join(self.output_dir, self.csv_filename),
            "clips_dir": clips_dir
        }


# Default fallback lanes (divided road layout)
# Used when manual lane polygons are requested; otherwise Adaptive Flow Field is active.
DEFAULT_LANES_720P: List[LaneDefinition] = [
    LaneDefinition(
        lane_id=1,
        name="Upper Roadway",
        polygon=[(0, 0), (1280, 0), (1280, 420), (0, 420)],
        expected_heading_deg=30.0,    # Top flow towards bottom-right (ESE)
        allowed_angle_deviation_deg=110.0,
    ),
    LaneDefinition(
        lane_id=2,
        name="Lower Roadway",
        polygon=[(0, 420), (1280, 420), (1280, 720), (0, 720)],
        expected_heading_deg=190.0,   # Bottom flow towards bottom-left (WSW)
        allowed_angle_deviation_deg=110.0,
    ),
]
