"""
Unit tests for Temporal and Spatial Localizers.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.localization.temporal_localizer import TemporalLocalizer
from src.localization.spatial_localizer import SpatialLocalizer

class TestLocalization(unittest.TestCase):
    def test_temporal_localizer(self):
        localizer = TemporalLocalizer(threshold=0.6, min_duration_seconds=1.0, merge_gap_seconds=1.0)
        
        # 10 segments of 1.0s duration each
        timestamps = [(float(i), float(i+1)) for i in range(10)]
        # Anomaly peak at segments 3, 4, 5
        scores = np.array([0.1, 0.1, 0.2, 0.85, 0.90, 0.80, 0.2, 0.1, 0.1, 0.1])
        probs = np.zeros((10, 3))
        probs[:, 0] = 0.9 # Normal default
        probs[3:6, 1] = 0.85 # Fighting

        events = localizer.localize_events(scores, probs, timestamps)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "Fighting")
        self.assertAlmostEqual(events[0]["start_time"], 3.0, delta=1.0)
        self.assertAlmostEqual(events[0]["end_time"], 6.0, delta=1.0)

    def test_spatial_localizer(self):
        localizer = SpatialLocalizer(min_box_area_ratio=0.001, threshold_ratio=0.2)
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        # Add high motion in center
        flow[30:70, 30:70, 0] = 10.0
        flow[30:70, 30:70, 1] = 10.0

        boxes = localizer.extract_motion_rois(flow, frame_shape=(200, 200))
        self.assertGreaterEqual(len(boxes), 1)
        bx, by, bw, bh = boxes[0]["bbox_pixel"]
        self.assertGreater(bw, 0)
        self.assertGreater(bh, 0)

if __name__ == "__main__":
    unittest.main()
