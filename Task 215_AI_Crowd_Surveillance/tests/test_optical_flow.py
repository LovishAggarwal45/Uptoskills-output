"""
Unit tests for OpticalFlowEngine.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.optical_flow import OpticalFlowEngine

class TestOpticalFlow(unittest.TestCase):
    def setUp(self):
        self.engine = OpticalFlowEngine(target_size=(64, 64))

    def test_flow_pair(self):
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        # Shift a white square by 5 pixels horizontally
        frame1[20:50, 20:50] = 255
        frame2[20:50, 25:55] = 255

        flow = self.engine.compute_flow_pair(frame1, frame2)
        self.assertEqual(flow.shape, (64, 64, 2))
        
        # Test color wheel visualization conversion
        vis_rgb = self.engine.flow_to_rgb(flow)
        self.assertEqual(vis_rgb.shape, (64, 64, 3))
        self.assertEqual(vis_rgb.dtype, np.uint8)

    def test_clip_flow(self):
        frames = np.random.randint(0, 255, (4, 100, 100, 3), dtype=np.uint8)
        res = self.engine.compute_clip_flow(frames)
        self.assertIn("flow", res)
        self.assertIn("magnitude", res)
        self.assertIn("mean_energy", res)
        self.assertEqual(res["flow"].shape, (4, 2, 64, 64))
        self.assertEqual(len(res["mean_energy"]), 4)

if __name__ == "__main__":
    unittest.main()
