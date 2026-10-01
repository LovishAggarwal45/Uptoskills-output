"""
Unit tests for VideoReader and frame sampling.
"""

import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.video_reader import VideoReader

class TestVideoReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = Path(self.temp_dir.name) / "test_video.mp4"
        
        # Create a tiny 1-second 16fps dummy video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.video_path), fourcc, 16, (160, 120))
        for _ in range(16):
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_probe(self):
        reader = VideoReader(str(self.video_path), target_fps=16, target_size=(224, 224))
        meta = reader.get_metadata()
        self.assertEqual(meta["native_frame_count"], 16)
        self.assertAlmostEqual(meta["duration_seconds"], 1.0, delta=0.1)

    def test_stream_clips(self):
        reader = VideoReader(str(self.video_path), target_fps=16, target_size=(224, 224))
        clips = list(reader.stream_clips(clip_length=16, stride=16))
        self.assertGreaterEqual(len(clips), 1)
        first_clip = clips[0]
        self.assertEqual(first_clip["frames"].shape, (16, 224, 224, 3))

if __name__ == "__main__":
    unittest.main()
