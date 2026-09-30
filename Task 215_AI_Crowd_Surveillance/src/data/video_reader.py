"""
Stream-based video reader for long CCTV videos.
Handles FPS normalization, temporal clip generation, and memory-efficient streaming.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any, Tuple, Optional

class VideoReader:
    def __init__(self, video_path: str, target_fps: int = 16, target_size: Tuple[int, int] = (224, 224)):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.target_fps = target_fps
        self.target_size = target_size
        self._probe_video()

    def _probe_video(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
        
        self.native_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        self.native_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.native_frame_count / self.native_fps if self.native_fps > 0 else 0.0
        cap.release()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "path": str(self.video_path),
            "video_id": self.video_path.stem,
            "native_fps": self.native_fps,
            "native_frame_count": self.native_frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration,
            "target_fps": self.target_fps,
            "target_size": self.target_size
        }

    def stream_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Streams frames re-sampled to target_fps.
        Yields: (frame_idx, timestamp_seconds, rgb_frame)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        step = self.native_fps / self.target_fps if self.target_fps > 0 else 1.0
        current_target_frame = 0
        
        frame_idx = 0
        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                break
            
            # Check if this frame should be sampled
            if frame_idx >= current_target_frame * step:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if self.target_size is not None and (bgr_frame.shape[1], bgr_frame.shape[0]) != self.target_size:
                    rgb_frame = cv2.resize(rgb_frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                
                timestamp = current_target_frame / self.target_fps
                yield current_target_frame, timestamp, rgb_frame
                current_target_frame += 1

            frame_idx += 1

        cap.release()

    def stream_clips(self, clip_length: int = 16, stride: int = 16) -> Generator[Dict[str, Any], None, None]:
        """
        Streams temporal clips of length `clip_length` with specified `stride`.
        Yields a dict with:
          - 'frames': np.ndarray of shape [clip_length, H, W, 3] (RGB uint8)
          - 'start_time': float seconds
          - 'end_time': float seconds
          - 'clip_idx': int
        """
        buffer = []
        timestamps = []
        clip_idx = 0

        for f_idx, ts, frame in self.stream_frames():
            buffer.append(frame)
            timestamps.append(ts)

            if len(buffer) == clip_length:
                clip_arr = np.stack(buffer, axis=0) # [T, H, W, 3]
                start_t = timestamps[0]
                end_t = timestamps[-1]
                
                yield {
                    "clip_idx": clip_idx,
                    "frames": clip_arr,
                    "start_time": start_t,
                    "end_time": end_t,
                    "duration": end_t - start_t
                }
                clip_idx += 1

                # Advance buffer by stride
                if stride < clip_length:
                    buffer = buffer[stride:]
                    timestamps = timestamps[stride:]
                else:
                    buffer = []
                    timestamps = []

        # If remaining frames exist and buffer is at least half clip_length, pad to clip_length
        if len(buffer) >= max(4, clip_length // 2):
            while len(buffer) < clip_length:
                buffer.append(buffer[-1])
                timestamps.append(timestamps[-1] + (1.0 / self.target_fps))
            clip_arr = np.stack(buffer, axis=0)
            yield {
                "clip_idx": clip_idx,
                "frames": clip_arr,
                "start_time": timestamps[0],
                "end_time": timestamps[-1],
                "duration": timestamps[-1] - timestamps[0]
            }
