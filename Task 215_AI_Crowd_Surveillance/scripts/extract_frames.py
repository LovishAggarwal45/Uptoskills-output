#!/usr/bin/env python3
"""
CLI tool for extracting and inspecting sampled video frames.
Saves frames to disk with timestamps and metadata for visual verification.
"""

import os
import sys
import argparse
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.video_reader import VideoReader

def extract_frames(video_path, output_dir, target_fps=16, target_size=(224, 224), max_frames=None):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(str(video_path), target_fps=target_fps, target_size=target_size)
    meta = reader.get_metadata()
    print(f"Extracting frames from: {video_path.name}")
    print(f"Video metadata: {meta['width']}x{meta['height']}, {meta['native_fps']:.1f} FPS, {meta['duration_seconds']:.2f}s")
    print(f"Target sampling: {target_fps} FPS, resized to {target_size}")

    count = 0
    for f_idx, ts, rgb_frame in reader.stream_frames():
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        fname = f"frame_{f_idx:05d}_ts{ts:06.2f}s.jpg"
        cv2.imwrite(str(output_dir / fname), bgr_frame)
        count += 1
        if max_frames and count >= max_frames:
            break

    print(f"Extracted {count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract sampled frames from video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output", type=str, default="outputs/extracted_frames", help="Output directory.")
    parser.add_argument("--fps", type=int, default=16, help="Target sampling FPS.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to extract.")
    args = parser.parse_args()

    extract_frames(args.video, args.output, target_fps=args.fps, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
