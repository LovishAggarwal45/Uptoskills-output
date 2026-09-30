#!/usr/bin/env python3
"""
CLI tool for computing and visualizing optical flow from video sequences.
Demonstrates motion direction, magnitude, and motion energy.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.video_reader import VideoReader
from src.preprocessing.optical_flow import OpticalFlowEngine

def compute_and_visualize_flow(video_path, output_dir, max_frames=64):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(str(video_path), target_fps=16, target_size=(224, 224))
    flow_engine = OpticalFlowEngine(target_size=(224, 224))

    print(f"Computing optical flow for: {video_path.name}")
    frames = []
    timestamps = []

    for f_idx, ts, rgb in reader.stream_frames():
        frames.append(rgb)
        timestamps.append(ts)
        if len(frames) >= max_frames:
            break

    if len(frames) < 2:
        print("Error: Need at least 2 frames to compute optical flow.")
        return

    frames_arr = np.stack(frames, axis=0) # [T, H, W, 3]
    flow_res = flow_engine.compute_clip_flow(frames_arr)
    
    # Save a side-by-side visualization video of RGB + Optical Flow
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vis_path = output_dir / f"{video_path.stem}_flow_vis.mp4"
    h, w = frames[0].shape[0], frames[0].shape[1]
    out = cv2.VideoWriter(str(vis_path), fourcc, 16, (w * 2, h))

    for t in range(len(frames)):
        rgb_vis = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        # Reconstruct [H, W, 2] flow
        fl_hw2 = np.transpose(flow_res["flow"][t], (1, 2, 0))
        flow_rgb = flow_engine.flow_to_rgb(fl_hw2)
        flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)

        # Annotate
        energy = flow_res["mean_energy"][t]
        cv2.putText(rgb_vis, f"RGB [{timestamps[t]:.2f}s]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(flow_bgr, f"Flow (Energy: {energy:.2f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([rgb_vis, flow_bgr])
        out.write(combined)

    out.release()
    print(f"Optical flow visualization saved to: {vis_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute and visualize optical flow.")
    parser.add_argument("--video", type=str, required=True, help="Input video.")
    parser.add_argument("--output", type=str, default="outputs/visualizations", help="Output directory.")
    parser.add_argument("--max-frames", type=int, default=80, help="Max frames to process.")
    args = parser.parse_args()

    compute_and_visualize_flow(args.video, args.output, args.max_frames)

if __name__ == "__main__":
    main()
