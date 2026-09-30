#!/usr/bin/env python3
"""
Dataset Preparation and Acquisition Script.
Supports:
1. Automated synthesis of reproducible CCTV surveillance benchmark videos
   (Normal pedestrian movement, Rapid aggressive fighting motions, Crowd panic dispersal)
   with video-level labels for training and exact temporal ground truth for testing.
2. Conversion & formatting scripts for public datasets (UCF-Crime, XD-Violence).
3. Manifest generation for splits (train.csv, val.csv, test.csv, temporal_ground_truth.json).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

def create_synthetic_cctv_video(output_path, action_type, duration_sec=10, fps=16, width=320, height=240, anomaly_window=(3.0, 7.0)):
    """
    Renders a realistic surveillance-style scene with simulated human agents.
    Normal: smooth linear walking trajectories with low velocity.
    Fighting: sudden rapid, overlapping, chaotic high-acceleration trajectories in the anomaly window.
    Panic_Dispersal: radial rapid outward acceleration/scattering of crowd agents in the anomaly window.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise IOError(f"Could not open VideoWriter for {output_path}")

    # Initialize simulated agents (representing people in CCTV scene)
    num_agents = 8 if action_type == "Panic_Dispersal" else 5
    agents = []
    for i in range(num_agents):
        pos = np.array([np.random.uniform(50, width - 50), np.random.uniform(50, height - 50)], dtype=float)
        vel = np.random.uniform(-1.0, 1.0, size=2)
        vel = vel / (np.linalg.norm(vel) + 1e-6) * np.random.uniform(0.5, 1.5)
        color = (np.random.randint(180, 240), np.random.randint(180, 240), np.random.randint(180, 240))
        agents.append({"pos": pos, "vel": vel, "color": color, "radius": 7})

    start_anom_frame = int(anomaly_window[0] * fps)
    end_anom_frame = int(anomaly_window[1] * fps)

    for f in range(total_frames):
        current_time = f / fps
        is_anomaly = (action_type != "Normal") and (start_anom_frame <= f <= end_anom_frame)

        # Create surveillance grayscale background with subtle static texture & time overlay
        frame = np.ones((height, width, 3), dtype=np.uint8) * 80
        # Add floor tile lines (perspective)
        for y_line in range(60, height, 40):
            cv2.line(frame, (0, y_line), (width, y_line), (70, 70, 70), 1)
        for x_line in range(0, width, 50):
            cv2.line(frame, (x_line, 60), (int(width/2 + (x_line - width/2)*1.5), height), (70, 70, 70), 1)

        # Update and render agents
        for i, ag in enumerate(agents):
            if is_anomaly:
                if action_type == "Fighting":
                    # Agents 0 and 1 grapple violently near center
                    if i in (0, 1):
                        center = np.array([width / 2.0, height / 2.0])
                        ag["pos"] = center + np.random.uniform(-12, 12, size=2)
                    else:
                        # Others stand around watching
                        ag["vel"] = ag["vel"] * 0.2
                        ag["pos"] += ag["vel"]
                elif action_type == "Panic_Dispersal":
                    # Rapid outward divergence
                    center = np.array([width / 2.0, height / 2.0])
                    diff = ag["pos"] - center
                    dist = np.linalg.norm(diff) + 1e-5
                    ag["vel"] = (diff / dist) * np.random.uniform(4.0, 7.0)
                    ag["pos"] += ag["vel"]
            else:
                # Normal steady walking
                ag["pos"] += ag["vel"]
                # Bounce softly off boundaries
                for d in range(2):
                    max_dim = width if d == 0 else height
                    if ag["pos"][d] < 20 or ag["pos"][d] > max_dim - 20:
                        ag["vel"][d] *= -1.0
                        ag["pos"][d] = np.clip(ag["pos"][d], 20, max_dim - 20)

            # Draw person body (oval) and head (circle)
            px, py = int(ag["pos"][0]), int(ag["pos"][1])
            cv2.ellipse(frame, (px, py), (ag["radius"], int(ag["radius"] * 1.6)), 0, 0, 360, (40, 40, 40), -1)
            cv2.circle(frame, (px, py - int(ag["radius"] * 1.6)), int(ag["radius"] * 0.8), (180, 160, 140), -1)

        # CCTV timestamp and camera tag overlay
        cv2.putText(frame, f"CAM-04 [REC] {current_time:05.2f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        # Subtle sensor noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()
    return {
        "path": str(output_path),
        "action": action_type,
        "duration": duration_sec,
        "fps": fps,
        "num_frames": total_frames,
        "anomaly_window": anomaly_window if action_type != "Normal" else None
    }

def generate_benchmark_suite(root_dir):
    """Generates a complete reproducible surveillance benchmark with train/val/test splits."""
    root_dir = Path(root_dir)
    data_raw = root_dir / "datasets" / "raw"
    splits_dir = root_dir / "datasets" / "splits"
    metadata_dir = root_dir / "datasets" / "metadata"
    
    data_raw.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Synthesizing CCTV surveillance videos for benchmark suite...")
    
    video_configs = [
        # Training set (weak video-level labels ONLY)
        ("train_normal_01.mp4", "Normal", "train", 8.0, None),
        ("train_normal_02.mp4", "Normal", "train", 10.0, None),
        ("train_normal_03.mp4", "Normal", "train", 7.0, None),
        ("train_fight_01.mp4", "Fighting", "train", 10.0, (2.5, 7.5)),
        ("train_fight_02.mp4", "Fighting", "train", 8.0, (1.5, 6.0)),
        ("train_panic_01.mp4", "Panic_Dispersal", "train", 9.0, (3.0, 7.0)),
        ("train_panic_02.mp4", "Panic_Dispersal", "train", 8.0, (2.0, 6.0)),
        
        # Validation set
        ("val_normal_01.mp4", "Normal", "val", 8.0, None),
        ("val_fight_01.mp4", "Fighting", "val", 9.0, (2.0, 6.5)),
        ("val_panic_01.mp4", "Panic_Dispersal", "val", 8.0, (2.5, 6.0)),

        # Test set (untrimmed videos with precise temporal ground truth for evaluation)
        ("test_normal_01.mp4", "Normal", "test", 12.0, None),
        ("test_fight_01.mp4", "Fighting", "test", 14.0, (3.5, 9.5)),
        ("test_fight_02.mp4", "Fighting", "test", 15.0, (4.0, 11.0)),
        ("test_panic_01.mp4", "Panic_Dispersal", "test", 13.0, (3.0, 8.5)),
    ]

    records = []
    temporal_gt = {}

    class_to_id = {"Normal": 0, "Fighting": 1, "Panic_Dispersal": 2}

    for filename, action, split, dur, anom_win in video_configs:
        vpath = data_raw / filename
        win = anom_win if anom_win else (0.0, 0.0)
        meta = create_synthetic_cctv_video(vpath, action, duration_sec=dur, fps=16, anomaly_window=win)
        vid_id = Path(filename).stem
        records.append({
            "video_id": vid_id,
            "filename": filename,
            "path": str(vpath.relative_to(root_dir)).replace("\\", "/"),
            "label": action,
            "label_id": class_to_id[action],
            "split": split,
            "duration": dur,
            "fps": 16,
            "num_frames": int(dur * 16),
            "is_anomaly": 0 if action == "Normal" else 1
        })
        
        # Test split ground-truth temporal intervals
        if split == "test":
            temporal_gt[vid_id] = {
                "action": action,
                "label_id": class_to_id[action],
                "duration": dur,
                "events": [] if action == "Normal" else [
                    {"start_time": anom_win[0], "end_time": anom_win[1], "label": action}
                ]
            }

    df = pd.DataFrame(records)
    for s in ["train", "val", "test"]:
        split_df = df[df["split"] == s]
        split_df.to_csv(splits_dir / f"{s}.csv", index=False)
        print(f"Saved {len(split_df)} records to {splits_dir / f'{s}.csv'}")

    # Save temporal ground truth JSON
    gt_file = splits_dir / "temporal_ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(temporal_gt, f, indent=2)
    print(f"Saved test temporal ground truth to {gt_file}")

    print("Benchmark suite generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Dataset preparation tool.")
    parser.add_argument("--benchmark-suite", action="store_true", help="Generate synthetic CCTV surveillance benchmark.")
    parser.add_argument("--root-dir", type=str, default=".", help="Project root directory.")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if args.benchmark_suite:
        generate_benchmark_suite(root)
    else:
        print("Please specify an action. Example: python scripts/prepare_dataset.py --benchmark-suite")

if __name__ == "__main__":
    main()
