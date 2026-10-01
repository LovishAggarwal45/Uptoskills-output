#!/usr/bin/env python3
"""
Dataset Inspector and Integrity Auditor.
Scans dataset splits, verifies video file headers, counts frames, validates
FPS and resolutions, checks class balance, and summarizes temporal ground truth.
"""

import os
import sys
import json
import argparse
import pandas as pd
import cv2
from pathlib import Path

def inspect_dataset(splits_dir, root_dir="."):
    splits_dir = Path(splits_dir)
    root_dir = Path(root_dir).resolve()
    
    print("=" * 65)
    print("AI CROWD SURVEILLANCE SYSTEM - DATASET AUDIT REPORT")
    print("=" * 65)

    if not splits_dir.exists():
        print(f"Error: Splits directory does not exist: {splits_dir}")
        return

    split_files = ["train.csv", "val.csv", "test.csv"]
    total_videos = 0

    for s_file in split_files:
        csv_path = splits_dir / s_file
        if not csv_path.exists():
            print(f"Split file missing: {s_file}")
            continue

        df = pd.read_csv(csv_path)
        split_name = s_file.replace(".csv", "").upper()
        print(f"\n[{split_name} SPLIT] ({len(df)} videos):")
        print(f"  * Class Breakdown:")
        counts = df["label"].value_counts()
        for cls, count in counts.items():
            print(f"      - {cls:<18}: {count} ({count/len(df)*100:.1f}%)")

        # Verify readability of first 3 videos per split
        readable_count = 0
        total_duration = 0.0
        for idx, row in df.iterrows():
            vpath = root_dir / row["path"]
            if not vpath.exists():
                print(f"  [!] Missing file: {vpath}")
                continue

            cap = cv2.VideoCapture(str(vpath))
            if cap.isOpened():
                readable_count += 1
                fps = cap.get(cv2.CAP_PROP_FPS) or row.get("fps", 16)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or row.get("num_frames", 0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur = frames / fps if fps > 0 else 0
                total_duration += dur
                if idx == 0:
                    print(f"  * Sample Probe ({row['video_id']}): {w}x{h} @ {fps:.1f} FPS, {dur:.1f}s ({int(frames)} frames)")
                cap.release()
            else:
                print(f"  [!] Corrupt or unreadable video: {vpath}")

        print(f"  * Readability Integrity: {readable_count}/{len(df)} verified readable")
        print(f"  * Total Split Duration: {total_duration:.1f} seconds (~{total_duration/60:.2f} minutes)")
        total_videos += len(df)

    print("-" * 65)
    print(f"TOTAL VIDEOS ACROSS SPLITS: {total_videos}")

    # Check temporal ground truth
    gt_path = splits_dir / "temporal_ground_truth.json"
    if gt_path.exists():
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        print(f"\n[TEMPORAL GROUND TRUTH AUDIT]")
        print(f"  * Ground truth records found for {len(gt_data)} test videos")
        for vid, info in gt_data.items():
            events = info.get("events", [])
            ev_str = ", ".join([f"{e['label']} [{e['start_time']:.1f}s - {e['end_time']:.1f}s]" for e in events]) if events else "Normal (No anomalies)"
            print(f"    - {vid:<16} ({info['duration']}s): {ev_str}")
    else:
        print("\n[TEMPORAL GROUND TRUTH AUDIT]: No temporal_ground_truth.json found in splits directory.")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Dataset audit tool.")
    parser.add_argument("--splits-dir", type=str, default="datasets/splits", help="Directory with split CSVs.")
    parser.add_argument("--root-dir", type=str, default=".", help="Project root directory.")
    args = parser.parse_args()
    inspect_dataset(args.splits_dir, args.root_dir)

if __name__ == "__main__":
    main()
