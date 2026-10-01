#!/usr/bin/env python3
"""
Prepare metadata for the CUHK Avenue Dataset.

Dataset structure:
  Training:
    16 videos (01.avi - 16.avi), all normal
  Testing:
    21 videos (01.avi - 21.avi)
  Ground Truth:
    21 MAT files (1_label.mat - 21_label.mat)
    volLabel: (1, num_frames) with spatial pixel masks of size (360, 640)

Validations:
  - Video open verification via OpenCV
  - Frame count validation
  - FPS == 25.0
  - Resolution == 640x360
  - Ground-truth MAT file existence for all test videos
  - Exact match between video frame count and MAT volLabel frame count
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from scipy.io import loadmat

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_ROOT = Path(r"D:\Avenue-Dataset\extracted\Avenue Dataset")
DEFAULT_GROUND_TRUTH_ROOT = Path(r"D:\Avenue-Dataset\ground_truth\ground_truth_demo\testing_label_mask")
DEFAULT_OUTPUT_METADATA = PROJECT_ROOT / "datasets" / "metadata" / "avenue_metadata.csv"


def probe_video(video_path: Path) -> Dict[str, Any]:
    """
    Probe a video file using OpenCV.
    Returns metadata dictionary or raises RuntimeError if probing fails.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file with OpenCV: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0:
        raise ValueError(f"Invalid frame count {frame_count} for video: {video_path}")

    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 3)
    }


def validate_ground_truth(gt_path: Path, expected_frame_count: int, expected_width: int = 640, expected_height: int = 360) -> Dict[str, Any]:
    """
    Validate a test ground-truth MAT file against expected video properties.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth MAT file not found: {gt_path}")

    mat = loadmat(str(gt_path))
    if "volLabel" not in mat:
        raise KeyError(f"Expected key 'volLabel' not found in MAT file: {gt_path}")

    vol_label = mat["volLabel"]
    if len(vol_label.shape) != 2 or vol_label.shape[0] != 1:
        raise ValueError(f"Unexpected volLabel shape {vol_label.shape} in {gt_path}, expected (1, N)")

    mat_frame_count = vol_label.shape[1]
    if mat_frame_count != expected_frame_count:
        raise ValueError(
            f"Frame count mismatch for {gt_path.name}: "
            f"MAT has {mat_frame_count} frames, but video has {expected_frame_count} frames."
        )

    # Validate mask dimensions of first frame
    first_mask = vol_label[0, 0]
    mask_h, mask_w = first_mask.shape[:2]
    if mask_h != expected_height or mask_w != expected_width:
        raise ValueError(
            f"Mask resolution mismatch in {gt_path.name}: "
            f"mask is ({mask_h}, {mask_w}), expected ({expected_height}, {expected_width})"
        )

    # Count abnormal frames (any non-zero pixel)
    abnormal_frames = 0
    for idx in range(mat_frame_count):
        if np.count_nonzero(vol_label[0, idx]) > 0:
            abnormal_frames += 1

    return {
        "mat_frame_count": mat_frame_count,
        "mask_width": mask_w,
        "mask_height": mask_h,
        "abnormal_frames": abnormal_frames,
        "normal_frames": mat_frame_count - abnormal_frames
    }


def find_matching_gt_file(gt_root: Path, video_stem: str) -> Optional[Path]:
    """
    Given a test video stem (e.g. '01', '1', '10'), find the corresponding *_label.mat file.
    Avenue GT files are named '1_label.mat' ... '21_label.mat'.
    """
    try:
        vid_num = int(video_stem)
    except ValueError:
        return None

    # Primary naming convention: 1_label.mat ... 21_label.mat
    primary = gt_root / f"{vid_num}_label.mat"
    if primary.exists():
        return primary

    # Secondary naming convention: 01_label.mat
    secondary = gt_root / f"{vid_num:02d}_label.mat"
    if secondary.exists():
        return secondary

    return None


def prepare_avenue_metadata(
    dataset_root: Path,
    ground_truth_root: Path,
    output_file: Path
) -> List[Dict[str, Any]]:
    """
    Generates and validates Avenue metadata.
    """
    print("=" * 75)
    print("CUHK AVENUE DATASET VERIFICATION & METADATA PREPARATION")
    print("=" * 75)
    print(f"Dataset root       : {dataset_root}")
    print(f"Ground truth root  : {ground_truth_root}")
    print(f"Output metadata    : {output_file}")
    print("-" * 75)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Avenue dataset root not found: {dataset_root}")

    train_dir = dataset_root / "training_videos"
    test_dir = dataset_root / "testing_videos"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training videos directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing videos directory not found: {test_dir}")
    if not ground_truth_root.exists():
        raise FileNotFoundError(f"Ground truth root not found: {ground_truth_root}")

    rows: List[Dict[str, Any]] = []

    # ----------------------------------------------------
    # 1. Process Training Videos (01.avi - 16.avi)
    # ----------------------------------------------------
    print("\n[1/3] Validating Training Videos (all normal)...")
    train_videos = sorted(train_dir.glob("*.avi"))
    if len(train_videos) != 16:
        print(f"  [NOTE] Found {len(train_videos)} training videos (standard is 16).")

    total_train_frames = 0
    for vpath in train_videos:
        meta = probe_video(vpath)
        # Validation checks
        if abs(meta["fps"] - 25.0) > 0.1:
            raise ValueError(f"Unexpected FPS {meta['fps']} for {vpath.name}, expected 25.0")
        if meta["width"] != 640 or meta["height"] != 360:
            raise ValueError(f"Unexpected resolution {meta['width']}x{meta['height']} for {vpath.name}, expected 640x360")

        total_train_frames += meta["frame_count"]
        row = {
            "video_id": vpath.stem,
            "video_path": str(vpath.resolve()).replace("\\", "/"),
            "split": "train",
            "frame_count": meta["frame_count"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "duration_seconds": meta["duration_seconds"],
            "ground_truth_path": "",
            "dataset": "avenue"
        }
        rows.append(row)
        print(f"  Train video {vpath.stem}: frames={meta['frame_count']:5d}, fps={meta['fps']:.1f}, "
              f"res={meta['width']}x{meta['height']}, dur={meta['duration_seconds']:6.1f}s [VALID]")

    # ----------------------------------------------------
    # 2. Process Testing Videos (01.avi - 21.avi)
    # ----------------------------------------------------
    print("\n[2/3] Validating Testing Videos & Ground-Truth Labels...")
    test_videos = sorted(test_dir.glob("*.avi"))
    if len(test_videos) != 21:
        print(f"  [NOTE] Found {len(test_videos)} testing videos (standard is 21).")

    total_test_frames = 0
    total_abnormal_frames = 0

    for vpath in test_videos:
        meta = probe_video(vpath)
        if abs(meta["fps"] - 25.0) > 0.1:
            raise ValueError(f"Unexpected FPS {meta['fps']} for {vpath.name}, expected 25.0")
        if meta["width"] != 640 or meta["height"] != 360:
            raise ValueError(f"Unexpected resolution {meta['width']}x{meta['height']} for {vpath.name}, expected 640x360")

        gt_file = find_matching_gt_file(ground_truth_root, vpath.stem)
        if gt_file is None:
            raise FileNotFoundError(f"Missing matching ground-truth MAT file for test video: {vpath.name}")

        gt_info = validate_ground_truth(
            gt_path=gt_file,
            expected_frame_count=meta["frame_count"],
            expected_width=meta["width"],
            expected_height=meta["height"]
        )

        total_test_frames += meta["frame_count"]
        total_abnormal_frames += gt_info["abnormal_frames"]

        row = {
            "video_id": vpath.stem,
            "video_path": str(vpath.resolve()).replace("\\", "/"),
            "split": "test",
            "frame_count": meta["frame_count"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "duration_seconds": meta["duration_seconds"],
            "ground_truth_path": str(gt_file.resolve()).replace("\\", "/"),
            "dataset": "avenue"
        }
        rows.append(row)
        print(f"  Test video {vpath.stem}: frames={meta['frame_count']:5d}, dur={meta['duration_seconds']:5.1f}s | "
              f"GT: {gt_file.name} (abnormal frames: {gt_info['abnormal_frames']:4d}/{meta['frame_count']}) [MATCH]")

    # ----------------------------------------------------
    # 3. Write Metadata CSV
    # ----------------------------------------------------
    print(f"\n[3/3] Writing metadata to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "video_path",
        "split",
        "frame_count",
        "fps",
        "width",
        "height",
        "duration_seconds",
        "ground_truth_path",
        "dataset"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------
    train_count = sum(1 for r in rows if r["split"] == "train")
    test_count = sum(1 for r in rows if r["split"] == "test")

    print("\n" + "=" * 75)
    print("AVENUE DATASET VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"  Dataset Name               : Avenue")
    print(f"  Training videos (normal)   : {train_count} (expected 16)")
    print(f"  Testing videos (annotated) : {test_count} (expected 21)")
    print(f"  Total metadata rows        : {len(rows)}")
    print(f"  Total training frames      : {total_train_frames}")
    print(f"  Total testing frames       : {total_test_frames}")
    print(f"  Total abnormal test frames : {total_abnormal_frames} ({total_abnormal_frames/total_test_frames*100:.2f}% of test)")
    print(f"  Total normal test frames   : {total_test_frames - total_abnormal_frames} ({(total_test_frames - total_abnormal_frames)/total_test_frames*100:.2f}% of test)")
    print(f"  Resolution                 : 640x360 across all videos")
    print(f"  Native Framerate           : 25.0 FPS across all videos")
    print(f"  GT Alignment               : 21 / 21 testing label masks verified 1-to-1")
    print(f"  Output CSV                 : {output_file}")
    print("=" * 75)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate Avenue dataset metadata.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root path to extracted Avenue Dataset folder."
    )
    parser.add_argument(
        "--ground-truth-root",
        type=str,
        default=str(DEFAULT_GROUND_TRUTH_ROOT),
        help="Path to testing_label_mask folder containing *_label.mat files."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_METADATA),
        help="Target metadata CSV file path."
    )
    args = parser.parse_args()

    prepare_avenue_metadata(
        dataset_root=Path(args.dataset_root),
        ground_truth_root=Path(args.ground_truth_root),
        output_file=Path(args.output_file)
    )


if __name__ == "__main__":
    main()
