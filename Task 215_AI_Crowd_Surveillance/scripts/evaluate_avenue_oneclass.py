#!/usr/bin/env python3
"""
Evaluation Script for Avenue One-Class Anomaly Detection (Experiment G).

Evaluates the trained one-class prototype detector against the 21 Avenue test videos
using pixel-level ground-truth masks (testing_label_mask/*.mat).

Integrity guarantee:
- Ground-truth masks are NOT used during feature extraction, score generation, or thresholding.
- Predictions are generated entirely blind to test labels.
- Ground truth is passed strictly to AvenueEvaluator for metric calculation.
- Spatial pseudo-localization is based on optical flow motion energy.

Saves:
  - reports/metrics/avenue_oneclass/avenue_oneclass_evaluation_report.json
  - reports/metrics/avenue_oneclass/avenue_oneclass_frame_metrics.csv
  - reports/metrics/avenue_oneclass/avenue_oneclass_video_metrics.csv
  - reports/metrics/avenue_oneclass/score_distribution.json
  - reports/metrics/avenue_oneclass/threshold_selection.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.avenue_dataset import AvenueDataset
from src.features.feature_extractor import TwoStreamFeatureExtractor
from src.models.avenue_oneclass import (
    AvenueOneClassDetector,
    extract_clip_features,
    compute_score_statistics
)
from src.evaluation.avenue_evaluator import (
    AvenueEvaluator,
    align_clip_scores_to_frames
)
from src.localization.spatial_localizer import SpatialLocalizer


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_and_load_feature_extractor(
    config: Dict[str, Any],
    extra_metadata: Dict[str, Any],
    device: torch.device
) -> TwoStreamFeatureExtractor:
    """Builds and loads the feature extractor."""
    model_cfg = config.get("model", {})
    rgb_dim = model_cfg.get("rgb_feature_dim", 512)
    flow_dim = model_cfg.get("flow_feature_dim", 128)
    fusion_dim = model_cfg.get("fusion_dim", 256)

    feature_extractor = TwoStreamFeatureExtractor(
        rgb_feature_dim=rgb_dim,
        flow_feature_dim=flow_dim,
        fusion_dim=fusion_dim,
        pretrained=False,
        freeze_rgb_backbone=True
    )

    fe_state = extra_metadata.get("feature_extractor_state_dict")
    if fe_state is not None:
        feature_extractor.load_state_dict(fe_state)
    else:
        # Fall back to best_model.pt or default
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            fe_dict = {
                k.replace("feature_extractor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("feature_extractor.")
            }
            if fe_dict:
                feature_extractor.load_state_dict(fe_dict, strict=False)

    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def evaluate_avenue_oneclass(
    config_path: str = "configs/avenue.yaml",
    checkpoint_path: str = "checkpoints/avenue/avenue_oneclass_best.pt",
    output_dir: str = "reports/metrics/avenue_oneclass",
    max_videos: Optional[int] = None,
    device_name: str = "auto"
) -> Dict[str, Any]:
    """
    Evaluates Avenue One-Class Anomaly Detection on test videos.
    """
    print("=" * 75)
    print("EVALUATING CUHK AVENUE ONE-CLASS DETECTOR (EXPERIMENT G)")
    print("=" * 75)

    config_p = Path(config_path)
    if not config_p.is_absolute():
        config_p = PROJECT_ROOT / config_p
    cfg = load_config(config_p)

    out_dir_p = Path(output_dir)
    if not out_dir_p.is_absolute():
        out_dir_p = PROJECT_ROOT / out_dir_p
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_absolute():
        ckpt_p = PROJECT_ROOT / ckpt_p

    if not ckpt_p.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_p}\n"
            "Please run 'python scripts/train_avenue_oneclass.py' first."
        )

    # Resolve device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Device: {device}")

    # Load One-Class detector
    print(f"Loading checkpoint: {ckpt_p}")
    detector = AvenueOneClassDetector()
    extra_metadata = detector.load(ckpt_p)

    threshold = detector.threshold
    if threshold is None:
        raise ValueError("Loaded checkpoint has no threshold set.")
    print(f"Loaded Centroid Dimension   : {detector.feature_dim}")
    print(f"Operating Decision Threshold: {threshold:.4f} (calibrated at {detector.calibration_percentile}% of normal validation)")

    # Post-processing settings
    oc_cfg = cfg.get("avenue_oneclass", {})
    smoothing_window = int(oc_cfg.get("score_smoothing_window", 5))
    min_duration_sec = float(oc_cfg.get("min_anomaly_duration_seconds", 0.5))
    merge_gap_sec = float(oc_cfg.get("merge_gap_seconds", 0.5))

    print(f"Temporal Post-Processing:")
    print(f"  Smoothing Window          : {smoothing_window} frames")
    print(f"  Min Anomaly Duration      : {min_duration_sec}s")
    print(f"  Merge Gap                 : {merge_gap_sec}s")
    print("-" * 75)

    # Build feature extractor
    feature_extractor = build_and_load_feature_extractor(cfg, extra_metadata, device)

    # Initialize Test Dataset
    metadata_path = Path(cfg["dataset"]["metadata_file"])
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    gt_root_path = Path(cfg["dataset"]["ground_truth_root"])
    if not gt_root_path.is_absolute():
        gt_root_path = PROJECT_ROOT / gt_root_path

    test_dataset = AvenueDataset(
        metadata_file=str(metadata_path),
        split="test",
        clip_length=cfg["video"].get("clip_length", 16),
        stride=cfg["video"].get("stride", 16),
        target_fps=cfg["video"].get("target_fps", 16),
        image_size=cfg["video"].get("image_size", [224, 224]),
        flow_size=cfg["video"].get("flow_size", [112, 112]),
        full_video_clips=True,
        use_cache=True
    )

    num_test = len(test_dataset) if max_videos is None else min(max_videos, len(test_dataset))
    print(f"\nProcessing {num_test} Avenue test videos (ground truth withheld during inference)...")

    # Spatial localizer
    eval_cfg = cfg.get("evaluation", {})
    spatial_threshold = float(eval_cfg.get("spatial_threshold", 0.35))
    spatial_localizer = SpatialLocalizer(threshold_ratio=spatial_threshold)

    video_predictions = []
    all_raw_scores = []
    all_smoothed_scores = []

    for idx in range(num_test):
        item = test_dataset[idx]
        vid_id = str(item["video_id"]).zfill(2)
        total_native_frames = int(item["native_frame_count"])
        native_fps = float(item["native_fps"])
        gt_path = item.get("ground_truth_path")

        # 1. Extract clip features (no ground truth used)
        clip_features = extract_clip_features(
            feature_extractor=feature_extractor,
            rgb_tensor=item["rgb"],
            flow_tensor=item["flow"],
            device=device
        )

        # 2. Compute clip anomaly scores via cosine distance to normal centroid
        clip_scores = detector.predict_score(clip_features)

        # 3. Align clip scores to native frames
        frame_scores = align_clip_scores_to_frames(
            clip_scores=clip_scores,
            clip_native_ranges=item["clip_native_ranges"],
            total_native_frames=total_native_frames
        )

        # 4. Temporal post-processing: smoothing, thresholding, gap merging, min-duration filtering
        smoothed_scores, binary_preds, intervals = detector.temporal_postprocess(
            frame_scores=frame_scores,
            threshold=threshold,
            smoothing_window=smoothing_window,
            min_anomaly_duration_seconds=min_duration_sec,
            merge_gap_seconds=merge_gap_sec,
            fps=native_fps
        )

        all_raw_scores.extend(frame_scores.tolist())
        all_smoothed_scores.extend(smoothed_scores.tolist())

        # 5. Motion-energy spatial pseudo-localization for predicted abnormal frames
        pred_spatial_masks = [None] * total_native_frames
        flow_np = item["flow"].numpy() # [num_clips, T, 2, H_f, W_f]
        clip_native_ranges = item["clip_native_ranges"]

        for c_i, c_score in enumerate(clip_scores):
            s_nat, e_nat = clip_native_ranges[c_i]
            # If any frame in clip range is predicted abnormal
            if np.any(binary_preds[max(0, s_nat):min(total_native_frames, e_nat + 1)] == 1):
                mean_flow = np.mean(flow_np[c_i], axis=0) # [2, H_f, W_f]
                rois = spatial_localizer.extract_motion_rois(mean_flow, (360, 640))
                if rois:
                    mask = np.zeros((360, 640), dtype=np.uint8)
                    for r in rois:
                        x, y, w, h = r["bbox_pixel"]
                        mask[y:y+h, x:x+w] = 1

                    for f in range(max(0, s_nat), min(total_native_frames, e_nat + 1)):
                        if binary_preds[f] == 1:
                            if pred_spatial_masks[f] is None:
                                pred_spatial_masks[f] = mask
                            else:
                                pred_spatial_masks[f] = np.maximum(pred_spatial_masks[f], mask)

        pred_abnormal = int(np.sum(binary_preds == 1))
        print(f"  Test video {vid_id}: {total_native_frames:5d} frames | "
              f"Predicted abnormal: {pred_abnormal:4d} frames ({pred_abnormal/total_native_frames*100:5.1f}%) | "
              f"Score range: [{float(np.min(frame_scores)):.3f}, {float(np.max(frame_scores)):.3f}] | "
              f"Intervals: {len(intervals)}")

        video_predictions.append({
            "video_id": vid_id,
            "pred_frame_scores": smoothed_scores,
            "pred_frame_labels": binary_preds,
            "pred_spatial_masks": pred_spatial_masks,
            "ground_truth_path": gt_path,
            "fps": native_fps
        })

    # Evaluate predictions using AvenueEvaluator
    print("\nEvaluating predictions against ground truth masks via AvenueEvaluator...")
    evaluator = AvenueEvaluator(
        ground_truth_root=gt_root_path,
        default_threshold=threshold
    )

    eval_result = evaluator.evaluate_test_set(
        video_predictions=video_predictions,
        threshold=threshold
    )

    summary = eval_result["summary"]

    # Enrich summary with Experiment G specific information
    summary["experiment"] = "Experiment G: CUHK Avenue One-Class Anomaly Detection"
    summary["methodology"] = {
        "paradigm": "Normal-only unsupervised one-class anomaly detection",
        "training_videos": oc_cfg.get("train_video_ids", ["01-12"]),
        "validation_videos": oc_cfg.get("validation_video_ids", ["13-16"]),
        "validation_percentile": detector.calibration_percentile,
        "calibrated_threshold": threshold,
        "threshold_source": "Normal validation videos only (no test labels used)",
        "smoothing_window": smoothing_window,
        "min_anomaly_duration_seconds": min_duration_sec,
        "merge_gap_seconds": merge_gap_sec,
        "spatial_method": "Motion-Energy Based Spatial Pseudo-Localization",
        "spatial_evaluation_notice": "Avenue ground-truth masks are used solely for spatial evaluation, never for prediction."
    }
    summary["normal_training_stats"] = detector.normal_training_stats
    summary["normal_validation_stats"] = detector.normal_validation_stats

    # Save outputs
    report_json_path = out_dir_p / "avenue_oneclass_evaluation_report.json"
    frame_csv_path = out_dir_p / "avenue_oneclass_frame_metrics.csv"
    video_csv_path = out_dir_p / "avenue_oneclass_video_metrics.csv"
    thresh_json_path = out_dir_p / "threshold_selection.json"
    score_dist_path = out_dir_p / "score_distribution.json"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    eval_result["frame_metrics_df"].to_csv(frame_csv_path, index=False)
    eval_result["video_metrics_df"].to_csv(video_csv_path, index=False)

    threshold_selection_data = {
        "method": "Normal Validation Score Percentile",
        "validation_video_ids": oc_cfg.get("validation_video_ids", ["13-16"]),
        "percentile": detector.calibration_percentile,
        "calibrated_threshold": round(float(threshold), 6),
        "validation_normal_statistics": detector.normal_validation_stats,
        "integrity_confirmation": "Threshold selected using normal validation videos only. No test ground truth was inspected or used."
    }
    with open(thresh_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_selection_data, f, indent=2)

    score_dist_data = {
        "training_normal_scores": detector.normal_training_stats,
        "validation_normal_scores": detector.normal_validation_stats,
        "test_scores_summary": compute_score_statistics(np.array(all_smoothed_scores, dtype=np.float32)),
        "threshold": round(float(threshold), 6)
    }
    with open(score_dist_path, "w", encoding="utf-8") as f:
        json.dump(score_dist_data, f, indent=2)

    # Print Terminal Summary
    g_fm = summary["global_frame_metrics"]
    g_tm = summary["global_temporal_metrics"]
    g_sm = summary["global_spatial_metrics"]

    print("\n" + "=" * 75)
    print("CUHK AVENUE ONE-CLASS EXPERIMENT G RESULTS")
    print("=" * 75)
    print(f"  Training videos             : {oc_cfg.get('train_video_ids')}")
    print(f"  Validation videos           : {oc_cfg.get('validation_video_ids')}")
    print(f"  Test videos                 : {summary['evaluated_videos']} videos ({summary['total_test_frames']} frames)")
    print(f"  Threshold                   : {threshold:.4f}")
    print("  Threshold source            : Normal validation videos only.")
    print("-" * 75)
    print(f"  Accuracy                    : {g_fm['accuracy'] * 100:.2f}%")
    print(f"  Precision                   : {g_fm['precision'] * 100:.2f}%")
    print(f"  Recall                      : {g_fm['recall'] * 100:.2f}%")
    print(f"  F1 Score                    : {g_fm['f1']:.4f}")
    print(f"  ROC-AUC                     : {g_fm.get('roc_auc', 'N/A')}")
    print(f"  PR-AUC                      : {g_fm.get('pr_auc', 'N/A')}")
    print("-" * 75)
    print(f"  Temporal IoU                : {g_tm['global_temporal_iou']:.4f}")
    print(f"  Mean Spatial IoU            : {g_sm.get('mean_iou', 'N/A')}")
    print("-" * 75)
    print(f"  TP: {g_fm['tp']:5d}   | FP: {g_fm['fp']:5d}")
    print(f"  FN: {g_fm['fn']:5d}   | TN: {g_fm['tn']:5d}")
    print(f"  Predicted abnormal frames   : {summary['total_pred_abnormal_frames']}")
    print(f"  Ground-truth abnormal frames: {summary['total_gt_abnormal_frames']}")
    print("-" * 75)
    print(f"  Checkpoint                  : {ckpt_p}")
    print(f"  Evaluation report           : {report_json_path}")
    print("=" * 75)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Avenue One-Class Detector.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/avenue.yaml",
        help="Path to Avenue configuration YAML."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/avenue/avenue_oneclass_best.pt",
        help="Path to one-class detector checkpoint."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/metrics/avenue_oneclass",
        help="Output directory for reports and metrics."
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on number of test videos."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'cpu', 'cuda', or 'auto'."
    )
    args = parser.parse_args()

    evaluate_avenue_oneclass(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
