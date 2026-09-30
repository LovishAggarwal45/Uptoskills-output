#!/usr/bin/env python3
"""
Training and Validation Script for Avenue One-Class Anomaly Detection (Experiment G).

Trains a normal-prototype one-class anomaly detector using ONLY normal Avenue training videos.
Splits normal videos into:
  - Training normal videos:   01-12 (learns normal prototype centroid)
  - Validation normal videos: 13-16 (calibrates decision threshold via score percentile)

Does NOT use any testing videos or ground-truth masks for training or threshold selection.
Saves:
  - checkpoints/avenue/avenue_oneclass_best.pt
  - checkpoints/avenue/avenue_oneclass_metadata.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

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
    extract_clip_features
)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads configuration YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_feature_extractor(config: Dict[str, Any], device: torch.device) -> TwoStreamFeatureExtractor:
    """
    Initializes TwoStreamFeatureExtractor.
    Reuses pretrained feature extractor weights from checkpoints/best_model.pt if available,
    otherwise initializes with ImageNet pretrained weights.
    """
    model_cfg = config.get("model", {})
    rgb_dim = model_cfg.get("rgb_feature_dim", 512)
    flow_dim = model_cfg.get("flow_feature_dim", 128)
    fusion_dim = model_cfg.get("fusion_dim", 256)

    feature_extractor = TwoStreamFeatureExtractor(
        rgb_feature_dim=rgb_dim,
        flow_feature_dim=flow_dim,
        fusion_dim=fusion_dim,
        pretrained=True,
        freeze_rgb_backbone=True
    )

    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    if ckpt_path.exists():
        try:
            print(f"Loading feature extractor weights from {ckpt_path.name}...")
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            fe_dict = {
                k.replace("feature_extractor.", ""): v
                for k, v in state_dict.items()
                if k.startswith("feature_extractor.")
            }
            if fe_dict:
                feature_extractor.load_state_dict(fe_dict, strict=True)
                print("Successfully loaded trained feature extractor backbone weights.")
        except Exception as e:
            print(f"[NOTE] Could not load checkpoint weights ({e}). Using ImageNet pretrained backbone.")

    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def train_avenue_oneclass(
    config_path: str = "configs/avenue.yaml",
    output_dir: str = "checkpoints/avenue",
    device_name: str = "auto"
) -> Dict[str, Any]:
    """
    Main training and validation routine for Avenue One-Class anomaly detection.
    """
    print("=" * 75)
    print("TRAINING CUHK AVENUE ONE-CLASS ANOMALY DETECTOR (EXPERIMENT G)")
    print("=" * 75)

    config_p = Path(config_path)
    if not config_p.is_absolute():
        config_p = PROJECT_ROOT / config_p
    cfg = load_config(config_p)

    out_dir_p = Path(output_dir)
    if not out_dir_p.is_absolute():
        out_dir_p = PROJECT_ROOT / out_dir_p
    out_dir_p.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(cfg["dataset"]["metadata_file"])
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    # Resolve device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Computation Device        : {device}")

    # One-Class configuration
    oc_cfg = cfg.get("avenue_oneclass", {})
    train_ids = [str(x).zfill(2) for x in oc_cfg.get(
        "train_video_ids",
        ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    )]
    val_ids = [str(x).zfill(2) for x in oc_cfg.get(
        "validation_video_ids",
        ["13", "14", "15", "16"]
    )]
    percentile = float(oc_cfg.get("validation_normal_percentile", 99.0))

    print(f"Normal Training Videos    : {train_ids} ({len(train_ids)} videos)")
    print(f"Normal Validation Videos  : {val_ids} ({len(val_ids)} videos)")
    print(f"Threshold Percentile      : {percentile}%")
    print("-" * 75)

    # Load Avenue training dataset (normal only)
    train_dataset = AvenueDataset(
        metadata_file=str(metadata_path),
        split="train",
        clip_length=cfg["video"].get("clip_length", 16),
        stride=cfg["video"].get("stride", 16),
        target_fps=cfg["video"].get("target_fps", 16),
        image_size=cfg["video"].get("image_size", [224, 224]),
        flow_size=cfg["video"].get("flow_size", [112, 112]),
        full_video_clips=True,
        use_cache=True
    )

    feature_extractor = build_feature_extractor(cfg, device)

    # 1. Extract Training Normal Features (01-12)
    print("\n[1/4] Extracting features from normal training videos (01-12)...")
    train_features_list = []
    train_clip_counts = {}

    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        vid_id = str(item["video_id"]).zfill(2)
        if vid_id in train_ids:
            feats = extract_clip_features(
                feature_extractor=feature_extractor,
                rgb_tensor=item["rgb"],
                flow_tensor=item["flow"],
                device=device
            )
            train_features_list.append(feats)
            train_clip_counts[vid_id] = len(feats)
            print(f"  Train video {vid_id}: extracted {len(feats):4d} clips, feature shape: {feats.shape}")

    if not train_features_list:
        raise RuntimeError("No training features could be extracted.")

    all_train_features = np.concatenate(train_features_list, axis=0) # [total_train_clips, fusion_dim]
    print(f"Total normal training clips: {len(all_train_features)}, dimension: {all_train_features.shape[1]}")

    # 2. Fit Normal Centroid / Prototype
    print("\n[2/4] Fitting normal prototype centroid...")
    detector = AvenueOneClassDetector(feature_dim=all_train_features.shape[1])
    train_stats = detector.fit(all_train_features)
    print(f"  Training distance mean  : {train_stats['mean']:.4f} (std: {train_stats['std']:.4f})")
    print(f"  Training distance range : [{train_stats['min']:.4f}, {train_stats['max']:.4f}]")
    print(f"  Training distance p90   : {train_stats['p90']:.4f}")
    print(f"  Training distance p99   : {train_stats['p99']:.4f}")

    # 3. Extract Validation Normal Features & Calibrate Threshold (13-16)
    print(f"\n[3/4] Calibrating decision threshold on normal validation videos (13-16) at {percentile}th percentile...")
    val_scores_list = []
    val_clip_counts = {}

    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        vid_id = str(item["video_id"]).zfill(2)
        if vid_id in val_ids:
            feats = extract_clip_features(
                feature_extractor=feature_extractor,
                rgb_tensor=item["rgb"],
                flow_tensor=item["flow"],
                device=device
            )
            scores = detector.predict_score(feats)
            val_scores_list.append(scores)
            val_clip_counts[vid_id] = len(scores)
            print(f"  Validation video {vid_id}: {len(scores):4d} clips, mean score: {np.mean(scores):.4f}, max: {np.max(scores):.4f}")

    if not val_scores_list:
        raise RuntimeError("No validation features could be extracted.")

    all_val_scores = np.concatenate(val_scores_list, axis=0)
    threshold = detector.calibrate_threshold(all_val_scores, percentile=percentile)
    val_stats = detector.normal_validation_stats

    print(f"\nCalculated Operating Threshold: {threshold:.4f} (at {percentile}% of normal validation scores)")
    print(f"  Validation normal score mean  : {val_stats['mean']:.4f} (std: {val_stats['std']:.4f})")
    print(f"  Validation normal score range : [{val_stats['min']:.4f}, {val_stats['max']:.4f}]")

    # 4. Save Checkpoint and Metadata
    print("\n[4/4] Saving model checkpoint and training metadata...")
    ckpt_file = out_dir_p / "avenue_oneclass_best.pt"
    meta_file = out_dir_p / "avenue_oneclass_metadata.json"

    # Save detector state
    detector.save(
        checkpoint_path=ckpt_file,
        extra_metadata={
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "config": oc_cfg
        }
    )

    metadata_payload = {
        "dataset": "CUHK Avenue",
        "method": "One-Class Cosine Distance Prototype Detector",
        "training_paradigm": "Normal-only unsupervised anomaly detection",
        "train_video_ids": train_ids,
        "train_clip_counts": train_clip_counts,
        "total_train_clips": int(len(all_train_features)),
        "validation_video_ids": val_ids,
        "validation_clip_counts": val_clip_counts,
        "total_validation_clips": int(len(all_val_scores)),
        "feature_dimension": int(all_train_features.shape[1]),
        "calibration_percentile": percentile,
        "calibrated_threshold": round(float(threshold), 6),
        "normal_training_stats": {k: round(v, 6) if isinstance(v, float) else v for k, v in train_stats.items()},
        "normal_validation_stats": {k: round(v, 6) if isinstance(v, float) else v for k, v in val_stats.items()},
        "device": str(device),
        "configuration": oc_cfg,
        "integrity_notice": "Threshold selected using normal validation videos only. No test ground truth used."
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    print("\n" + "=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)
    print(f"  Saved Detector Checkpoint : {ckpt_file}")
    print(f"  Saved Training Metadata   : {meta_file}")
    print(f"  Calibrated Threshold      : {threshold:.4f}")
    print("  Threshold Source          : Normal validation videos (13-16) only.")
    print("=" * 75)

    return metadata_payload


def main():
    parser = argparse.ArgumentParser(description="Train Avenue One-Class Anomaly Detector.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/avenue.yaml",
        help="Path to Avenue configuration YAML."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/avenue",
        help="Directory to save checkpoint and metadata."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cpu', 'cuda', or 'auto')."
    )
    args = parser.parse_args()

    train_avenue_oneclass(
        config_path=args.config,
        output_dir=args.output_dir,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
