#!/usr/bin/env python3
"""
CLI wrapper for running evaluation on the CUHK Avenue benchmark.

Loads model weights (if compatible), runs inference without ground truth,
aligns predictions to native video frames, and evaluates rigorously against
Avenue pixel-level ground-truth masks (testing_label_mask/*.mat).

Saves:
  - reports/metrics/avenue_evaluation_report.json
  - reports/metrics/avenue_frame_metrics.csv
  - reports/metrics/avenue_video_metrics.csv
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.data.avenue_dataset import AvenueDataset
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


def validate_avenue_metadata(metadata_file: Path) -> pd.DataFrame:
    """Validates that metadata exists and contains valid test entries."""
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Avenue metadata file missing: {metadata_file}\n"
            "Please run 'python scripts/prepare_avenue_dataset.py' first."
        )

    df = pd.read_csv(metadata_file, dtype={"video_id": str})
    test_df = df[df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError(f"No test records found in metadata: {metadata_file}")

    print(f"Validated Avenue metadata: {len(test_df)} test videos found in {metadata_file}.")
    return test_df


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: torch.device
) -> Optional[TwoStreamWSSTALNet]:
    """
    Loads TwoStreamWSSTALNet from checkpoint if available and compatible.
    """
    if not checkpoint_path.exists():
        print(f"[WARNING] Checkpoint file does not exist: {checkpoint_path}")
        return None

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Determine num_classes from action_head weight shape in checkpoint if possible
    num_classes = 3
    if "action_head.3.weight" in state_dict:
        num_classes = state_dict["action_head.3.weight"].shape[0]

    model_cfg = config.get("model", {})
    model = TwoStreamWSSTALNet(
        num_classes=num_classes,
        rgb_feature_dim=model_cfg.get("rgb_feature_dim", 512),
        flow_feature_dim=model_cfg.get("flow_feature_dim", 128),
        fusion_dim=model_cfg.get("fusion_dim", 256),
        encoder_layers=model_cfg.get("encoder_layers", 2),
        nheads=model_cfg.get("nheads", 4),
        dropout=model_cfg.get("dropout", 0.15),
        mil_top_k=model_cfg.get("mil_top_k", 3),
        pretrained=False
    )

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded model state_dict (strict=True).")
    except Exception as e:
        print(f"[WARNING] Strict state_dict loading failed: {e}. Attempting non-strict load...")
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Non-strict load complete. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e2:
            print(f"[ERROR] Could not load checkpoint into model: {e2}")
            return None

    model.to(device)
    model.eval()
    return model


def run_video_inference(
    model: TwoStreamWSSTALNet,
    video_item: Dict[str, Any],
    device: torch.device,
    spatial_localizer: Optional[SpatialLocalizer] = None,
    spatial_threshold: float = 0.35
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """
    Runs model inference on sequential clips of a test video without ground truth.
    Returns:
      - clip_scores: np.ndarray of shape [num_clips]
      - pred_spatial_masks: optional list of [H, W] binary masks per frame
    """
    rgb_tensor = video_item["rgb"].to(device)   # [num_clips, T, 3, H, W]
    flow_tensor = video_item["flow"].to(device) # [num_clips, T, 2, H_f, W_f]

    total_clips = rgb_tensor.size(0)

    # Process in chunks through feature extractor to maintain safe memory footprint
    chunk_size = 16
    fused_chunks = []
    with torch.no_grad():
        for i in range(0, total_clips, chunk_size):
            r_c = rgb_tensor[i:i + chunk_size].unsqueeze(0) # [1, C_chunk, T, 3, H, W]
            f_c = flow_tensor[i:i + chunk_size].unsqueeze(0) # [1, C_chunk, T, 2, H_f, W_f]
            fused_c = model.feature_extractor(r_c, f_c)     # [1, C_chunk, fusion_dim]
            fused_chunks.append(fused_c)

        fused = torch.cat(fused_chunks, dim=1) # [1, total_clips, fusion_dim]
        encoded = model.temporal_encoder(fused) # [1, total_clips, fusion_dim]
        segment_anomaly = model.anomaly_head(encoded).squeeze(0).squeeze(-1) # [total_clips]
        clip_scores = segment_anomaly.cpu().numpy()

    # Spatial mask prediction using optical flow motion energy
    pred_spatial_masks = None
    if spatial_localizer is not None:
        total_native_frames = video_item["native_frame_count"]
        pred_spatial_masks = [None] * total_native_frames
        flow_np = video_item["flow"].numpy() # [num_clips, T, 2, H_f, W_f]
        clip_native_ranges = video_item["clip_native_ranges"]

        for c_idx, score in enumerate(clip_scores):
            if score >= spatial_threshold:
                # Extract mean optical flow for this clip
                mean_flow = np.mean(flow_np[c_idx], axis=0) # [2, H_f, W_f]
                rois = spatial_localizer.extract_motion_rois(mean_flow, (360, 640))
                if rois:
                    mask = np.zeros((360, 640), dtype=np.uint8)
                    for roi in rois:
                        x, y, w, h = roi["bbox_pixel"]
                        mask[y:y+h, x:x+w] = 1

                    s_native, e_native = clip_native_ranges[c_idx]
                    for f in range(max(0, s_native), min(total_native_frames, e_native + 1)):
                        if pred_spatial_masks[f] is None:
                            pred_spatial_masks[f] = mask
                        else:
                            pred_spatial_masks[f] = np.maximum(pred_spatial_masks[f], mask)

    return clip_scores, pred_spatial_masks


def evaluate_avenue(
    config_path: str = "configs/avenue.yaml",
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_dir: str = "reports/metrics",
    threshold: Optional[float] = None,
    device_name: str = "auto",
    max_videos: Optional[int] = None,
    include_spatial: bool = True
) -> Dict[str, Any]:
    """
    Main evaluation pipeline for the CUHK Avenue benchmark.
    """
    config_p = Path(config_path)
    cfg = load_config(config_p)

    metadata_path = Path(cfg["dataset"]["metadata_file"])
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    gt_root_path = Path(cfg["dataset"]["ground_truth_root"])
    if not gt_root_path.is_absolute():
        gt_root_path = PROJECT_ROOT / gt_root_path

    output_dir_p = Path(output_dir)
    if not output_dir_p.is_absolute():
        output_dir_p = PROJECT_ROOT / output_dir_p
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # Validate metadata
    test_df = validate_avenue_metadata(metadata_path)

    # Resolve evaluation threshold
    eval_cfg = cfg.get("evaluation", {})
    if threshold is None:
        threshold = float(eval_cfg.get("anomaly_threshold", 0.50))
    spatial_threshold = float(eval_cfg.get("spatial_threshold", 0.35))

    # Resolve device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Evaluation device: {device}")

    # Load model
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_absolute():
        ckpt_p = PROJECT_ROOT / ckpt_p

    model = load_model_from_checkpoint(ckpt_p, cfg, device)
    if model is None:
        print("[ERROR] Evaluation aborted because a valid model could not be loaded.")
        print("Note: In accordance with project requirements, fake predictions will NOT be generated.")
        return {
            "status": "error",
            "message": f"Could not load model from {ckpt_p}. No evaluation performed."
        }

    # Initialize Dataset (Testing split, full video clips)
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

    spatial_localizer = SpatialLocalizer(threshold_ratio=spatial_threshold) if include_spatial else None

    # Run inference across test videos
    num_to_eval = len(test_dataset) if max_videos is None else min(max_videos, len(test_dataset))
    print(f"\nRunning model inference on {num_to_eval} Avenue test videos (ground truth withheld during inference)...")

    video_predictions = []
    for idx in range(num_to_eval):
        item = test_dataset[idx]
        vid_id = item["video_id"]
        total_native_frames = item["native_frame_count"]
        native_fps = item["native_fps"]
        gt_path = item.get("ground_truth_path")

        # Run model inference
        clip_scores, pred_spatial_masks = run_video_inference(
            model=model,
            video_item=item,
            device=device,
            spatial_localizer=spatial_localizer,
            spatial_threshold=spatial_threshold
        )

        # Align clip-level anomaly scores to all native frames
        frame_scores = align_clip_scores_to_frames(
            clip_scores=clip_scores,
            clip_native_ranges=item["clip_native_ranges"],
            total_native_frames=total_native_frames
        )

        # Binary frame labels thresholded at specified decision boundary
        frame_labels = (frame_scores >= threshold).astype(np.int32)

        video_predictions.append({
            "video_id": vid_id,
            "pred_frame_scores": frame_scores,
            "pred_frame_labels": frame_labels,
            "pred_spatial_masks": pred_spatial_masks,
            "ground_truth_path": gt_path,
            "fps": native_fps
        })
        print(f"  Processed test video {vid_id}: {total_native_frames} frames, "
              f"predicted abnormal frames: {int(np.sum(frame_labels == 1))}/{total_native_frames} "
              f"(mean score: {float(np.mean(frame_scores)):.3f}, max: {float(np.max(frame_scores)):.3f})")

    # Evaluate predictions with AvenueEvaluator
    print("\nEvaluating predictions against ground truth masks...")
    evaluator = AvenueEvaluator(
        ground_truth_root=gt_root_path,
        default_threshold=threshold
    )

    eval_result = evaluator.evaluate_test_set(
        video_predictions=video_predictions,
        threshold=threshold
    )

    # Save output artifacts
    report_json_path = output_dir_p / "avenue_evaluation_report.json"
    frame_csv_path = output_dir_p / "avenue_frame_metrics.csv"
    video_csv_path = output_dir_p / "avenue_video_metrics.csv"

    # Add run metadata to report
    summary = eval_result["summary"]
    summary["evaluation_metadata"] = {
        "config_file": str(config_p.resolve()).replace("\\", "/"),
        "checkpoint_file": str(ckpt_p.resolve()).replace("\\", "/"),
        "anomaly_threshold": threshold,
        "spatial_threshold": spatial_threshold,
        "device": str(device)
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    eval_result["frame_metrics_df"].to_csv(frame_csv_path, index=False)
    eval_result["video_metrics_df"].to_csv(video_csv_path, index=False)

    print("\n" + "=" * 75)
    print("CUHK AVENUE BENCHMARK EVALUATION RESULTS")
    print("=" * 75)
    g_fm = summary["global_frame_metrics"]
    g_tm = summary["global_temporal_metrics"]
    g_sm = summary["global_spatial_metrics"]

    print(f"  Evaluated Videos            : {summary['evaluated_videos']}")
    print(f"  Total Evaluated Frames      : {summary['total_test_frames']}")
    print(f"  Ground-Truth Abnormal Frames: {summary['total_gt_abnormal_frames']}")
    print(f"  Predicted Abnormal Frames   : {summary['total_pred_abnormal_frames']}")
    print(f"  Decision Threshold          : {threshold:.2f}")
    print("-" * 75)
    print("  FRAME-LEVEL METRICS:")
    print(f"    TP: {g_fm['tp']:5d}   | FP: {g_fm['fp']:5d}")
    print(f"    FN: {g_fm['fn']:5d}   | TN: {g_fm['tn']:5d}")
    print(f"    Accuracy                  : {g_fm['accuracy'] * 100:.2f}%")
    print(f"    Precision                 : {g_fm['precision'] * 100:.2f}%")
    print(f"    Recall                    : {g_fm['recall'] * 100:.2f}%")
    print(f"    F1 Score                  : {g_fm['f1']:.4f}")
    if g_fm.get("roc_auc") is not None:
        print(f"    Global Frame ROC-AUC      : {g_fm['roc_auc']:.4f}")
    if g_fm.get("pr_auc") is not None:
        print(f"    Global Frame PR-AUC       : {g_fm['pr_auc']:.4f}")
    print("-" * 75)
    print("  TEMPORAL METRICS:")
    print(f"    Temporal Overlap (frames) : {g_tm['temporal_overlap_frames']}")
    print(f"    Temporal Union (frames)   : {g_tm['temporal_union_frames']}")
    print(f"    Global Temporal IoU       : {g_tm['global_temporal_iou']:.4f}")
    print("-" * 75)
    print("  SPATIAL METRICS:")
    print(f"    Available                 : {g_sm['spatial_metrics_available']}")
    if g_sm['mean_iou'] is not None:
        print(f"    Mean Spatial IoU (mIoU)   : {g_sm['mean_iou']:.4f}")
    else:
        print(f"    Note                      : {g_sm.get('note')}")
    print("=" * 75)
    print(f"Saved JSON Report : {report_json_path}")
    print(f"Saved Frame CSV   : {frame_csv_path}")
    print(f"Saved Video CSV   : {video_csv_path}")
    print("=" * 75)

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate on CUHK Avenue Dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/avenue.yaml",
        help="Path to Avenue configuration YAML."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint PT file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/metrics",
        help="Directory to save evaluation reports."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional custom anomaly threshold override."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: 'cpu', 'cuda', or 'auto'."
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optionally limit evaluation to first N test videos for quick verification."
    )
    args = parser.parse_args()

    evaluate_avenue(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device_name=args.device,
        max_videos=args.max_videos
    )


if __name__ == "__main__":
    main()
