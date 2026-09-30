#!/usr/bin/env python3
"""
Evaluation Script for CUHK Avenue Feature-Memory / KNN
One-Class Anomaly Detection (Experiment H).

Integrity guarantee:
- Ground-truth masks are NOT used during inference.
- Test labels are only passed to AvenueEvaluator after predictions
  have already been generated.
- Threshold was calibrated using normal validation videos 13-16.
- Spatial outputs are motion-energy pseudo-localizations.
- Experiment G is not modified.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import torch
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.data.avenue_dataset import AvenueDataset
from src.features.feature_extractor import TwoStreamFeatureExtractor

from src.models.avenue_knn import AvenueKNNDetector

from src.models.avenue_oneclass import (
    extract_clip_features,
    compute_score_statistics,
)

from src.evaluation.avenue_evaluator import (
    AvenueEvaluator,
    align_clip_scores_to_frames,
)

from src.localization.spatial_localizer import SpatialLocalizer


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

def smooth_scores(
    scores: np.ndarray,
    window_size: int = 5
) -> np.ndarray:
    """
    Apply centered moving-average smoothing.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    if scores.size == 0:
        return scores

    if window_size <= 1:
        return scores.copy()

    window_size = int(window_size)

    # Use odd window
    if window_size % 2 == 0:
        window_size += 1

    # Don't make window larger than sequence
    if window_size > len(scores):
        window_size = len(scores)

        if window_size % 2 == 0:
            window_size -= 1

    if window_size <= 1:
        return scores.copy()

    kernel = (
        np.ones(
            window_size,
            dtype=np.float32
        )
        / float(window_size)
    )

    pad = window_size // 2

    padded = np.pad(
        scores,
        (pad, pad),
        mode="edge"
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    return smoothed.astype(
        np.float32
    )


# ============================================================
# TEMPORAL POST-PROCESSING
# ============================================================

def temporal_postprocess(
    scores: np.ndarray,
    threshold: float,
    fps: float,
    smoothing_window: int = 5,
    min_anomaly_duration_seconds: float = 0.5,
    merge_gap_seconds: float = 0.5,
):
    """
    Convert continuous anomaly scores into cleaned
    binary anomaly predictions.

    Steps:
        1. Smooth scores.
        2. Apply threshold.
        3. Remove very short anomaly segments.
        4. Merge nearby anomaly segments.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    if scores.size == 0:
        return (
            np.zeros(
                0,
                dtype=np.float32
            ),
            np.zeros(
                0,
                dtype=np.int32
            ),
        )

    # --------------------------------------------------------
    # 1. Smooth
    # --------------------------------------------------------

    smoothed = smooth_scores(
        scores,
        window_size=smoothing_window
    )

    # --------------------------------------------------------
    # 2. Threshold
    # --------------------------------------------------------

    predictions = (
        smoothed >= float(threshold)
    ).astype(
        np.int32
    )

    if fps <= 0:
        fps = 25.0

    # --------------------------------------------------------
    # 3. Convert seconds to frames
    # --------------------------------------------------------

    min_duration_frames = max(
        1,
        int(
            round(
                min_anomaly_duration_seconds
                * fps
            )
        )
    )

    merge_gap_frames = max(
        0,
        int(
            round(
                merge_gap_seconds
                * fps
            )
        )
    )

    # --------------------------------------------------------
    # 4. Find anomaly segments
    # --------------------------------------------------------

    segments = []

    start = None

    for i, value in enumerate(
        predictions
    ):

        if value == 1 and start is None:
            start = i

        elif value == 0 and start is not None:

            segments.append(
                (
                    start,
                    i - 1
                )
            )

            start = None

    if start is not None:

        segments.append(
            (
                start,
                len(predictions) - 1
            )
        )

    # --------------------------------------------------------
    # 5. Remove short segments
    # --------------------------------------------------------

    filtered_segments = []

    for start, end in segments:

        duration = (
            end - start + 1
        )

        if duration >= min_duration_frames:

            filtered_segments.append(
                (
                    start,
                    end
                )
            )

    # --------------------------------------------------------
    # 6. Merge nearby segments
    # --------------------------------------------------------

    merged_segments = []

    for start, end in filtered_segments:

        if not merged_segments:

            merged_segments.append(
                [
                    start,
                    end
                ]
            )

            continue

        previous_start, previous_end = (
            merged_segments[-1]
        )

        gap = (
            start
            - previous_end
            - 1
        )

        if gap <= merge_gap_frames:

            merged_segments[-1][1] = end

        else:

            merged_segments.append(
                [
                    start,
                    end
                ]
            )

    # --------------------------------------------------------
    # 7. Reconstruct predictions
    # --------------------------------------------------------

    final_predictions = np.zeros_like(
        predictions
    )

    for start, end in merged_segments:

        final_predictions[
            start:end + 1
        ] = 1

    return (
        smoothed,
        final_predictions
    )


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

def build_and_load_feature_extractor(
    config: Dict[str, Any],
    device: torch.device,
) -> TwoStreamFeatureExtractor:
    """
    Build the same feature extractor used by
    Experiment H.

    H's detector checkpoint stores the KNN detector.
    The feature extractor is recovered from the
    trained best_model.pt checkpoint.
    """

    model_cfg = config.get(
        "model",
        {}
    )

    rgb_dim = model_cfg.get(
        "rgb_feature_dim",
        512
    )

    flow_dim = model_cfg.get(
        "flow_feature_dim",
        128
    )

    fusion_dim = model_cfg.get(
        "fusion_dim",
        256
    )

    # --------------------------------------------------------
    # Build feature extractor
    # --------------------------------------------------------

    feature_extractor = (
        TwoStreamFeatureExtractor(
            rgb_feature_dim=rgb_dim,
            flow_feature_dim=flow_dim,
            fusion_dim=fusion_dim,
            pretrained=False,
            freeze_rgb_backbone=True,
        )
    )

    feature_extractor = (
        feature_extractor.to(device)
    )

    # --------------------------------------------------------
    # Load feature extractor from best_model.pt
    # --------------------------------------------------------

    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / "best_model.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "The Experiment H checkpoint does not "
            "contain feature extractor weights and "
            f"fallback checkpoint was not found:\n"
            f"{checkpoint_path}"
        )

    print(
        "Loading trained feature extractor "
        f"from: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    # --------------------------------------------------------
    # Extract only feature_extractor.* weights
    # --------------------------------------------------------

    feature_extractor_state = {}

    for key, value in model_state_dict.items():

        if key.startswith(
            "feature_extractor."
        ):

            new_key = key.replace(
                "feature_extractor.",
                "",
                1
            )

            feature_extractor_state[
                new_key
            ] = value

    if not feature_extractor_state:

        raise RuntimeError(
            "Could not find any "
            "'feature_extractor.*' weights "
            "inside best_model.pt."
        )

    missing, unexpected = (
        feature_extractor.load_state_dict(
            feature_extractor_state,
            strict=False
        )
    )

    print(
        "Successfully loaded trained "
        "feature extractor weights."
    )

    if missing:

        print(
            f"Feature extractor missing keys: "
            f"{len(missing)}"
        )

    if unexpected:

        print(
            f"Unexpected feature extractor keys: "
            f"{len(unexpected)}"
        )

    feature_extractor.eval()

    return feature_extractor


# ============================================================
# SPATIAL PSEUDO-LOCALIZATION
# ============================================================

def generate_spatial_masks(
    item: Dict[str, Any],
    frame_predictions: np.ndarray,
    total_native_frames: int,
    spatial_threshold: float,
):
    """
    Generate motion-energy based spatial pseudo-localization.

    These are NOT ground-truth boxes.
    """

    masks = [
        None
        for _ in range(
            total_native_frames
        )
    ]

    if not np.any(
        frame_predictions
    ):
        return masks

    flow_tensor = item.get(
        "flow"
    )

    if flow_tensor is None:
        return masks

    if isinstance(
        flow_tensor,
        torch.Tensor
    ):

        flow_np = (
            flow_tensor
            .detach()
            .cpu()
            .numpy()
        )

    else:

        flow_np = np.asarray(
            flow_tensor
        )

    if flow_np.size == 0:
        return masks

    # --------------------------------------------------------
    # Average temporal flow
    # --------------------------------------------------------

    if flow_np.ndim == 4:

        mean_flow = np.mean(
            flow_np,
            axis=0
        )

    elif flow_np.ndim == 3:

        mean_flow = flow_np

    else:

        return masks

    # --------------------------------------------------------
    # Motion-energy ROI
    # --------------------------------------------------------

    try:

        spatial_localizer = (
            SpatialLocalizer(
                threshold_ratio=
                spatial_threshold
            )
        )

        rois = (
            spatial_localizer
            .extract_motion_rois(
                mean_flow,
                (
                    360,
                    640
                )
            )
        )

    except Exception:

        try:

            rois = (
                SpatialLocalizer
                .extract_motion_rois(
                    mean_flow,
                    (
                        360,
                        640
                    ),
                    threshold=
                    spatial_threshold
                )
            )

        except Exception:

            return masks

    if rois is None:
        return masks

    # --------------------------------------------------------
    # Create combined mask
    # --------------------------------------------------------

    combined_mask = np.zeros(
        (
            360,
            640
        ),
        dtype=np.uint8
    )

    for roi in rois:

        if roi is None:
            continue

        try:

            if len(roi) != 4:
                continue

            x1, y1, x2, y2 = [
                int(round(float(v)))
                for v in roi
            ]

            x1 = max(
                0,
                min(639, x1)
            )

            x2 = max(
                0,
                min(640, x2)
            )

            y1 = max(
                0,
                min(359, y1)
            )

            y2 = max(
                0,
                min(360, y2)
            )

            if (
                x2 > x1
                and y2 > y1
            ):

                combined_mask[
                    y1:y2,
                    x1:x2
                ] = 1

        except Exception:

            continue

    if not np.any(
        combined_mask
    ):
        return masks

    # --------------------------------------------------------
    # Assign mask only to abnormal frames
    # --------------------------------------------------------

    usable_frames = min(
        len(frame_predictions),
        total_native_frames
    )

    for frame_idx in range(
        usable_frames
    ):

        if (
            frame_predictions[
                frame_idx
            ]
            == 1
        ):

            masks[
                frame_idx
            ] = combined_mask.copy()

    return masks


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_avenue_knn(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    max_videos: Optional[int] = None,
    device_name: str = "cpu",
):
    """
    Run Experiment H evaluation.
    """

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    config_path = Path(
        config_path
    )

    if not config_path.is_absolute():

        config_path = (
            PROJECT_ROOT
            / config_path
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_absolute():

        checkpoint_path = (
            PROJECT_ROOT
            / checkpoint_path
        )

    output_dir = Path(
        output_dir
    )

    if not output_dir.is_absolute():

        output_dir = (
            PROJECT_ROOT
            / output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    cfg = load_config(
        config_path
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if (
        device_name == "cuda"
        and torch.cuda.is_available()
    ):

        device = torch.device(
            "cuda"
        )

    elif (
        device_name == "mps"
        and torch.backends.mps.is_available()
    ):

        device = torch.device(
            "mps"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        "=" * 75
    )

    print(
        "EVALUATING CUHK AVENUE FEATURE-MEMORY / "
        "KNN DETECTOR (EXPERIMENT H)"
    )

    print(
        "=" * 75
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load H detector
    # --------------------------------------------------------

    print(
        f"Loading KNN checkpoint: "
        f"{checkpoint_path}"
    )

    detector = (
        AvenueKNNDetector()
    )

    detector.load(
        checkpoint_path
    )

    # --------------------------------------------------------
    # Detector information
    # --------------------------------------------------------

    memory_bank = getattr(
        detector,
        "memory_bank",
        None
    )

    if memory_bank is not None:

        print(
            f"Memory Bank Size          : "
            f"{memory_bank.shape}"
        )

    k_neighbors = getattr(
        detector,
        "k_neighbors",
        5
    )

    print(
        f"K Neighbors               : "
        f"{k_neighbors}"
    )

    threshold = (
        detector.threshold
    )

    if threshold is None:

        raise ValueError(
            "Loaded H checkpoint has "
            "no operating threshold."
        )

    threshold = float(
        threshold
    )

    print(
        "Operating Decision Threshold: "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # Post-processing settings
    # --------------------------------------------------------

    oc_cfg = cfg.get(
        "avenue_oneclass",
        {}
    )

    smoothing_window = int(
        oc_cfg.get(
            "score_smoothing_window",
            5
        )
    )

    min_duration_sec = float(
        oc_cfg.get(
            "min_anomaly_duration_seconds",
            0.5
        )
    )

    merge_gap_sec = float(
        oc_cfg.get(
            "merge_gap_seconds",
            0.5
        )
    )

    print(
        "Temporal Post-Processing:"
    )

    print(
        f"  Smoothing Window          : "
        f"{smoothing_window} frames"
    )

    print(
        f"  Min Anomaly Duration      : "
        f"{min_duration_sec}s"
    )

    print(
        f"  Merge Gap                 : "
        f"{merge_gap_sec}s"
    )

    # --------------------------------------------------------
    # Build feature extractor
    # --------------------------------------------------------

    feature_extractor = (
        build_and_load_feature_extractor(
            cfg,
            device
        )
    )

    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    metadata_path = Path(
        cfg["dataset"][
            "metadata_file"
        ]
    )

    if not metadata_path.is_absolute():

        metadata_path = (
            PROJECT_ROOT
            / metadata_path
        )

    gt_root_path = Path(
        cfg["dataset"][
            "ground_truth_root"
        ]
    )

    if not gt_root_path.is_absolute():

        gt_root_path = (
            PROJECT_ROOT
            / gt_root_path
        )

    # --------------------------------------------------------
    # Avenue dataset
    #
    # IMPORTANT:
    # AvenueDataset does NOT accept ground_truth_root.
    # Ground truth path comes from metadata.
    # --------------------------------------------------------

    test_dataset = AvenueDataset(
        metadata_file=str(
            metadata_path
        ),
        split="test",
        clip_length=cfg[
            "video"
        ].get(
            "clip_length",
            16
        ),
        stride=cfg[
            "video"
        ].get(
            "stride",
            16
        ),
        target_fps=cfg[
            "video"
        ].get(
            "target_fps",
            16
        ),
        image_size=cfg[
            "video"
        ].get(
            "image_size",
            [224, 224]
        ),
        flow_size=cfg[
            "video"
        ].get(
            "flow_size",
            [112, 112]
        ),
        full_video_clips=True,
        use_cache=True
    )

    num_test = len(
        test_dataset
    )

    if max_videos is not None:

        num_test = min(
            int(max_videos),
            num_test
        )

    print()

    print(
        f"Processing {num_test} Avenue test videos "
        "(ground truth withheld during inference)..."
    )

    # --------------------------------------------------------
    # Spatial threshold
    # --------------------------------------------------------

    eval_cfg = cfg.get(
        "evaluation",
        {}
    )

    spatial_threshold = float(
        eval_cfg.get(
            "spatial_threshold",
            0.35
        )
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    video_predictions = []

    all_raw_scores = []

    all_smoothed_scores = []

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    for idx in range(
        num_test
    ):

        item = test_dataset[
            idx
        ]

        vid_id = str(
            item["video_id"]
        ).zfill(2)

        total_native_frames = int(
            item["native_frame_count"]
        )

        native_fps = float(
            item["native_fps"]
        )

        gt_path = item.get(
            "ground_truth_path"
        )

        print()
        print(
            f"[{vid_id}] Processing..."
        )

        # ----------------------------------------------------
        # 1. Feature extraction
        #
        # Ground truth is NOT passed.
        # ----------------------------------------------------

        clip_features = (
            extract_clip_features(
                feature_extractor=
                feature_extractor,

                rgb_tensor=
                item["rgb"],

                flow_tensor=
                item["flow"],

                device=device
            )
        )

        # ----------------------------------------------------
        # 2. KNN anomaly scores
        # ----------------------------------------------------

        clip_scores = (
            detector.predict_score(
                clip_features
            )
        )

        clip_scores = np.asarray(
            clip_scores,
            dtype=np.float32
        )

        all_raw_scores.extend(
            clip_scores.tolist()
        )

        # ----------------------------------------------------
        # 3. Align clip scores to native frames
        # ----------------------------------------------------

        clip_native_ranges = (
            item[
                "clip_native_ranges"
            ]
        )

        frame_scores = (
            align_clip_scores_to_frames(
                clip_scores,
                clip_native_ranges,
                total_native_frames
            )
        )

        frame_scores = np.asarray(
            frame_scores,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # 4. Temporal post-processing
        # ----------------------------------------------------

        (
            smoothed_scores,
            frame_predictions
        ) = temporal_postprocess(
            scores=frame_scores,
            threshold=threshold,
            fps=native_fps,
            smoothing_window=
                smoothing_window,
            min_anomaly_duration_seconds=
                min_duration_sec,
            merge_gap_seconds=
                merge_gap_sec
        )

        all_smoothed_scores.extend(
            smoothed_scores.tolist()
        )

        # ----------------------------------------------------
        # 5. Spatial pseudo-localization
        # ----------------------------------------------------

        spatial_masks = (
            generate_spatial_masks(
                item=item,
                frame_predictions=
                    frame_predictions,
                total_native_frames=
                    total_native_frames,
                spatial_threshold=
                    spatial_threshold
            )
        )

        # ----------------------------------------------------
        # 6. Prediction record
        # ----------------------------------------------------

        video_predictions.append(
            {
                "video_id": vid_id,

                "pred_frame_scores":
                    smoothed_scores,

                "pred_frame_labels":
                    frame_predictions,

                "pred_spatial_masks":
                    spatial_masks,

                "ground_truth_path":
                    gt_path,

                "fps":
                    native_fps,
            }
        )

        predicted_abnormal = int(
            np.sum(
                frame_predictions
            )
        )

        print(
            f"[{vid_id}] "
            f"clips={len(clip_scores):4d} | "
            f"frames={len(frame_scores):5d} | "
            f"predicted abnormal="
            f"{predicted_abnormal:5d} | "
            f"score mean="
            f"{float(np.mean(clip_scores)):.6f} | "
            f"score max="
            f"{float(np.max(clip_scores)):.6f}"
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    if not video_predictions:

        raise RuntimeError(
            "No video predictions were generated."
        )

    print()
    print(
        "=" * 75
    )

    print(
        "Evaluating predictions against "
        "Avenue ground truth..."
    )

    evaluator = AvenueEvaluator(
        ground_truth_root=
            str(gt_root_path),

        default_threshold=
            threshold
    )

    evaluation_result = (
        evaluator.evaluate_test_set(
            video_predictions=
                video_predictions,

            threshold=
                threshold
        )
    )

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    summary = evaluation_result.get(
        "summary",
        {}
    )

    frame_metrics_df = (
        evaluation_result.get(
            "frame_metrics_df"
        )
    )

    video_metrics_df = (
        evaluation_result.get(
            "video_metrics_df"
        )
    )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------

    if frame_metrics_df is not None:

        frame_metrics_df.to_csv(
            output_dir
            / "avenue_knn_frame_metrics.csv",
            index=False
        )

    if video_metrics_df is not None:

        video_metrics_df.to_csv(
            output_dir
            / "avenue_knn_video_metrics.csv",
            index=False
        )

    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    score_statistics = {}

    if all_raw_scores:

        score_statistics = (
            compute_score_statistics(
                np.asarray(
                    all_raw_scores,
                    dtype=np.float32
                )
            )
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    evaluation_metadata = {
        "experiment": "H",

        "method":
            "Normal Feature Memory / "
            "K-Nearest Normal Features",

        "config_file":
            str(
                config_path.resolve()
            ),

        "checkpoint_file":
            str(
                checkpoint_path.resolve()
            ),

        "device":
            str(device),

        "memory_bank_size":
            (
                list(
                    memory_bank.shape
                )
                if memory_bank is not None
                else None
            ),

        "k_neighbors":
            int(k_neighbors),

        "operating_threshold":
            threshold,

        "threshold_protocol":
            "99th percentile of normal "
            "validation videos 13-16",

        "ground_truth_used_during_inference":
            False,

        "temporal_postprocessing": {
            "smoothing_window":
                smoothing_window,

            "min_anomaly_duration_seconds":
                min_duration_sec,

            "merge_gap_seconds":
                merge_gap_sec
        },

        "spatial_localization_type":
            "Motion-Energy Based "
            "Spatial Pseudo-Localization",

        "spatial_ground_truth_boxes":
            False
    }

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {
        "experiment":
            "H",

        "method":
            "Normal Feature Memory / "
            "KNN One-Class Anomaly Detection",

        "summary":
            summary,

        "score_statistics":
            score_statistics,

        "evaluation_metadata":
            evaluation_metadata,

        "video_predictions":
            video_predictions
    }

    report_path = (
        output_dir
        / "avenue_knn_evaluation_report.json"
    )

    # --------------------------------------------------------
    # JSON-safe conversion
    # --------------------------------------------------------

    def make_json_safe(
        obj
    ):

        if isinstance(
            obj,
            dict
        ):

            return {
                str(k):
                    make_json_safe(v)
                for k, v in obj.items()
            }

        if isinstance(
            obj,
            list
        ):

            return [
                make_json_safe(v)
                for v in obj
            ]

        if isinstance(
            obj,
            tuple
        ):

            return [
                make_json_safe(v)
                for v in obj
            ]

        if isinstance(
            obj,
            np.ndarray
        ):

            return obj.tolist()

        if isinstance(
            obj,
            np.integer
        ):

            return int(obj)

        if isinstance(
            obj,
            np.floating
        ):

            return float(obj)

        if isinstance(
            obj,
            torch.Tensor
        ):

            return (
                obj
                .detach()
                .cpu()
                .tolist()
            )

        return obj

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            make_json_safe(
                report
            ),
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with open(
        output_dir
        / "avenue_knn_evaluation_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            make_json_safe(
                evaluation_metadata
            ),
            f,
            indent=2
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 75
    )

    print(
        "EXPERIMENT H EVALUATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Videos evaluated          : "
        f"{len(video_predictions)}"
    )

    def get_metric(
        name,
        default=None
    ):

        if isinstance(
            summary,
            dict
        ):

            return summary.get(
                name,
                default
            )

        return default

    accuracy = get_metric(
        "accuracy"
    )

    precision = get_metric(
        "precision"
    )

    recall = get_metric(
        "recall"
    )

    f1 = get_metric(
        "f1"
    )

    roc_auc = get_metric(
        "roc_auc"
    )

    pr_auc = get_metric(
        "pr_auc"
    )

    temporal_iou = get_metric(
        "temporal_iou"
    )

    spatial_miou = get_metric(
        "spatial_miou"
    )

    if accuracy is not None:

        print(
            f"Frame Accuracy            : "
            f"{float(accuracy) * 100:.2f}%"
        )

    if precision is not None:

        print(
            f"Precision                 : "
            f"{float(precision) * 100:.2f}%"
        )

    if recall is not None:

        print(
            f"Recall                    : "
            f"{float(recall) * 100:.2f}%"
        )

    if f1 is not None:

        print(
            f"F1 Score                  : "
            f"{float(f1) * 100:.2f}%"
        )

    if roc_auc is not None:

        print(
            f"ROC-AUC                   : "
            f"{float(roc_auc):.4f}"
        )

    if pr_auc is not None:

        print(
            f"PR-AUC                    : "
            f"{float(pr_auc):.4f}"
        )

    if temporal_iou is not None:

        print(
            f"Temporal IoU              : "
            f"{float(temporal_iou) * 100:.2f}%"
        )

    if spatial_miou is not None:

        print(
            f"Spatial mIoU              : "
            f"{float(spatial_miou) * 100:.2f}%"
        )

    print(
        f"Decision Threshold        : "
        f"{threshold:.6f}"
    )

    print()
    print(
        "Report saved to:"
    )

    print(
        report_path
    )

    print(
        "=" * 75
    )

    return report


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CUHK Avenue "
            "Experiment H KNN detector."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/avenue.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/avenue/"
            "avenue_knn_best.pt"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "reports/metrics/"
            "avenue_knn"
        )
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=[
            "cpu",
            "cuda",
            "mps"
        ]
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    evaluate_avenue_knn(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_videos=args.max_videos,
        device_name=args.device,
    )