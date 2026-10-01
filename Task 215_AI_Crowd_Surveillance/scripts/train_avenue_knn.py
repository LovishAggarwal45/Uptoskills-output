#!/usr/bin/env python3
"""
Training and Validation Script for Avenue Feature-Memory / kNN
One-Class Anomaly Detection.

Experiment J:
    Improved threshold calibration using normal validation videos only.

Training:
    Normal Avenue videos 01-12

Validation:
    Normal Avenue videos 13-16

Test:
    Avenue test videos are NOT used during training or calibration.

Method:
    1. Extract fused RGB + optical-flow features.
    2. Build a normal feature memory bank.
    3. Calculate kNN cosine-distance anomaly scores.
    4. Evaluate several threshold percentiles using normal validation
       scores only.
    5. Select a stricter operating threshold according to a predefined
       validation-only policy.
    6. Save the detector checkpoint and calibration metadata.

Integrity:
    - No test videos are used for fitting.
    - No test ground-truth masks are used.
    - No test labels are used for threshold selection.
    - Threshold calibration uses normal validation videos only.

Outputs:
    checkpoints/avenue/avenue_knn_experiment_j.pt
    checkpoints/avenue/avenue_knn_experiment_j_metadata.json
    checkpoints/avenue/avenue_knn_calibration_candidates.json

The original Experiment H checkpoint is NOT overwritten.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.avenue_dataset import AvenueDataset
from src.features.feature_extractor import TwoStreamFeatureExtractor
from src.models.avenue_knn import AvenueKNNDetector
from src.models.avenue_oneclass import extract_clip_features


# ============================================================
# Configuration
# ============================================================

DEFAULT_PERCENTILES = [
    99.0,
    99.5,
    99.7,
    99.9,
    100.0
]

DEFAULT_K = 5

# Experiment J selection policy.
#
# We want to reduce false positives on normal validation videos.
# Because validation contains NORMAL videos only, the threshold is
# selected according to the smallest false-positive rate available
# in the predefined candidate set.
#
# 100.0 percentile means the maximum observed validation-normal
# score. It is NOT a claim of 100% accuracy.
DEFAULT_TARGET_VALIDATION_FPR = 0.01


# ============================================================
# Utility Functions
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
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    return config


def safe_float(value: Any) -> float:
    """Convert a value to a finite Python float."""

    value = float(value)

    if not np.isfinite(value):
        return 0.0

    return value


def calculate_threshold_metrics(
    scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Calculate threshold behavior on NORMAL validation data.

    Since every validation sample is normal:

        predicted abnormal = false positive
        predicted normal   = true negative

    No abnormal validation labels are assumed.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64
    ).reshape(-1)

    if len(scores) == 0:
        raise ValueError(
            "Cannot calculate threshold metrics from empty scores."
        )

    predicted_abnormal = scores >= threshold

    fp = int(
        np.sum(predicted_abnormal)
    )

    tn = int(
        len(scores) - fp
    )

    total = int(
        len(scores)
    )

    false_positive_rate = (
        fp / total
        if total > 0
        else 0.0
    )

    false_positive_percentage = (
        false_positive_rate * 100.0
    )

    return {
        "threshold": safe_float(threshold),
        "validation_total_scores": total,
        "false_positives": fp,
        "true_negatives": tn,
        "false_positive_rate": safe_float(
            false_positive_rate
        ),
        "false_positive_percentage": safe_float(
            false_positive_percentage
        )
    }


def build_calibration_candidates(
    validation_scores: np.ndarray,
    percentiles: List[float]
) -> List[Dict[str, Any]]:
    """
    Build threshold candidates from normal validation scores only.
    """

    validation_scores = np.asarray(
        validation_scores,
        dtype=np.float64
    ).reshape(-1)

    if len(validation_scores) == 0:
        raise ValueError(
            "Validation score array is empty."
        )

    candidates = []

    for percentile in percentiles:

        threshold = float(
            np.percentile(
                validation_scores,
                percentile
            )
        )

        metrics = calculate_threshold_metrics(
            validation_scores,
            threshold
        )

        candidate = {
            "percentile": safe_float(
                percentile
            ),
            **metrics
        }

        candidates.append(
            candidate
        )

    return candidates


def select_threshold(
    candidates: List[Dict[str, Any]],
    target_validation_fpr: float
) -> Dict[str, Any]:
    """
    Select an operating threshold using validation-normal behavior only.

    Selection policy:

    1. Prefer candidates whose validation FPR is <= target FPR.
    2. Among those candidates, choose the lowest threshold.

    If no candidate reaches the requested FPR, choose the candidate
    with the smallest validation FPR. Ties are resolved by choosing
    the lower threshold.

    This policy never examines Avenue test labels.
    """

    if not candidates:
        raise ValueError(
            "No calibration candidates were generated."
        )

    target_validation_fpr = float(
        target_validation_fpr
    )

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["false_positive_rate"]
        <= target_validation_fpr
    ]

    if valid_candidates:

        selected = min(
            valid_candidates,
            key=lambda x: (
                x["threshold"],
                x["false_positive_rate"]
            )
        )

        selection_reason = (
            "Selected the lowest threshold among "
            "predefined candidates satisfying the "
            "target validation false-positive rate."
        )

    else:

        selected = min(
            candidates,
            key=lambda x: (
                x["false_positive_rate"],
                x["threshold"]
            )
        )

        selection_reason = (
            "No predefined candidate satisfied the "
            "target validation false-positive rate. "
            "Selected the candidate with the lowest "
            "validation false-positive rate."
        )

    selected = dict(selected)

    selected["selection_policy"] = (
        "target_validation_fpr"
    )

    selected["target_validation_fpr"] = safe_float(
        target_validation_fpr
    )

    selected["selection_reason"] = (
        selection_reason
    )

    return selected


# ============================================================
# Feature Extractor
# ============================================================

def build_feature_extractor(
    config: Dict[str, Any],
    device: torch.device
) -> TwoStreamFeatureExtractor:
    """
    Build the frozen feature extractor used by Experiment J.

    The feature extractor architecture remains compatible with
    the existing Avenue Experiment H pipeline.

    When checkpoints/best_model.pt exists, trained
    feature_extractor.* weights are loaded from it.
    Otherwise, ImageNet-pretrained ResNet18 weights are used.
    """

    model_cfg = config.get(
        "model",
        {}
    )

    rgb_dim = int(
        model_cfg.get(
            "rgb_feature_dim",
            512
        )
    )

    flow_dim = int(
        model_cfg.get(
            "flow_feature_dim",
            128
        )
    )

    fusion_dim = int(
        model_cfg.get(
            "fusion_dim",
            256
        )
    )

    pretrained = bool(
        model_cfg.get(
            "pretrained",
            True
        )
    )

    freeze_rgb_backbone = bool(
        model_cfg.get(
            "freeze_rgb_backbone",
            True
        )
    )

    feature_extractor = TwoStreamFeatureExtractor(
        rgb_feature_dim=rgb_dim,
        flow_feature_dim=flow_dim,
        fusion_dim=fusion_dim,
        pretrained=pretrained,
        freeze_rgb_backbone=freeze_rgb_backbone
    )

    ckpt_path = (
        PROJECT_ROOT /
        "checkpoints" /
        "best_model.pt"
    )

    if ckpt_path.exists():

        try:

            print(
                f"Loading feature extractor weights "
                f"from {ckpt_path.name}..."
            )

            checkpoint = torch.load(
                str(ckpt_path),
                map_location="cpu"
            )

            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint
            )

            feature_extractor_state = {
                key.replace(
                    "feature_extractor.",
                    ""
                ): value
                for key, value in state_dict.items()
                if key.startswith(
                    "feature_extractor."
                )
            }

            if feature_extractor_state:

                feature_extractor.load_state_dict(
                    feature_extractor_state,
                    strict=True
                )

                print(
                    "Successfully loaded trained "
                    "feature extractor weights."
                )

            else:

                print(
                    "[NOTE] No feature_extractor.* "
                    "weights found in checkpoint."
                )

        except Exception as exc:

            print(
                "[NOTE] Could not load checkpoint "
                f"weights ({exc})."
            )

            print(
                "       Continuing with the configured "
                "feature extractor initialization."
            )

    else:

        print(
            "[NOTE] checkpoints/best_model.pt was not found."
        )

        print(
            "       Continuing with the configured "
            "feature extractor initialization."
        )

    feature_extractor.to(
        device
    )

    feature_extractor.eval()

    return feature_extractor


# ============================================================
# Dataset Helpers
# ============================================================

def build_avenue_dataset(
    config: Dict[str, Any],
    metadata_path: Path
) -> AvenueDataset:
    """
    Build the Avenue dataset using the project's existing
    AvenueDataset interface.
    """

    video_cfg = config.get(
        "video",
        {}
    )

    dataset = AvenueDataset(
        metadata_file=str(
            metadata_path
        ),
        split="train",
        clip_length=int(
            video_cfg.get(
                "clip_length",
                16
            )
        ),
        stride=int(
            video_cfg.get(
                "stride",
                16
            )
        ),
        target_fps=int(
            video_cfg.get(
                "target_fps",
                16
            )
        ),
        image_size=video_cfg.get(
            "image_size",
            [224, 224]
        ),
        flow_size=video_cfg.get(
            "flow_size",
            [112, 112]
        ),
        full_video_clips=True,
        use_cache=True
    )

    return dataset


# ============================================================
# Feature Extraction
# ============================================================

def extract_features_for_video(
    dataset: AvenueDataset,
    video_id: str,
    feature_extractor: TwoStreamFeatureExtractor,
    device: torch.device
) -> np.ndarray:
    """
    Extract fused features for one Avenue video.
    """

    target_id = str(
        video_id
    ).zfill(2)

    for idx in range(
        len(dataset)
    ):

        item = dataset[idx]

        current_id = str(
            item["video_id"]
        ).zfill(2)

        if current_id != target_id:
            continue

        features = extract_clip_features(
            feature_extractor=feature_extractor,
            rgb_tensor=item["rgb"],
            flow_tensor=item["flow"],
            device=device
        )

        return features

    raise RuntimeError(
        f"Video {target_id} was not found in the "
        "Avenue dataset metadata."
    )


# ============================================================
# Main Training Function
# ============================================================

def train_avenue_knn(
    config_path: str = "configs/avenue.yaml",
    output_dir: str = "checkpoints/avenue",
    device_name: str = "auto"
) -> Dict[str, Any]:
    """
    Train Experiment J.

    Experiment H is preserved because this script writes to new
    Experiment J checkpoint/metadata filenames.
    """

    print("=" * 80)

    print(
        "CUHK AVENUE FEATURE-MEMORY / KNN "
        "ONE-CLASS ANOMALY DETECTION"
    )

    print(
        "EXPERIMENT J - VALIDATION-ONLY THRESHOLD CALIBRATION"
    )

    print("=" * 80)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config_p = Path(
        config_path
    )

    if not config_p.is_absolute():

        config_p = (
            PROJECT_ROOT /
            config_p
        )

    cfg = load_config(
        config_p
    )

    output_dir_p = Path(
        output_dir
    )

    if not output_dir_p.is_absolute():

        output_dir_p = (
            PROJECT_ROOT /
            output_dir_p
        )

    output_dir_p.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = Path(
        cfg["dataset"]["metadata_file"]
    )

    if not metadata_path.is_absolute():

        metadata_path = (
            PROJECT_ROOT /
            metadata_path
        )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if device_name == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device_name
        )

    print(
        f"Computation Device        : {device}"
    )

    # ---------------------------------------------------------
    # Experiment configuration
    # ---------------------------------------------------------

    oc_cfg = cfg.get(
        "avenue_oneclass",
        {}
    )

    train_ids = [
        str(video_id).zfill(2)
        for video_id in oc_cfg.get(
            "train_video_ids",
            [
                "01", "02", "03", "04",
                "05", "06", "07", "08",
                "09", "10", "11", "12"
            ]
        )
    ]

    validation_ids = [
        str(video_id).zfill(2)
        for video_id in oc_cfg.get(
            "validation_video_ids",
            [
                "13", "14", "15", "16"
            ]
        )
    ]

    k_neighbors = int(
        oc_cfg.get(
            "knn_k",
            DEFAULT_K
        )
    )

    configured_percentile = float(
        oc_cfg.get(
            "validation_normal_percentile",
            99.0
        )
    )

    configured_target_fpr = float(
        oc_cfg.get(
            "target_validation_fpr",
            DEFAULT_TARGET_VALIDATION_FPR
        )
    )

    configured_percentiles = oc_cfg.get(
        "calibration_percentiles",
        DEFAULT_PERCENTILES
    )

    percentiles = sorted(
        set(
            float(value)
            for value in configured_percentiles
        )
    )

    # Always preserve the original H 99th percentile as a
    # candidate for direct comparison.
    if configured_percentile not in percentiles:

        percentiles.append(
            configured_percentile
        )

        percentiles = sorted(
            set(percentiles)
        )

    print(
        f"Normal Training Videos    : "
        f"{train_ids} ({len(train_ids)} videos)"
    )

    print(
        f"Normal Validation Videos  : "
        f"{validation_ids} ({len(validation_ids)} videos)"
    )

    print(
        f"K Nearest Normal Features : "
        f"{k_neighbors}"
    )

    print(
        f"Calibration Percentiles   : "
        f"{percentiles}"
    )

    print(
        f"Target Validation FPR     : "
        f"{configured_target_fpr:.4f} "
        f"({configured_target_fpr * 100.0:.2f}%)"
    )

    print("-" * 80)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print(
        "\nBuilding Avenue dataset..."
    )

    dataset = build_avenue_dataset(
        config=cfg,
        metadata_path=metadata_path
    )

    print(
        f"Dataset clips available   : "
        f"{len(dataset)}"
    )

    # ---------------------------------------------------------
    # Feature extractor
    # ---------------------------------------------------------

    print(
        "\nBuilding feature extractor..."
    )

    feature_extractor = build_feature_extractor(
        config=cfg,
        device=device
    )

    # ---------------------------------------------------------
    # 1. Extract normal training features
    # ---------------------------------------------------------

    print(
        "\n[1/5] Extracting normal training features "
        "from Avenue videos 01-12..."
    )

    train_features_list = []

    train_clip_counts = {}

    for video_id in train_ids:

        features = extract_features_for_video(
            dataset=dataset,
            video_id=video_id,
            feature_extractor=feature_extractor,
            device=device
        )

        train_features_list.append(
            features
        )

        train_clip_counts[
            video_id
        ] = int(
            len(features)
        )

        print(
            f"  Train video {video_id}: "
            f"{len(features):4d} clips | "
            f"feature shape: {features.shape}"
        )

    if not train_features_list:

        raise RuntimeError(
            "No training features could be extracted."
        )

    all_train_features = np.concatenate(
        train_features_list,
        axis=0
    )

    print(
        f"\nTotal normal training clips: "
        f"{len(all_train_features)}"
    )

    print(
        f"Feature dimension          : "
        f"{all_train_features.shape[1]}"
    )

    # ---------------------------------------------------------
    # 2. Build normal memory bank
    # ---------------------------------------------------------

    print(
        "\n[2/5] Building normal feature memory bank..."
    )

    detector = AvenueKNNDetector(
        feature_dim=int(
            all_train_features.shape[1]
        ),
        k_neighbors=k_neighbors
    )

    train_stats = detector.fit(
        all_train_features
    )

    print(
        f"  Memory bank size         : "
        f"{detector.memory_bank.shape}"
    )

    print(
        f"  Effective K              : "
        f"{detector.k_neighbors}"
    )

    print(
        f"  Training score mean      : "
        f"{train_stats['mean']:.6f}"
    )

    print(
        f"  Training score std       : "
        f"{train_stats['std']:.6f}"
    )

    print(
        f"  Training score range     : "
        f"[{train_stats['min']:.6f}, "
        f"{train_stats['max']:.6f}]"
    )

    print(
        f"  Training score p99       : "
        f"{train_stats['p99']:.6f}"
    )

    # ---------------------------------------------------------
    # 3. Extract NORMAL validation scores
    # ---------------------------------------------------------

    print(
        "\n[3/5] Extracting normal validation scores "
        "from Avenue videos 13-16..."
    )

    validation_scores_list = []

    validation_clip_counts = {}

    validation_video_statistics = {}

    for video_id in validation_ids:

        features = extract_features_for_video(
            dataset=dataset,
            video_id=video_id,
            feature_extractor=feature_extractor,
            device=device
        )

        scores = detector.predict_score(
            features
        )

        scores = np.asarray(
            scores,
            dtype=np.float64
        )

        validation_scores_list.append(
            scores
        )

        validation_clip_counts[
            video_id
        ] = int(
            len(scores)
        )

        video_stats = {
            "video_id": video_id,
            "clips": int(
                len(scores)
            ),
            "mean": safe_float(
                np.mean(scores)
            ),
            "std": safe_float(
                np.std(scores)
            ),
            "min": safe_float(
                np.min(scores)
            ),
            "max": safe_float(
                np.max(scores)
            ),
            "p95": safe_float(
                np.percentile(
                    scores,
                    95
                )
            ),
            "p99": safe_float(
                np.percentile(
                    scores,
                    99
                )
            )
        }

        validation_video_statistics[
            video_id
        ] = video_stats

        print(
            f"  Validation video {video_id}: "
            f"{len(scores):4d} clips | "
            f"mean={video_stats['mean']:.6f} | "
            f"std={video_stats['std']:.6f} | "
            f"max={video_stats['max']:.6f}"
        )

    if not validation_scores_list:

        raise RuntimeError(
            "No validation features could be extracted."
        )

    all_validation_scores = np.concatenate(
        validation_scores_list,
        axis=0
    )

    print(
        f"\nTotal normal validation clips: "
        f"{len(all_validation_scores)}"
    )

    print(
        f"Validation score mean          : "
        f"{np.mean(all_validation_scores):.6f}"
    )

    print(
        f"Validation score std           : "
        f"{np.std(all_validation_scores):.6f}"
    )

    print(
        f"Validation score range         : "
        f"[{np.min(all_validation_scores):.6f}, "
        f"{np.max(all_validation_scores):.6f}]"
    )

    # ---------------------------------------------------------
    # 4. Validation-only calibration
    # ---------------------------------------------------------

    print(
        "\n[4/5] Calibrating candidate thresholds "
        "using NORMAL validation videos only..."
    )

    candidates = build_calibration_candidates(
        validation_scores=all_validation_scores,
        percentiles=percentiles
    )

    print(
        "\nCalibration Candidate Results"
    )

    print(
        "-" * 80
    )

    print(
        f"{'Percentile':>12} | "
        f"{'Threshold':>12} | "
        f"{'FP':>8} | "
        f"{'TN':>8} | "
        f"{'Val FPR':>10}"
    )

    print(
        "-" * 80
    )

    for candidate in candidates:

        print(
            f"{candidate['percentile']:>11.2f}% | "
            f"{candidate['threshold']:>12.6f} | "
            f"{candidate['false_positives']:>8d} | "
            f"{candidate['true_negatives']:>8d} | "
            f"{candidate['false_positive_percentage']:>9.3f}%"
        )

    print(
        "-" * 80
    )

    selected = select_threshold(
        candidates=candidates,
        target_validation_fpr=configured_target_fpr
    )

    selected_threshold = float(
        selected["threshold"]
    )

    print(
        "\nSelected Experiment J Operating Point"
    )

    print(
        f"  Percentile               : "
        f"{selected['percentile']:.2f}%"
    )

    print(
        f"  Threshold                : "
        f"{selected_threshold:.6f}"
    )

    print(
        f"  Validation false positives: "
        f"{selected['false_positives']}"
    )

    print(
        f"  Validation false-positive rate: "
        f"{selected['false_positive_percentage']:.3f}%"
    )

    print(
        f"  Selection policy         : "
        f"{selected['selection_policy']}"
    )

    print(
        f"  Selection reason         : "
        f"{selected['selection_reason']}"
    )

    # ---------------------------------------------------------
    # Important:
    # ---------------------------------------------------------
    # The detector itself remains the same. We only change its
    # operating threshold after validation calibration.
    # ---------------------------------------------------------

    detector.threshold = selected_threshold

    detector.calibration_percentile = float(
        selected["percentile"]
    )

    # ---------------------------------------------------------
    # 5. Save Experiment J checkpoint
    # ---------------------------------------------------------

    print(
        "\n[5/5] Saving Experiment J checkpoint..."
    )

    checkpoint_file = (
        output_dir_p /
        "avenue_knn_experiment_j.pt"
    )

    metadata_file = (
        output_dir_p /
        "avenue_knn_experiment_j_metadata.json"
    )

    calibration_file = (
        output_dir_p /
        "avenue_knn_calibration_candidates.json"
    )

    # ---------------------------------------------------------
    # Save detector
    # ---------------------------------------------------------

    detector.save(
        checkpoint_path=checkpoint_file,
        extra_metadata={
            "feature_extractor_state_dict":
                feature_extractor.state_dict(),

            "config":
                oc_cfg,

            "experiment":
                "J",

            "experiment_name":
                "Validation-only threshold calibration",

            "training_video_ids":
                train_ids,

            "validation_video_ids":
                validation_ids,

            "calibration_percentiles":
                percentiles,

            "selected_percentile":
                float(
                    selected["percentile"]
                ),

            "selected_threshold":
                selected_threshold,

            "target_validation_fpr":
                configured_target_fpr,

            "calibration_candidates":
                candidates
        }
    )

    # ---------------------------------------------------------
    # Save calibration report
    # ---------------------------------------------------------

    calibration_payload = {

        "experiment":
            "J",

        "description":
            "Validation-only threshold calibration "
            "for Avenue normal-only kNN anomaly detection.",

        "training_video_ids":
            train_ids,

        "validation_video_ids":
            validation_ids,

        "k_neighbors":
            int(
                detector.k_neighbors
            ),

        "validation_total_scores":
            int(
                len(all_validation_scores)
            ),

        "validation_score_statistics":
            {
                "mean":
                    safe_float(
                        np.mean(
                            all_validation_scores
                        )
                    ),

                "std":
                    safe_float(
                        np.std(
                            all_validation_scores
                        )
                    ),

                "min":
                    safe_float(
                        np.min(
                            all_validation_scores
                        )
                    ),

                "max":
                    safe_float(
                        np.max(
                            all_validation_scores
                        )
                    ),

                "p95":
                    safe_float(
                        np.percentile(
                            all_validation_scores,
                            95
                        )
                    ),

                "p99":
                    safe_float(
                        np.percentile(
                            all_validation_scores,
                            99
                        )
                    )
            },

        "candidate_percentiles":
            percentiles,

        "candidate_results":
            candidates,

        "selected_operating_point":
            selected,

        "validation_video_statistics":
            validation_video_statistics,

        "integrity_notice":
            (
                "All threshold candidates and the selected "
                "operating point were determined using normal "
                "validation videos only. Avenue test videos "
                "and test ground-truth masks were not used."
            )
    }

    with open(
        calibration_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            calibration_payload,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # Save main metadata
    # ---------------------------------------------------------

    metadata_payload = {

        "dataset":
            "CUHK Avenue",

        "method":
            (
                "Normal Feature Memory "
                "K-Nearest-Neighbor Cosine Distance"
            ),

        "experiment":
            "Experiment J",

        "experiment_name":
            (
                "Validation-only threshold calibration"
            ),

        "training_paradigm":
            (
                "Normal-only unsupervised anomaly detection"
            ),

        "train_video_ids":
            train_ids,

        "train_clip_counts":
            train_clip_counts,

        "total_train_clips":
            int(
                len(all_train_features)
            ),

        "validation_video_ids":
            validation_ids,

        "validation_clip_counts":
            validation_clip_counts,

        "total_validation_clips":
            int(
                len(all_validation_scores)
            ),

        "feature_dimension":
            int(
                all_train_features.shape[1]
            ),

        "k_neighbors":
            int(
                detector.k_neighbors
            ),

        "calibration_percentiles":
            percentiles,

        "selected_calibration_percentile":
            float(
                selected["percentile"]
            ),

        "calibrated_threshold":
            safe_float(
                selected_threshold
            ),

        "target_validation_fpr":
            safe_float(
                configured_target_fpr
            ),

        "validation_false_positive_rate":
            safe_float(
                selected["false_positive_rate"]
            ),

        "validation_false_positive_percentage":
            safe_float(
                selected[
                    "false_positive_percentage"
                ]
            ),

        "normal_training_stats":
            {
                key:
                    round(value, 6)
                    if isinstance(
                        value,
                        (float, np.floating)
                    )
                    else value
                for key, value
                in train_stats.items()
            },

        "normal_validation_stats":
            {
                "mean":
                    safe_float(
                        np.mean(
                            all_validation_scores
                        )
                    ),

                "std":
                    safe_float(
                        np.std(
                            all_validation_scores
                        )
                    ),

                "min":
                    safe_float(
                        np.min(
                            all_validation_scores
                        )
                    ),

                "max":
                    safe_float(
                        np.max(
                            all_validation_scores
                        )
                    ),

                "p95":
                    safe_float(
                        np.percentile(
                            all_validation_scores,
                            95
                        )
                    ),

                "p99":
                    safe_float(
                        np.percentile(
                            all_validation_scores,
                            99
                        )
                    )
            },

        "calibration_candidates":
            candidates,

        "selected_operating_point":
            selected,

        "device":
            str(device),

        "configuration":
            oc_cfg,

        "integrity_notice":
            (
                "Threshold selected using normal validation "
                "videos 13-16 only. No Avenue test ground "
                "truth was used for calibration."
            )
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata_payload,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENT J TRAINING AND CALIBRATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"  Training Videos          : "
        f"{train_ids}"
    )

    print(
        f"  Validation Videos        : "
        f"{validation_ids}"
    )

    print(
        f"  Training Clips           : "
        f"{len(all_train_features)}"
    )

    print(
        f"  Validation Clips         : "
        f"{len(all_validation_scores)}"
    )

    print(
        f"  Feature Dimension        : "
        f"{all_train_features.shape[1]}"
    )

    print(
        f"  K Neighbors              : "
        f"{detector.k_neighbors}"
    )

    print(
        f"  Selected Percentile      : "
        f"{selected['percentile']:.2f}%"
    )

    print(
        f"  Selected Threshold       : "
        f"{selected_threshold:.6f}"
    )

    print(
        f"  Validation FPR           : "
        f"{selected['false_positive_percentage']:.3f}%"
    )

    print(
        f"  Detector Checkpoint      : "
        f"{checkpoint_file}"
    )

    print(
        f"  Metadata                 : "
        f"{metadata_file}"
    )

    print(
        f"  Calibration Report       : "
        f"{calibration_file}"
    )

    print(
        "\n  Test ground truth used?  : NO"
    )

    print(
        "  Test labels used for "
        "threshold selection?     : NO"
    )

    print(
        "=" * 80
    )

    return metadata_payload


# ============================================================
# Command-Line Interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Avenue Feature-Memory / kNN "
            "One-Class Detector with validation-only "
            "threshold calibration."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/avenue.yaml",
        help=(
            "Path to Avenue configuration YAML."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/avenue",
        help=(
            "Directory for Experiment J outputs."
        )
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device: auto, cpu, or cuda."
        )
    )

    args = parser.parse_args()

    train_avenue_knn(
        config_path=args.config,
        output_dir=args.output_dir,
        device_name=args.device
    )


if __name__ == "__main__":
    main()