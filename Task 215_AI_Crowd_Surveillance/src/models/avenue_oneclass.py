"""
Avenue One-Class / Unsupervised Anomaly Detection Module.

Because the CUHK Avenue training set contains ONLY normal videos (01-16),
ordinary supervised or MIL-ranking objectives cannot be trained on Avenue directly.

This module implements a lightweight, numerically stable one-class anomaly detector:
1. Feature vectors from normal training clips are L2-normalized.
2. A normal centroid / prototype is computed and L2-normalized.
3. Anomaly scores are computed via cosine distance:
       anomaly_score = 1.0 - cosine_similarity(feature, normal_centroid)
4. An internal normal-only validation procedure (videos 13-16) calibrates the decision
   threshold using a high normal percentile (e.g. 99th percentile) without inspecting test labels.
5. Configurable temporal post-processing (moving-average smoothing, min-duration filtering,
   and gap merging) produces final binary frame predictions.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter1d

from src.features.feature_extractor import TwoStreamFeatureExtractor


def l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalizes feature vectors along the last axis.
    Input shape: [N, D] or [D]
    """
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return features / norm


def compute_score_statistics(scores: np.ndarray) -> Dict[str, float]:
    """Computes descriptive statistics for a 1D array of scores."""
    if len(scores) == 0:
        return {}
    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "p50": float(np.percentile(scores, 50.0)),
        "p75": float(np.percentile(scores, 75.0)),
        "p90": float(np.percentile(scores, 90.0)),
        "p95": float(np.percentile(scores, 95.0)),
        "p99": float(np.percentile(scores, 99.0)),
    }


class AvenueOneClassDetector:
    """
    One-Class prototype anomaly detector based on cosine distance in fused feature space.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        threshold: Optional[float] = None
    ):
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.centroid: Optional[np.ndarray] = None  # Normalized centroid [D]
        self.normal_training_stats: Dict[str, float] = {}
        self.normal_validation_stats: Dict[str, float] = {}
        self.calibration_percentile: Optional[float] = None

    def fit(self, training_features: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
        """
        Fits the normal prototype using normal-only training clip features.

        training_features: np.ndarray of shape [N, D]
        """
        if len(training_features) == 0:
            raise ValueError("Cannot fit detector with 0 training features.")

        if training_features.ndim != 2:
            raise ValueError(f"Expected 2D array [N, D], got shape {training_features.shape}")

        self.feature_dim = training_features.shape[1]

        # 1. L2-normalize each feature vector
        norm_features = l2_normalize(training_features.astype(np.float32), eps=eps)

        # 2. Compute normal centroid
        raw_centroid = np.mean(norm_features, axis=0)

        # 3. L2-normalize the centroid
        self.centroid = l2_normalize(raw_centroid, eps=eps)

        # 4. Compute cosine distance on training set: d = 1 - cosine_similarity
        sims = np.dot(norm_features, self.centroid)
        distances = np.clip(1.0 - sims, 0.0, 2.0)

        # 5. Store distance statistics
        self.normal_training_stats = compute_score_statistics(distances)
        return self.normal_training_stats

    def predict_score(self, features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Computes anomaly scores (cosine distance from normal centroid) for feature vectors.
        Higher score = further from normal prototype = more anomalous.

        Returns 1D array of shape [N] in [0.0, 2.0].
        """
        if self.centroid is None:
            raise RuntimeError("Detector must be fitted before predicting scores.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        norm_features = l2_normalize(features.astype(np.float32), eps=eps)
        sims = np.dot(norm_features, self.centroid)
        scores = np.clip(1.0 - sims, 0.0, 2.0)
        return scores.astype(np.float32)

    def calibrate_threshold(
        self,
        validation_scores: np.ndarray,
        percentile: float = 99.0
    ) -> float:
        """
        Determines the decision threshold using normal-only validation scores.
        Threshold is set to the specified percentile of the normal validation distribution.
        """
        if len(validation_scores) == 0:
            raise ValueError("Cannot calibrate threshold with empty validation scores.")

        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"Percentile must be in (0, 100], got {percentile}")

        self.calibration_percentile = percentile
        self.threshold = float(np.percentile(validation_scores, percentile))
        self.normal_validation_stats = compute_score_statistics(validation_scores)
        return self.threshold

    @staticmethod
    def smooth_scores(scores: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Applies 1D moving-average smoothing over 1D continuous frame anomaly scores.
        Preserves score range and boundary behavior with 'nearest' padding.
        """
        if len(scores) <= 1 or window_size <= 1:
            return scores.copy()

        w = min(window_size, len(scores))
        smoothed = uniform_filter1d(scores.astype(np.float32), size=w, mode="nearest")
        return smoothed.astype(np.float32)

    def temporal_postprocess(
        self,
        frame_scores: np.ndarray,
        threshold: Optional[float] = None,
        smoothing_window: int = 5,
        min_anomaly_duration_seconds: float = 0.5,
        merge_gap_seconds: float = 0.5,
        fps: float = 25.0
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Applies temporal post-processing to raw continuous frame scores:
        1. Moving-average smoothing
        2. Thresholding: smoothed >= threshold
        3. Contiguous interval extraction
        4. Gap merging: merge intervals separated by <= merge_gap_seconds
        5. Min duration filtering: drop intervals shorter than min_anomaly_duration_seconds

        Returns:
          - smoothed_scores: 1D array of shape [N]
          - binary_predictions: 1D int32 array of shape [N] (0 = normal, 1 = abnormal)
          - surviving_intervals: list of interval dicts
        """
        thresh = self.threshold if threshold is None else threshold
        if thresh is None:
            raise ValueError("Decision threshold is not set. Calibrate or specify threshold.")

        total_frames = len(frame_scores)
        if total_frames == 0:
            return np.array([]), np.array([]), []

        # 1. Moving-average smoothing
        smoothed = self.smooth_scores(frame_scores, window_size=smoothing_window)

        # 2. Raw candidate mask
        candidates = (smoothed >= thresh)

        # 3. Extract contiguous candidate intervals
        raw_intervals = []
        in_interval = False
        start_idx = 0
        for i, val in enumerate(candidates):
            if val and not in_interval:
                in_interval = True
                start_idx = i
            elif not val and in_interval:
                in_interval = False
                raw_intervals.append((start_idx, i - 1))
        if in_interval:
            raw_intervals.append((start_idx, total_frames - 1))

        if not raw_intervals:
            return smoothed, np.zeros(total_frames, dtype=np.int32), []

        # 4. Merge intervals separated by <= merge_gap_frames
        merge_gap_frames = int(round(merge_gap_seconds * fps))
        merged_intervals = []
        curr_start, curr_end = raw_intervals[0]

        for nxt_start, nxt_end in raw_intervals[1:]:
            gap = nxt_start - curr_end - 1
            if gap <= merge_gap_frames:
                curr_end = nxt_end
            else:
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = nxt_start, nxt_end
        merged_intervals.append((curr_start, curr_end))

        # 5. Filter by min_duration_frames
        min_duration_frames = int(round(min_anomaly_duration_seconds * fps))
        surviving_intervals = []
        binary_predictions = np.zeros(total_frames, dtype=np.int32)

        for s_idx, e_idx in merged_intervals:
            span = e_idx - s_idx + 1
            if span >= min_duration_frames:
                binary_predictions[s_idx:e_idx + 1] = 1
                surviving_intervals.append({
                    "start_frame": int(s_idx),
                    "end_frame": int(e_idx),
                    "frame_span": int(span),
                    "start_time": round(float(s_idx / fps), 3),
                    "end_time": round(float((e_idx + 1) / fps), 3),
                    "duration_seconds": round(float(span / fps), 3)
                })

        return smoothed, binary_predictions, surviving_intervals

    def save(
        self,
        checkpoint_path: Union[str, Path],
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """Saves the detector state dictionary to disk."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "centroid": torch.from_numpy(self.centroid) if self.centroid is not None else None,
            "feature_dim": self.feature_dim,
            "threshold": self.threshold,
            "calibration_percentile": self.calibration_percentile,
            "normal_training_stats": self.normal_training_stats,
            "normal_validation_stats": self.normal_validation_stats,
            "extra_metadata": extra_metadata or {}
        }
        torch.save(payload, str(path))

    def load(self, checkpoint_path: Union[str, Path]):
        """Loads detector state dictionary from disk."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        payload = torch.load(str(path), map_location="cpu")
        c = payload.get("centroid")
        if c is not None:
            self.centroid = c.numpy() if isinstance(c, torch.Tensor) else np.array(c)
        else:
            self.centroid = None

        self.feature_dim = payload.get("feature_dim", 256)
        self.threshold = payload.get("threshold")
        self.calibration_percentile = payload.get("calibration_percentile")
        self.normal_training_stats = payload.get("normal_training_stats", {})
        self.normal_validation_stats = payload.get("normal_validation_stats", {})
        return payload.get("extra_metadata", {})


def extract_clip_features(
    feature_extractor: TwoStreamFeatureExtractor,
    rgb_tensor: torch.Tensor,
    flow_tensor: torch.Tensor,
    device: torch.device,
    chunk_size: int = 16
) -> np.ndarray:
    """
    Extracts fused two-stream feature vectors for sequential clips of a video.

    rgb_tensor:  [num_clips, T, 3, H, W]
    flow_tensor: [num_clips, T, 2, H_f, W_f]
    Returns: np.ndarray of shape [num_clips, fusion_dim]
    """
    total_clips = rgb_tensor.size(0)
    features_list = []

    feature_extractor.to(device)
    feature_extractor.eval()

    with torch.no_grad():
        for i in range(0, total_clips, chunk_size):
            r_c = rgb_tensor[i:i + chunk_size].unsqueeze(0).to(device) # [1, C, T, 3, H, W]
            f_c = flow_tensor[i:i + chunk_size].unsqueeze(0).to(device) # [1, C, T, 2, H_f, W_f]
            fused = feature_extractor(r_c, f_c) # [1, C, fusion_dim]
            features_list.append(fused.squeeze(0).cpu().numpy())

    return np.concatenate(features_list, axis=0) # [total_clips, fusion_dim]
