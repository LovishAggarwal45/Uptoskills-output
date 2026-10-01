#!/usr/bin/env python3
"""
Experiment I
CUHK Avenue:
Experiment H + Motion-Energy Spatial Pseudo-Localization

Purpose
-------
Evaluate the Avenue one-class KNN detector from Experiment H while adding
motion-energy based spatial pseudo-localization.

Important scientific note
-------------------------
The spatial localization produced by this experiment is NOT ground-truth
person detection. It is a motion-energy pseudo-localization derived from
optical flow.

Ground-truth Avenue masks are used ONLY by the evaluator for measuring
spatial performance. They are never used to generate predictions.

Usage
-----
python -m scripts.evaluate_avenue_knn_spatial ^
    --config configs/avenue.yaml ^
    --checkpoint checkpoints/avenue/avenue_knn_best.pt ^
    --output-dir reports/metrics/avenue_knn_spatial_test ^
    --device cpu ^
    --max-videos 1

Full evaluation
---------------
python -m scripts.evaluate_avenue_knn_spatial ^
    --config configs/avenue.yaml ^
    --checkpoint checkpoints/avenue/avenue_knn_best.pt ^
    --output-dir reports/metrics/avenue_knn_spatial_test ^
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.data.avenue_dataset import AvenueDataset
from src.evaluation.avenue_evaluator import (
    AvenueEvaluator,
    align_clip_scores_to_frames,
)
from src.features.feature_extractor import TwoStreamFeatureExtractor
from src.localization.spatial_localizer import SpatialLocalizer
from src.models.avenue_knn import AvenueKNNDetector
from src.models.avenue_oneclass import extract_clip_features


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def resolve_path(path_value: str | Path) -> Path:
    """
    Resolve a project-relative path or preserve an absolute Windows path.
    """

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds where practical."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: Path) -> None:
    """Create directory if required."""

    path.mkdir(parents=True, exist_ok=True)


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensors/lists to NumPy arrays."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


# ---------------------------------------------------------------------------
# Experiment H-compatible temporal processing
# ---------------------------------------------------------------------------

def smooth_scores(
    scores: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Smooth anomaly scores using a centered moving average.

    The implementation is deliberately simple and deterministic so that
    Experiment I remains compatible with Experiment H.
    """

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if scores.size == 0:
        return scores

    window = int(window)

    if window <= 1:
        return scores.copy()

    if window % 2 == 0:
        window += 1

    if window > scores.size:
        window = scores.size

        if window % 2 == 0:
            window -= 1

    if window <= 1:
        return scores.copy()

    kernel = np.ones(window, dtype=np.float32) / float(window)

    pad = window // 2

    padded = np.pad(
        scores,
        (pad, pad),
        mode="edge",
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    return smoothed.astype(np.float32)


def temporal_postprocess(
    frame_scores: np.ndarray,
    fps: float,
    threshold: float,
    smoothing_window: int = 5,
    min_duration_seconds: float = 0.5,
    merge_gap_seconds: float = 0.5,
) -> np.ndarray:
    """
    Convert continuous frame anomaly scores into binary frame predictions.

    Processing:
        1. temporal smoothing
        2. thresholding
        3. minimum-duration filtering
        4. gap merging
    """

    scores = np.asarray(frame_scores, dtype=np.float32).reshape(-1)

    if scores.size == 0:
        return np.zeros(0, dtype=np.uint8)

    fps = max(float(fps), 1e-6)

    smoothed = smooth_scores(
        scores,
        window=smoothing_window,
    )

    labels = (smoothed >= float(threshold)).astype(np.uint8)

    # ------------------------------------------------------------------
    # Minimum duration filtering
    # ------------------------------------------------------------------

    min_frames = max(
        1,
        int(round(float(min_duration_seconds) * fps)),
    )

    if min_frames > 1:
        filtered = np.zeros_like(labels)

        start = None

        for i, value in enumerate(labels):
            if value == 1 and start is None:
                start = i

            is_end = (
                value == 0
                and start is not None
            )

            if is_end:
                end = i - 1

                if end - start + 1 >= min_frames:
                    filtered[start:end + 1] = 1

                start = None

        if start is not None:
            end = len(labels) - 1

            if end - start + 1 >= min_frames:
                filtered[start:end + 1] = 1

        labels = filtered

    # ------------------------------------------------------------------
    # Merge short gaps
    # ------------------------------------------------------------------

    merge_gap_frames = max(
        0,
        int(round(float(merge_gap_seconds) * fps)),
    )

    if merge_gap_frames > 0:
        positive_indices = np.where(labels > 0)[0]

        if positive_indices.size > 1:

            merged = labels.copy()

            previous = int(positive_indices[0])

            for current in positive_indices[1:]:

                current = int(current)

                gap = current - previous - 1

                if 0 < gap <= merge_gap_frames:
                    merged[previous + 1:current] = 1

                previous = current

            labels = merged

    return labels.astype(np.uint8)


# ---------------------------------------------------------------------------
# Feature extractor loading
# ---------------------------------------------------------------------------

def extract_state_dict(
    checkpoint: Any,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract a state dictionary from several common checkpoint formats.
    """

    if isinstance(checkpoint, dict):

        # Direct state dictionary.
        if checkpoint:
            tensor_values = [
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            ]

            if tensor_values and all(tensor_values):
                return checkpoint

        # Common checkpoint keys.
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "feature_extractor",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

    return None


def load_feature_extractor(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> TwoStreamFeatureExtractor:
    """
    Build and load the two-stream feature extractor.

    The constructor arguments are matched to the actual project
    TwoStreamFeatureExtractor implementation.
    """

    model_cfg = config.get("model", {})

    rgb_feature_dim = int(
        model_cfg.get("rgb_feature_dim", 512)
    )

    flow_feature_dim = int(
        model_cfg.get("flow_feature_dim", 128)
    )

    fusion_dim = int(
        model_cfg.get("fusion_dim", 256)
    )

    pretrained = bool(
        model_cfg.get("pretrained", True)
    )

    freeze_rgb_backbone = bool(
        model_cfg.get("freeze_rgb_backbone", True)
    )

    print()
    print("Building feature extractor...")

    feature_extractor = TwoStreamFeatureExtractor(
        rgb_feature_dim=rgb_feature_dim,
        flow_feature_dim=flow_feature_dim,
        fusion_dim=fusion_dim,
        pretrained=pretrained,
        freeze_rgb_backbone=freeze_rgb_backbone,
    )

    feature_extractor.to(device)
    feature_extractor.eval()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Feature checkpoint not found: {checkpoint_path}"
        )

    print()
    print("Loading feature extractor weights...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = extract_state_dict(checkpoint)

    if state_dict is None:
        raise RuntimeError(
            "Could not find a valid state dictionary in the feature "
            f"checkpoint:\n{checkpoint_path}"
        )

    # ---------------------------------------------------------------
    # The training checkpoint commonly stores:
    #
    # feature_extractor.xxx
    #
    # Strip that prefix when necessary.
    # ---------------------------------------------------------------

    cleaned_state = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("feature_extractor."):
            new_key = new_key[len("feature_extractor."):]

        cleaned_state[new_key] = value

    try:

        result = feature_extractor.load_state_dict(
            cleaned_state,
            strict=False,
        )

    except RuntimeError as exc:

        raise RuntimeError(
            "Failed to load feature extractor checkpoint.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Original error:\n{exc}"
        ) from exc

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    if missing:
        print(
            f"WARNING: {len(missing)} feature-extractor keys were missing."
        )

        for key in missing[:10]:
            print(f"  missing: {key}")

    if unexpected:
        print(
            f"WARNING: {len(unexpected)} unexpected checkpoint keys found."
        )

        for key in unexpected[:10]:
            print(f"  unexpected: {key}")

    feature_extractor.eval()

    print("Loaded feature extractor weights.")

    return feature_extractor


# ---------------------------------------------------------------------------
# Experiment H detector
# ---------------------------------------------------------------------------

def load_h_detector(
    checkpoint_path: Path,
) -> AvenueKNNDetector:
    """
    Load Experiment H KNN detector.

    Important:
        AvenueKNNDetector.load() is an INSTANCE method.
        Therefore we instantiate the detector first and then call load().
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"H checkpoint not found: {checkpoint_path}"
        )

    print()
    print("Loading Experiment H detector...")

    detector = AvenueKNNDetector(
        feature_dim=256,
        k_neighbors=5,
    )

    metadata = detector.load(
        checkpoint_path
    )

    print("H checkpoint loaded successfully.")

    memory_bank = getattr(
        detector,
        "memory_bank",
        None,
    )

    if memory_bank is not None:
        print(
            f"Memory bank size : {tuple(memory_bank.shape)}"
        )

    k_value = getattr(
        detector,
        "k_neighbors",
        None,
    )

    if k_value is not None:
        print(
            f"K                : {k_value}"
        )

    threshold = getattr(
        detector,
        "threshold",
        None,
    )

    if threshold is not None:
        print(
            f"H threshold      : {float(threshold):.9f}"
        )

    if isinstance(metadata, dict):

        if "threshold" in metadata:
            print(
                "Checkpoint metadata threshold: "
                f"{metadata['threshold']}"
            )

    return detector


# ---------------------------------------------------------------------------
# Motion-energy spatial pseudo-localization
# ---------------------------------------------------------------------------

def flow_to_numpy(
    flow_tensor: Any,
) -> np.ndarray:
    """
    Convert a flow tensor into NumPy.

    Expected common forms:
        [T, 2, H, W]
        [2, T, H, W]
        [N, T, 2, H, W]
    """

    array = to_numpy(flow_tensor)

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    return array


def normalize_flow_layout(
    flow: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Normalize flow into:
        [T, 2, H, W]
    """

    if flow.size == 0:
        return None

    flow = np.squeeze(flow)

    if flow.ndim == 4:

        # [T, 2, H, W]
        if flow.shape[1] == 2:
            return flow

        # [2, T, H, W]
        if flow.shape[0] == 2:
            return np.transpose(
                flow,
                (1, 0, 2, 3),
            )

    # Sometimes a single flow field appears as [2, H, W].
    if flow.ndim == 3:

        if flow.shape[0] == 2:
            return flow[None, ...]

        if flow.shape[-1] == 2:
            flow = np.transpose(
                flow,
                (2, 0, 1),
            )

            return flow[None, ...]

    return None


def create_motion_mask_from_flow(
    flow: np.ndarray,
    frame_height: int,
    frame_width: int,
    spatial_localizer: SpatialLocalizer,
) -> np.ndarray:
    """
    Generate a single binary motion-energy mask from a temporal flow clip.

    The flow is averaged over time before localization.

    This is intentionally described as pseudo-localization, not person
    detection.
    """

    normalized = normalize_flow_layout(flow)

    output = np.zeros(
        (frame_height, frame_width),
        dtype=np.uint8,
    )

    if normalized is None:
        return output

    if normalized.shape[0] == 0:
        return output

    # Average optical flow over temporal dimension.
    mean_flow = np.mean(
        normalized,
        axis=0,
    )

    if mean_flow.shape[0] != 2:
        return output

    # Resize flow to native Avenue resolution.
    h = mean_flow.shape[1]
    w = mean_flow.shape[2]

    if h <= 0 or w <= 0:
        return output

    fx = cv2.resize(
        mean_flow[0],
        (frame_width, frame_height),
        interpolation=cv2.INTER_LINEAR,
    )

    fy = cv2.resize(
        mean_flow[1],
        (frame_width, frame_height),
        interpolation=cv2.INTER_LINEAR,
    )

    native_flow = np.stack(
        [fx, fy],
        axis=0,
    )

    # SpatialLocalizer expects a conventional H x W x 2 flow array.
    native_flow = np.transpose(
        native_flow,
        (1, 2, 0),
    )

    try:

        rois = spatial_localizer.extract_motion_rois(
            native_flow,
            (frame_height, frame_width),
        )

    except Exception as exc:

        print(
            "WARNING: spatial ROI extraction failed for one clip:"
        )
        print(f"  {type(exc).__name__}: {exc}")

        return output

    if not rois:
        return output

    for roi in rois:

        bbox = roi.get("bbox_pixel")

        if bbox is None:
            continue

        if len(bbox) != 4:
            continue

        x, y, bw, bh = [
            int(round(float(v)))
            for v in bbox
        ]

        x1 = max(0, min(frame_width - 1, x))
        y1 = max(0, min(frame_height - 1, y))

        x2 = max(
            x1 + 1,
            min(frame_width, x + bw),
        )

        y2 = max(
            y1 + 1,
            min(frame_height, y + bh),
        )

        output[y1:y2, x1:x2] = 1

    return output


def build_spatial_masks(
    flow_tensor: Any,
    clip_native_ranges: Sequence[Tuple[int, int]],
    native_frame_count: int,
    frame_height: int,
    frame_width: int,
    spatial_localizer: SpatialLocalizer,
) -> np.ndarray:
    """
    Build a native-resolution spatial mask for every native frame.

    Each clip produces one motion-energy pseudo-localization mask.
    That mask is assigned to the native frames covered by the clip.

    Where multiple clips overlap, masks are combined with OR.
    """

    output = np.zeros(
        (
            native_frame_count,
            frame_height,
            frame_width,
        ),
        dtype=np.uint8,
    )

    flow = flow_to_numpy(
        flow_tensor
    )

    # Remove an optional batch dimension.
    if flow.ndim == 5 and flow.shape[0] == 1:
        flow = flow[0]

    # We expect:
    # [num_clips, T, 2, H, W]
    if flow.ndim != 5:

        print(
            "WARNING: unexpected flow tensor shape:"
            f" {flow.shape}"
        )

        return output

    num_clips = min(
        len(clip_native_ranges),
        flow.shape[0],
    )

    for clip_idx in range(num_clips):

        start_frame, end_frame = (
            clip_native_ranges[clip_idx]
        )

        start_frame = max(
            0,
            min(
                native_frame_count - 1,
                int(start_frame),
            ),
        )

        end_frame = max(
            start_frame,
            min(
                native_frame_count - 1,
                int(end_frame),
            ),
        )

        clip_flow = flow[clip_idx]

        mask = create_motion_mask_from_flow(
            clip_flow,
            frame_height,
            frame_width,
            spatial_localizer,
        )

        if not np.any(mask):
            continue

        output[
            start_frame:end_frame + 1
        ] |= mask[None, :, :]

    return output


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def get_video_id(item: Dict[str, Any]) -> str:
    """Extract a stable video identifier."""

    value = item.get("video_id", "unknown")

    if isinstance(value, torch.Tensor):
        value = value.item()

    return str(value)


def get_native_frame_count(
    item: Dict[str, Any],
) -> int:
    """Read native frame count from AvenueDataset output."""

    value = item.get(
        "native_frame_count",
        0,
    )

    if isinstance(value, torch.Tensor):
        value = value.item()

    value = int(value)

    if value <= 0:
        raise ValueError(
            "Invalid native_frame_count returned by AvenueDataset."
        )

    return value


def get_native_fps(
    item: Dict[str, Any],
) -> float:
    """Read native FPS from AvenueDataset output."""

    value = item.get(
        "native_fps",
        25.0,
    )

    if isinstance(value, torch.Tensor):
        value = value.item()

    return float(value)


def get_clip_ranges(
    item: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Read clip native-frame ranges."""

    ranges = item.get(
        "clip_native_ranges",
        [],
    )

    normalized = []

    for value in ranges:

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()

        if len(value) < 2:
            continue

        normalized.append(
            (
                int(value[0]),
                int(value[1]),
            )
        )

    return normalized


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(
    dataset: AvenueDataset,
    index: int,
    feature_extractor: TwoStreamFeatureExtractor,
    detector: AvenueKNNDetector,
    spatial_localizer: SpatialLocalizer,
    device: torch.device,
    threshold: float,
    smoothing_window: int,
    min_duration_seconds: float,
    merge_gap_seconds: float,
) -> Dict[str, Any]:
    """
    Process one complete Avenue test video.
    """

    item = dataset[index]

    video_id = get_video_id(item)

    native_frame_count = get_native_frame_count(item)

    native_fps = get_native_fps(item)

    ground_truth_path = item.get(
        "ground_truth_path"
    )

    print()
    print("-" * 70)
    print(f"VIDEO {video_id}")
    print(f"Native frames : {native_frame_count}")
    print(f"Native FPS    : {native_fps:.3f}")
    print(f"Ground truth  : {ground_truth_path}")

    rgb = item.get("rgb")
    flow = item.get("flow")

    if rgb is None:
        raise RuntimeError(
            f"Video {video_id}: AvenueDataset returned no RGB tensor."
        )

    if flow is None:
        raise RuntimeError(
            f"Video {video_id}: AvenueDataset returned no flow tensor."
        )

    clip_native_ranges = get_clip_ranges(item)

    if not clip_native_ranges:
        raise RuntimeError(
            f"Video {video_id}: no clip_native_ranges returned."
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    print()

    try:

        clip_features = extract_clip_features(
            feature_extractor=feature_extractor,
            rgb_tensor=rgb,
            flow_tensor=flow,
            device=device,
            chunk_size=16,
        )

    except TypeError:

        # Defensive compatibility fallback for alternate positional
        # implementations.
        clip_features = extract_clip_features(
            feature_extractor,
            rgb,
            flow,
            device,
            16,
        )

    clip_features = np.asarray(
        clip_features,
        dtype=np.float32,
    )

    if clip_features.ndim != 2:
        raise RuntimeError(
            f"Unexpected clip feature shape: "
            f"{clip_features.shape}"
        )

    print(
        f"Clip features : {clip_features.shape}"
    )

    # ------------------------------------------------------------------
    # Experiment H KNN anomaly scores
    # ------------------------------------------------------------------

    try:

        clip_scores = detector.predict_score(
            clip_features
        )

    except TypeError:

        clip_scores = detector.predict_score(
            features=clip_features
        )

    clip_scores = np.asarray(
        clip_scores,
        dtype=np.float32,
    ).reshape(-1)

    if clip_scores.size == 0:
        raise RuntimeError(
            f"Video {video_id}: detector returned no scores."
        )

    print(
        "Clip score range: "
        f"[{clip_scores.min():.6f}, "
        f"{clip_scores.max():.6f}]"
    )

    # ------------------------------------------------------------------
    # CRITICAL FIX
    #
    # Actual project function signature:
    #
    # align_clip_scores_to_frames(
    #     clip_scores,
    #     clip_native_ranges,
    #     total_native_frames
    # )
    #
    # The previous script incorrectly used num_frames=...
    # ------------------------------------------------------------------

    frame_scores = align_clip_scores_to_frames(
        clip_scores=clip_scores,
        clip_native_ranges=clip_native_ranges,
        total_native_frames=native_frame_count,
    )

    frame_scores = np.asarray(
        frame_scores,
        dtype=np.float32,
    ).reshape(-1)

    if len(frame_scores) != native_frame_count:
        raise RuntimeError(
            "Frame-score alignment produced an unexpected length: "
            f"{len(frame_scores)} "
            f"(expected {native_frame_count})"
        )

    # ------------------------------------------------------------------
    # Temporal localization
    # ------------------------------------------------------------------

    frame_labels = temporal_postprocess(
        frame_scores=frame_scores,
        fps=native_fps,
        threshold=threshold,
        smoothing_window=smoothing_window,
        min_duration_seconds=min_duration_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )

    # ------------------------------------------------------------------
    # Spatial pseudo-localization
    # ------------------------------------------------------------------

    print(
        "Generating motion-energy spatial "
        "pseudo-localization..."
    )

    spatial_masks = build_spatial_masks(
        flow_tensor=flow,
        clip_native_ranges=clip_native_ranges,
        native_frame_count=native_frame_count,
        frame_height=360,
        frame_width=640,
        spatial_localizer=spatial_localizer,
    )

    spatial_available = bool(
        np.any(spatial_masks)
    )

    print(
        "Spatial pseudo-localization available: "
        f"{spatial_available}"
    )

    predicted_abnormal_frames = int(
        np.sum(frame_labels)
    )

    print(
        f"Predicted abnormal frames: "
        f"{predicted_abnormal_frames}"
    )

    print(
        f"Prediction rate: "
        f"{predicted_abnormal_frames / max(native_frame_count, 1):.4f}"
    )

    return {
        "video_id": video_id,
        "frame_count": native_frame_count,
        "fps": native_fps,
        "ground_truth_path": (
            str(ground_truth_path)
            if ground_truth_path is not None
            else None
        ),
        "pred_frame_scores": frame_scores,
        "pred_frame_labels": frame_labels,
        "pred_spatial_masks": spatial_masks,
        "clip_scores": clip_scores,
        "clip_features": clip_features,
        "clip_native_ranges": clip_native_ranges,
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def make_json_serializable(value: Any) -> Any:
    """
    Recursively convert evaluator results into JSON-serializable objects.

    AvenueEvaluator may return:
      - NumPy arrays
      - NumPy scalar values
      - Torch tensors
      - Pandas DataFrames
      - Pandas Series
      - dictionaries
      - lists / tuples
      - pathlib.Path objects
    """

    # ---------------------------------------------------------------
    # Pandas DataFrame
    # ---------------------------------------------------------------
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return [
                make_json_serializable(row)
                for row in value.to_dict(orient="records")
            ]

        if isinstance(value, pd.Series):
            return [
                make_json_serializable(item)
                for item in value.tolist()
            ]

    except ImportError:
        pass

    # ---------------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------------
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    # ---------------------------------------------------------------
    # List / tuple / set
    # ---------------------------------------------------------------
    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    # ---------------------------------------------------------------
    # NumPy ndarray
    # ---------------------------------------------------------------
    if isinstance(value, np.ndarray):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    # ---------------------------------------------------------------
    # NumPy scalar
    # ---------------------------------------------------------------
    if isinstance(value, np.generic):
        return value.item()

    # ---------------------------------------------------------------
    # PyTorch tensor
    # ---------------------------------------------------------------
    if isinstance(value, torch.Tensor):
        return make_json_serializable(
            value.detach().cpu().numpy()
        )

    # ---------------------------------------------------------------
    # pathlib.Path
    # ---------------------------------------------------------------
    if isinstance(value, Path):
        return str(value)

    # ---------------------------------------------------------------
    # Python float edge cases
    # JSON does not properly represent NaN / Inf.
    # ---------------------------------------------------------------
    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    # ---------------------------------------------------------------
    # Basic JSON types
    # ---------------------------------------------------------------
    if value is None:
        return None

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    # ---------------------------------------------------------------
    # Last-resort conversion for uncommon evaluator objects
    # ---------------------------------------------------------------
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def save_video_prediction(
    result: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save compact prediction arrays for one video.

    Large feature arrays are intentionally not saved here.
    """

    prediction_dir = output_dir / "predictions"

    ensure_directory(prediction_dir)

    video_id = str(
        result["video_id"]
    )

    output_file = (
        prediction_dir
        / f"video_{video_id}.npz"
    )

    np.savez_compressed(
        output_file,
        frame_scores=np.asarray(
            result["pred_frame_scores"],
            dtype=np.float32,
        ),
        frame_labels=np.asarray(
            result["pred_frame_labels"],
            dtype=np.uint8,
        ),
        spatial_masks=np.asarray(
            result["pred_spatial_masks"],
            dtype=np.uint8,
        ),
        clip_scores=np.asarray(
            result["clip_scores"],
            dtype=np.float32,
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation result handling
# ---------------------------------------------------------------------------

def print_metric_value(
    name: str,
    value: Any,
) -> None:
    """Print a metric safely."""

    if value is None:
        print(f"{name:<22}: None")
        return

    if isinstance(value, float):
        print(f"{name:<22}: {value:.6f}")
        return

    print(f"{name:<22}: {value}")


def extract_global_metrics(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Locate global metrics regardless of whether the evaluator calls the
    section 'global', 'global_metrics', or 'summary'.
    """

    for key in (
        "global",
        "global_metrics",
        "summary",
    ):
        value = report.get(key)

        if isinstance(value, dict):
            return value

    return {}


def print_evaluation_summary(
    report: Dict[str, Any],
) -> None:
    """Print the most important evaluation metrics."""

    metrics = extract_global_metrics(
        report
    )

    print()
    print("=" * 70)
    print("EXPERIMENT I EVALUATION SUMMARY")
    print("=" * 70)

    if not metrics:
        print(
            "Evaluator did not expose a global metric dictionary "
            "at the expected keys."
        )

        print(
            "Available top-level keys:"
        )

        for key in report.keys():
            print(f"  - {key}")

        print("=" * 70)

        return

    ordered_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Temporal IoU", "temporal_iou"),
        ("Spatial mIoU", "spatial_miou"),
        ("TP", "tp"),
        ("FP", "fp"),
        ("TN", "tn"),
        ("FN", "fn"),
    ]

    for label, key in ordered_metrics:
        print_metric_value(
            label,
            metrics.get(key),
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Experiment I: Avenue H + "
            "Motion-Energy Spatial Pseudo-Localization"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/avenue.yaml",
        help="Path to Avenue configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/avenue/avenue_knn_best.pt",
        help="Experiment H KNN checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/metrics/avenue_knn_spatial_test",
        help="Output directory.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of test videos. Default: all.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index from which to start.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    config_path = resolve_path(
        args.config
    )

    h_checkpoint = resolve_path(
        args.checkpoint
    )

    output_dir = resolve_path(
        args.output_dir
    )

    ensure_directory(output_dir)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT I")
    print("CUHK AVENUE:")
    print("H + Motion-Energy Spatial Pseudo-Localization")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config = load_yaml(
        config_path
    )

    runtime_cfg = config.get(
        "runtime",
        {},
    )

    configured_device = str(
        runtime_cfg.get(
            "device",
            "auto",
        )
    )

    if args.device == "auto":
        requested_device = configured_device
    else:
        requested_device = args.device

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            print(
                "WARNING: CUDA requested but unavailable. "
                "Using CPU."
            )

            device = torch.device("cpu")

        else:
            device = torch.device("cuda")

    elif requested_device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Device       : {device}"
    )

    print(
        f"Output dir   : {output_dir}"
    )

    seed = int(
        runtime_cfg.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    # ------------------------------------------------------------------
    # Avenue configuration
    # ------------------------------------------------------------------

    dataset_cfg = config.get(
        "dataset",
        {},
    )

    video_cfg = config.get(
        "video",
        {},
    )

    evaluation_cfg = config.get(
        "evaluation",
        {},
    )

    avenue_cfg = config.get(
        "avenue_oneclass",
        {},
    )

    metadata_file = resolve_path(
        dataset_cfg.get(
            "metadata_file",
            "datasets/metadata/avenue_metadata.csv",
        )
    )

    ground_truth_root = resolve_path(
        dataset_cfg.get(
            "ground_truth_root",
            "D:/Avenue-Dataset/"
            "ground_truth/ground_truth_demo/"
            "testing_label_mask",
        )
    )

    feature_checkpoint = resolve_path(
        "checkpoints/best_model.pt"
    )

    print()
    print("Metadata:")
    print(
        f"  {metadata_file}"
    )

    print("Ground truth:")
    print(
        f"  {ground_truth_root}"
    )

    print("H checkpoint:")
    print(
        f"  {h_checkpoint}"
    )

    print("Feature checkpoint:")
    print(
        f"  {feature_checkpoint}"
    )

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Avenue metadata file not found:\n"
            f"{metadata_file}"
        )

    if not ground_truth_root.exists():
        raise FileNotFoundError(
            f"Avenue ground-truth directory not found:\n"
            f"{ground_truth_root}"
        )

    # ------------------------------------------------------------------
    # Load feature extractor
    # ------------------------------------------------------------------

    feature_extractor = load_feature_extractor(
        checkpoint_path=feature_checkpoint,
        config=config,
        device=device,
    )

    # ------------------------------------------------------------------
    # Load Experiment H detector
    # ------------------------------------------------------------------

    detector = load_h_detector(
        h_checkpoint
    )

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    h_threshold = getattr(
        detector,
        "threshold",
        None,
    )

    if h_threshold is None:

        h_threshold = float(
            avenue_cfg.get(
                "threshold",
                evaluation_cfg.get(
                    "anomaly_threshold",
                    0.5,
                ),
            )
        )

    threshold = float(
        h_threshold
    )

    spatial_threshold = float(
        evaluation_cfg.get(
            "spatial_threshold",
            0.35,
        )
    )

    smoothing_window = int(
        avenue_cfg.get(
            "score_smoothing_window",
            evaluation_cfg.get(
                "temporal_smoothing_kernel",
                5,
            ),
        )
    )

    min_duration_seconds = float(
        avenue_cfg.get(
            "min_anomaly_duration_seconds",
            evaluation_cfg.get(
                "min_duration_seconds",
                0.5,
            ),
        )
    )

    merge_gap_seconds = float(
        avenue_cfg.get(
            "merge_gap_seconds",
            0.5,
        )
    )

    print(
        f"Spatial threshold: {spatial_threshold}"
    )

    # ------------------------------------------------------------------
    # Spatial localizer
    # ------------------------------------------------------------------

    spatial_localizer = SpatialLocalizer(
        min_box_area_ratio=0.005,
        threshold_ratio=spatial_threshold,
    )

    # ------------------------------------------------------------------
    # Load Avenue test dataset
    #
    # IMPORTANT:
    # AvenueDataset does NOT take root= or ground_truth_root=
    # directly. Its verified constructor uses metadata_file.
    # ------------------------------------------------------------------

    print()
    print("Loading Avenue test dataset...")

    image_size = tuple(
        video_cfg.get(
            "image_size",
            [224, 224],
        )
    )

    flow_size = tuple(
        video_cfg.get(
            "flow_size",
            [112, 112],
        )
    )

    clip_length = int(
        video_cfg.get(
            "clip_length",
            16,
        )
    )

    stride = int(
        video_cfg.get(
            "stride",
            16,
        )
    )

    target_fps = int(
        video_cfg.get(
            "target_fps",
            16,
        )
    )

    test_dataset = AvenueDataset(
        metadata_file=str(
            metadata_file
        ),
        split="test",
        clip_length=clip_length,
        stride=stride,
        image_size=image_size,
        flow_size=flow_size,
        target_fps=target_fps,
        is_training=False,
        use_cache=True,
        full_video_clips=True,
    )

    total_videos = len(
        test_dataset
    )

    print(
        f"Test videos: {total_videos}"
    )

    start_index = max(
        0,
        int(args.start_index),
    )

    if start_index >= total_videos:
        raise ValueError(
            f"start-index {start_index} is outside "
            f"the test dataset of {total_videos} videos."
        )

    if args.max_videos is None:
        end_index = total_videos
    else:
        end_index = min(
            total_videos,
            start_index + max(
                0,
                int(args.max_videos),
            ),
        )

    print(
        f"Videos to process: "
        f"{end_index - start_index}"
    )

    # ------------------------------------------------------------------
    # Process videos
    # ------------------------------------------------------------------

    video_predictions: List[Dict[str, Any]] = []

    for index in range(
        start_index,
        end_index,
    ):

        try:

            result = process_video(
                dataset=test_dataset,
                index=index,
                feature_extractor=feature_extractor,
                detector=detector,
                spatial_localizer=spatial_localizer,
                device=device,
                threshold=threshold,
                smoothing_window=smoothing_window,
                min_duration_seconds=min_duration_seconds,
                merge_gap_seconds=merge_gap_seconds,
            )

            video_predictions.append(
                result
            )

            save_video_prediction(
                result,
                output_dir,
            )

        except Exception as exc:

            print()
            print("=" * 70)
            print(
                f"ERROR PROCESSING VIDEO INDEX {index}"
            )
            print("=" * 70)

            print(
                f"{type(exc).__name__}: {exc}"
            )

            # For debugging, do not silently continue.
            raise

    if not video_predictions:
        raise RuntimeError(
            "No videos were successfully processed."
        )

    # ------------------------------------------------------------------
    # Avenue evaluation
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Running Avenue ground-truth evaluation...")
    print("=" * 70)

    evaluator = AvenueEvaluator(
        ground_truth_root=str(
            ground_truth_root
        ),
        default_threshold=threshold,
    )

    try:

        report = evaluator.evaluate_test_set(
            video_predictions,
            threshold=threshold,
        )

    except TypeError:

        # Compatibility fallback in case an evaluator implementation
        # uses a positional threshold argument.
        report = evaluator.evaluate_test_set(
            video_predictions,
            threshold,
        )

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------

    report_path = (
        output_dir
        / "avenue_knn_spatial_evaluation_report.json"
    )

    report_for_json = make_json_serializable(
        report
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report_for_json,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved evaluation report:"
    )

    print(
        f"  {report_path}"
    )

    # ------------------------------------------------------------------
    # Save experiment metadata
    # ------------------------------------------------------------------

    metadata = {
        "experiment": (
            "Experiment I: Avenue Experiment H "
            "+ Motion-Energy Spatial Pseudo-Localization"
        ),
        "config_file": str(
            config_path
        ),
        "h_checkpoint": str(
            h_checkpoint
        ),
        "feature_checkpoint": str(
            feature_checkpoint
        ),
        "device": str(
            device
        ),
        "videos_processed": len(
            video_predictions
        ),
        "threshold": threshold,
        "spatial_threshold": spatial_threshold,
        "smoothing_window": smoothing_window,
        "min_duration_seconds": min_duration_seconds,
        "merge_gap_seconds": merge_gap_seconds,
        "spatial_method": (
            "Motion-Energy Based Spatial "
            "Pseudo-Localization"
        ),
        "ground_truth_usage": (
            "Ground truth is used only for evaluation."
        ),
        "spatial_ground_truth_warning": (
            "Predicted spatial regions are "
            "motion-energy pseudo-localizations, "
            "not person-detection ground truth."
        ),
    }

    metadata_path = (
        output_dir
        / "experiment_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Print final metrics
    # ------------------------------------------------------------------

    print_evaluation_summary(
        report
    )

    print()
    print("=" * 70)
    print("EXPERIMENT I COMPLETED")
    print("=" * 70)

    print(
        f"Videos processed : "
        f"{len(video_predictions)}"
    )

    print(
        f"Report           : "
        f"{report_path}"
    )

    print(
        f"Predictions      : "
        f"{output_dir / 'predictions'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()