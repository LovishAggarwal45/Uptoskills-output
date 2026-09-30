"""
PyTorch Dataset for the CUHK Avenue Surveillance Dataset.

Supports:
- Normal-only training clips (unsupervised / one-class anomaly detection paradigm)
- Full temporal clip extraction with RGB normalization and dense optical flow
- Frame and clip indexing metadata for frame-aligned testing evaluation
- Ground-truth paths exposed as metadata without leaking masks into model inputs
- OpenCV-based frame decoding and caching for accelerated multi-epoch usage
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.optical_flow import OpticalFlowEngine

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class AvenueDataset(Dataset):
    """
    PyTorch Dataset for CUHK Avenue videos.

    In Avenue:
    - Training videos (01-16) contain ONLY normal activities.
    - Testing videos (01-21) contain both normal and abnormal intervals,
      evaluated against pixel-level ground-truth masks.
    """

    _cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        metadata_file: str,
        split: str = "train",
        num_segments: int = 12,
        clip_length: int = 16,
        stride: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        flow_size: Tuple[int, int] = (112, 112),
        target_fps: int = 16,
        is_training: Optional[bool] = None,
        use_cache: bool = True,
        full_video_clips: bool = False
    ):
        super().__init__()
        self.metadata_file = Path(metadata_file)
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Avenue metadata file not found: {self.metadata_file}")

        self.split = split.lower().strip()
        self.num_segments = num_segments
        self.clip_length = clip_length
        self.stride = stride
        self.image_size = tuple(image_size)
        self.flow_size = tuple(flow_size)
        self.target_fps = target_fps
        self.is_training = (self.split == "train") if is_training is None else is_training
        self.use_cache = use_cache
        self.full_video_clips = full_video_clips

        self.flow_engine = OpticalFlowEngine(target_size=self.flow_size)

        # Load metadata
        df_all = pd.read_csv(self.metadata_file, dtype={"video_id": str})
        required_cols = {"video_id", "video_path", "split", "frame_count", "fps"}
        missing = required_cols - set(df_all.columns)
        if missing:
            raise ValueError(f"Metadata file missing columns: {missing}")

        self.df = df_all[df_all["split"].astype(str).str.lower() == self.split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{self.split}' in {self.metadata_file}")

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_frames(self, frames_rgb: np.ndarray) -> np.ndarray:
        """
        Applies ImageNet normalization and transposes to [T, 3, H, W].
        frames_rgb: [T, H, W, 3] in uint8 [0, 255]
        """
        normed = (frames_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(normed, (0, 3, 1, 2)).astype(np.float32)

    def _read_video_and_extract_clips(self, video_path: Path) -> Dict[str, Any]:
        """
        Streams frames from video with target_fps resampling,
        extracts temporal clips, computes optical flow, and records frame mapping.
        """
        cache_key = f"{video_path.resolve()}_{self.target_fps}_{self.clip_length}_{self.stride}_{self.image_size}_{self.flow_size}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        native_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = native_frame_count / native_fps if native_fps > 0 else 0.0

        # Frame step for target FPS resampling
        step = native_fps / self.target_fps if self.target_fps > 0 else 1.0

        resampled_frames = []
        resampled_native_indices = []
        resampled_timestamps = []

        curr_target_idx = 0
        raw_idx = 0

        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            target_boundary = curr_target_idx * step
            if raw_idx >= target_boundary:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if (rgb.shape[1], rgb.shape[0]) != self.image_size:
                    rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_LINEAR)

                ts = raw_idx / native_fps
                resampled_frames.append(rgb)
                resampled_native_indices.append(raw_idx)
                resampled_timestamps.append(ts)
                curr_target_idx += 1

            raw_idx += 1

        cap.release()

        if len(resampled_frames) == 0:
            raise RuntimeError(f"No frames could be extracted from video: {video_path}")

        # Construct clips
        rgb_clips_list = []
        flow_clips_list = []
        clip_native_ranges = []
        clip_time_ranges = []

        total_resampled = len(resampled_frames)
        idx = 0
        while idx + self.clip_length <= total_resampled:
            clip_frames = np.stack(resampled_frames[idx:idx + self.clip_length], axis=0) # [T, H, W, 3]
            norm_rgb = self._normalize_frames(clip_frames)
            flow_data = self.flow_engine.compute_clip_flow(clip_frames)

            start_native = resampled_native_indices[idx]
            end_native = resampled_native_indices[idx + self.clip_length - 1]
            start_t = resampled_timestamps[idx]
            end_t = resampled_timestamps[idx + self.clip_length - 1]

            rgb_clips_list.append(norm_rgb)
            flow_clips_list.append(flow_data["flow"])
            clip_native_ranges.append((start_native, end_native))
            clip_time_ranges.append((start_t, end_t))

            idx += self.stride

        # If video is shorter than clip_length or has remaining tail, create at least 1 clip by padding
        if len(rgb_clips_list) == 0:
            pad_frames = list(resampled_frames)
            while len(pad_frames) < self.clip_length:
                pad_frames.append(pad_frames[-1])
            clip_frames = np.stack(pad_frames, axis=0)
            norm_rgb = self._normalize_frames(clip_frames)
            flow_data = self.flow_engine.compute_clip_flow(clip_frames)

            rgb_clips_list.append(norm_rgb)
            flow_clips_list.append(flow_data["flow"])
            start_native = resampled_native_indices[0]
            end_native = resampled_native_indices[-1]
            clip_native_ranges.append((start_native, end_native))
            clip_time_ranges.append((resampled_timestamps[0], resampled_timestamps[-1]))

        extracted = {
            "rgb": np.stack(rgb_clips_list, axis=0),   # [num_clips, T, 3, H, W]
            "flow": np.stack(flow_clips_list, axis=0), # [num_clips, T, 2, H_f, W_f]
            "clip_native_ranges": clip_native_ranges,
            "clip_time_ranges": clip_time_ranges,
            "native_frame_count": native_frame_count,
            "native_fps": native_fps,
            "duration": duration
        }

        if self.use_cache:
            self._cache[cache_key] = extracted

        return extracted

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        vpath = Path(str(row["video_path"]))
        video_id = str(row["video_id"]).zfill(2)
        gt_path = str(row.get("ground_truth_path", "") or "")

        data = self._read_video_and_extract_clips(vpath)
        all_rgb = data["rgb"]
        all_flow = data["flow"]
        total_clips = len(all_rgb)

        if self.full_video_clips or not self.is_training:
            # Return all sequential clips for complete temporal evaluation
            rgb_tensor = torch.from_numpy(all_rgb)
            flow_tensor = torch.from_numpy(all_flow)
            time_ranges = data["clip_time_ranges"]
            native_ranges = data["clip_native_ranges"]
        else:
            # Segment sampling for training (uniform distribution across video)
            if total_clips >= self.num_segments:
                sel_indices = np.linspace(0, total_clips - 1, self.num_segments, dtype=int)
            else:
                repeats = (self.num_segments // total_clips) + 1
                sel_indices = np.tile(np.arange(total_clips), repeats)[:self.num_segments]

            rgb_sel = all_rgb[sel_indices]
            flow_sel = all_flow[sel_indices]
            time_ranges = [data["clip_time_ranges"][i] for i in sel_indices]
            native_ranges = [data["clip_native_ranges"][i] for i in sel_indices]

            rgb_tensor = torch.from_numpy(rgb_sel)
            flow_tensor = torch.from_numpy(flow_sel)

        item: Dict[str, Any] = {
            "video_id": video_id,
            "split": self.split,
            "rgb": rgb_tensor,
            "flow": flow_tensor,
            "duration": float(data["duration"]),
            "native_frame_count": int(data["native_frame_count"]),
            "native_fps": float(data["native_fps"]),
            "clip_time_ranges": time_ranges,
            "clip_native_ranges": native_ranges,
        }

        # For training: all Avenue training videos are normal (label=0, is_anomaly=0.0)
        if self.split == "train":
            item["is_anomaly"] = torch.tensor(0.0, dtype=torch.float32)
            item["label"] = torch.tensor(0, dtype=torch.long)
        else:
            # For testing: provide ground_truth_path as reference metadata only.
            # DO NOT leak ground-truth masks into model inputs.
            item["ground_truth_path"] = gt_path

        return item


def get_avenue_dataloader(
    metadata_file: str,
    split: str = "train",
    batch_size: int = 1,
    num_segments: int = 12,
    clip_length: int = 16,
    stride: int = 16,
    target_fps: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    full_video_clips: bool = False
) -> DataLoader:
    """
    Factory function for Avenue DataLoader.
    """
    is_train = (split.lower() == "train")
    dataset = AvenueDataset(
        metadata_file=metadata_file,
        split=split,
        num_segments=num_segments,
        clip_length=clip_length,
        stride=stride,
        target_fps=target_fps,
        is_training=is_train,
        full_video_clips=full_video_clips
    )

    should_shuffle = is_train if shuffle is None else shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
