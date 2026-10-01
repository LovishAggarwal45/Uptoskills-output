"""
PyTorch Dataset and DataLoaders for Weakly Supervised Action Localization.

Supports:
- Existing 3-class CCTV dataset
- UCF-Crime binary anomaly dataset
- RGB temporal clip extraction
- Dense optical-flow extraction
- Uniform temporal sampling
- Balanced batches for the original 3-class dataset
- Anomaly-only batches for UCF-Crime
"""

import math
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from src.data.video_reader import VideoReader
from src.preprocessing.optical_flow import OpticalFlowEngine


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)


class BalancedBatchSampler(Sampler):
    """
    Balanced sampler for the original CCTV dataset.

    Each batch contains approximately:
    - half Normal
    - half Anomalous

    Sampling with replacement is used when a class has fewer
    samples than required for the requested number of batches.
    """

    def __init__(
        self,
        labels,
        batch_size: int = 4,
        seed: int = 42
    ):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.seed = seed

        self.normal_indices = np.where(self.labels == 0)[0]
        self.anomaly_indices = np.where(self.labels > 0)[0]

        if len(self.normal_indices) == 0:
            raise ValueError(
                "BalancedBatchSampler requires at least one Normal sample."
            )

        if len(self.anomaly_indices) == 0:
            raise ValueError(
                "BalancedBatchSampler requires at least one Anomalous sample."
            )

        self.num_batches = max(
            1,
            math.ceil(len(self.labels) / self.batch_size)
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)

        normal_per_batch = self.batch_size // 2
        anomaly_per_batch = self.batch_size - normal_per_batch

        for _ in range(self.num_batches):
            normal_batch = rng.choice(
                self.normal_indices,
                size=normal_per_batch,
                replace=True
            )

            anomaly_batch = rng.choice(
                self.anomaly_indices,
                size=anomaly_per_batch,
                replace=True
            )

            batch = np.concatenate(
                [normal_batch, anomaly_batch]
            )

            rng.shuffle(batch)

            yield batch.tolist()

    def __len__(self):
        return self.num_batches


class SurveillanceVideoDataset(Dataset):
    """
    Dataset for surveillance videos.

    Supported CSV formats:

    Existing project:
        video_id
        path
        label_id
        is_anomaly

    UCF-Crime:
        video_id
        path
        label
        class_name
        ucf_category
        evaluation_type
    """

    def __init__(
        self,
        split_csv: str,
        root_dir: str = ".",
        num_segments: int = 12,
        image_size: Tuple[int, int] = (224, 224),
        flow_size: Tuple[int, int] = (112, 112),
        target_fps: int = 16,
        max_duration_seconds: int = 3600,
        is_training: bool = False,
    ):
        self.split_csv = Path(split_csv)
        self.root_dir = Path(root_dir)

        self.num_segments = int(num_segments)
        self.image_size = tuple(image_size)
        self.flow_size = tuple(flow_size)
        self.target_fps = int(target_fps)
        self.max_duration_seconds = int(max_duration_seconds)
        self.is_training = bool(is_training)

        if not self.split_csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.split_csv}"
            )

        self.df = pd.read_csv(self.split_csv)

        if self.df.empty:
            raise ValueError(
                f"Dataset CSV is empty: {self.split_csv}"
            )

        # ---------------------------------------------------------
        # Detect dataset format
        # ---------------------------------------------------------

        self.is_ucf = (
            "label" in self.df.columns
            and "class_name" in self.df.columns
            and "ucf_category" in self.df.columns
        )

        if self.is_ucf:
            print(
                f"Detected UCF-Crime dataset: {self.split_csv}"
            )

            required_columns = {
                "video_id",
                "path",
                "label",
                "class_name",
                "ucf_category",
            }

            missing = required_columns - set(self.df.columns)

            if missing:
                raise ValueError(
                    f"Missing UCF columns: {sorted(missing)}"
                )

            self.df["label_id"] = (
                self.df["label"]
                .astype(int)
            )

            self.df["is_anomaly"] = (
                self.df["label_id"] > 0
            ).astype(int)

        else:
            print(
                f"Detected existing CCTV dataset: {self.split_csv}"
            )

            required_columns = {
                "video_id",
                "path",
                "label_id",
                "is_anomaly",
            }

            missing = required_columns - set(self.df.columns)

            if missing:
                raise ValueError(
                    f"Missing dataset columns: {sorted(missing)}"
                )

            self.df["label_id"] = (
                self.df["label_id"]
                .astype(int)
            )

            self.df["is_anomaly"] = (
                self.df["is_anomaly"]
                .astype(int)
            )

        # ---------------------------------------------------------
        # Validate video paths
        # ---------------------------------------------------------

        self._validate_video_paths()

        # ---------------------------------------------------------
        # Dataset summary
        # ---------------------------------------------------------

        normal_count = int(
            (self.df["is_anomaly"] == 0).sum()
        )

        anomaly_count = int(
            (self.df["is_anomaly"] == 1).sum()
        )

        if self.is_ucf:
            print(
                f"UCF-Crime dataset loaded: {len(self.df)} videos"
            )
            print(f"  Normal   : {normal_count}")
            print(f"  Anomaly  : {anomaly_count}")

            if self.is_training and normal_count == 0:
                print(
                    "Using UCF-Crime anomaly-only training batches."
                )
                print(
                    f"  Total training samples: {len(self.df)}"
                )

        else:
            print(
                f"CCTV dataset loaded: {len(self.df)} videos"
            )
            print(f"  Normal   : {normal_count}")
            print(f"  Anomaly  : {anomaly_count}")

    def _validate_video_paths(self):
        """
        Validate the first several video paths so that obvious
        metadata problems are detected before training.
        """

        check_count = min(20, len(self.df))

        for i in range(check_count):
            video_path = Path(
                str(self.df.iloc[i]["path"])
            )

            if not video_path.exists():
                raise FileNotFoundError(
                    f"Video file not found in metadata:\n"
                    f"{video_path}"
                )

    def __len__(self):
        return len(self.df)

    # -------------------------------------------------------------
    # RGB preprocessing
    # -------------------------------------------------------------

    def _prepare_rgb(self, frames: np.ndarray) -> torch.Tensor:
        """
        Convert RGB uint8 frames:

            [T, H, W, 3]

        into normalized PyTorch tensor:

            [T, 3, H, W]
        """

        frames = frames.astype(np.float32) / 255.0

        frames = (
            frames - IMAGENET_MEAN
        ) / IMAGENET_STD

        frames = np.transpose(
            frames,
            (0, 3, 1, 2)
        )

        return torch.from_numpy(
            frames.astype(np.float32)
        )

    # -------------------------------------------------------------
    # Flow preprocessing
    # -------------------------------------------------------------

    def _prepare_flow(self, flow: np.ndarray) -> torch.Tensor:
        """
        Convert optical flow:

            [T, 2, H, W]

        into a float32 PyTorch tensor.
        """

        return torch.from_numpy(
            flow.astype(np.float32)
        )

    # -------------------------------------------------------------
    # Temporal sampling
    # -------------------------------------------------------------

    def _sample_clip_indices(
        self,
        total_clips: int
    ) -> np.ndarray:
        """
        Uniformly select num_segments clips from the video.

        If the video has fewer clips than requested, indices are
        repeated so every sample has the same tensor shape.
        """

        if total_clips <= 0:
            raise ValueError(
                "Video contains no usable clips."
            )

        if total_clips >= self.num_segments:
            indices = np.linspace(
                0,
                total_clips - 1,
                self.num_segments
            ).round().astype(int)

        else:
            indices = np.linspace(
                0,
                total_clips - 1,
                self.num_segments
            ).round().astype(int)

        return indices

    # -------------------------------------------------------------
    # Main sample extraction
    # -------------------------------------------------------------

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row = self.df.iloc[index]

        video_id = str(row["video_id"])
        video_path = Path(str(row["path"]))

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found:\n{video_path}"
            )

        # ---------------------------------------------------------
        # Create project-native video reader
        # ---------------------------------------------------------

        reader = VideoReader(
            video_path=str(video_path),
            target_fps=self.target_fps,
            target_size=self.image_size,
        )

        # ---------------------------------------------------------
        # Create project-native optical flow engine
        # ---------------------------------------------------------

        flow_engine = OpticalFlowEngine(
            target_size=self.flow_size
        )

        # ---------------------------------------------------------
        # Read temporal clips
        # ---------------------------------------------------------

        clips = []

        for clip in reader.stream_clips(
            clip_length=16,
            stride=16
        ):
            clips.append(clip)

            # Prevent processing beyond configured duration.
            if (
                self.max_duration_seconds > 0
                and clip["end_time"] >= self.max_duration_seconds
            ):
                break

        if len(clips) == 0:
            raise RuntimeError(
                f"No temporal clips could be extracted from:\n"
                f"{video_path}"
            )

        # ---------------------------------------------------------
        # Uniformly select temporal clips
        # ---------------------------------------------------------

        selected_indices = self._sample_clip_indices(
            len(clips)
        )

        rgb_segments = []
        flow_segments = []

        selected_durations = []

        for clip_index in selected_indices:

            clip = clips[int(clip_index)]

            rgb_frames = clip["frames"]

            # -----------------------------------------------------
            # RGB
            # -----------------------------------------------------

            rgb_tensor = self._prepare_rgb(
                rgb_frames
            )

            # -----------------------------------------------------
            # Optical flow
            # -----------------------------------------------------

            flow_result = flow_engine.compute_clip_flow(
                rgb_frames
            )

            flow_tensor = self._prepare_flow(
                flow_result["flow"]
            )

            rgb_segments.append(
                rgb_tensor
            )

            flow_segments.append(
                flow_tensor
            )

            selected_durations.append(
                float(clip.get("duration", 0.0))
            )

        # ---------------------------------------------------------
        # Stack temporal segments
        #
        # RGB:
        #   [num_segments, T, 3, H, W]
        #
        # Flow:
        #   [num_segments, T, 2, Hf, Wf]
        # ---------------------------------------------------------

        rgb_tensor = torch.stack(
            rgb_segments,
            dim=0
        )

        flow_tensor = torch.stack(
            flow_segments,
            dim=0
        )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        label_id = int(
            row["label_id"]
        )

        is_anomaly = int(
            row["is_anomaly"]
        )

        # ---------------------------------------------------------
        # Final sample
        # ---------------------------------------------------------

        sample = {
            "video_id": video_id,

            "rgb": rgb_tensor,

            "flow": flow_tensor,

            "label_id": torch.tensor(
                label_id,
                dtype=torch.long
            ),

            "is_anomaly": torch.tensor(
                is_anomaly,
                dtype=torch.float32
            ),

            "duration": torch.tensor(
                float(reader.duration),
                dtype=torch.float32
            ),
        }

        # Add UCF metadata when available.

        if self.is_ucf:
            sample["class_name"] = str(
                row["class_name"]
            )

            sample["ucf_category"] = str(
                row["ucf_category"]
            )

            if "evaluation_type" in row.index:
                sample["evaluation_type"] = str(
                    row["evaluation_type"]
                )

        return sample

def get_dataloader(
    split_csv: str,
    root_dir: str = ".",
    batch_size: int = 4,
    num_segments: int = 12,
    is_training: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for UCF-Crime / CCTV surveillance data.

    Training:
        - Uses BalancedBatchSampler when both Normal and Anomaly
          classes are available.
        - This prevents the model from seeing mostly anomaly samples.

    Validation / testing:
        - Uses ordinary sequential DataLoader.
        - No oversampling is performed during evaluation.
    """

    dataset = SurveillanceVideoDataset(
        split_csv=split_csv,
        root_dir=root_dir,
        num_segments=num_segments,
        is_training=is_training,
    )

    # ---------------------------------------------------------
    # Training loader
    # ---------------------------------------------------------

    if is_training:

        metadata = pd.read_csv(split_csv)

        if "label" not in metadata.columns:
            raise ValueError(
                f"CSV must contain a 'label' column: {split_csv}"
            )

        labels = metadata["label"].astype(int).tolist()

        unique_labels = sorted(set(labels))

        # Use balanced batches only when both classes exist.
        if len(unique_labels) >= 2:

            sampler = BalancedBatchSampler(
                labels=labels,
                batch_size=batch_size,
                seed=42,
            )

            print(
                f"Using BalancedBatchSampler: "
                f"Normal={labels.count(0)}, "
                f"Anomaly={labels.count(1)}, "
                f"batch_size={batch_size}"
            )

            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,
            )

        # Fallback for anomaly-only training data
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    # ---------------------------------------------------------
    # Validation / test loader
    # ---------------------------------------------------------

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
