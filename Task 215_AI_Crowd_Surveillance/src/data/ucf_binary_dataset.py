#!/usr/bin/env python3

"""
UCF-Crime Binary Dataset

Binary labels:
    0 = Normal
    1 = Anomaly

This dataset class is intentionally separate from the existing
3-class surveillance dataset so previous experiments remain
reproducible.

Expected metadata columns:
    video_id
    path
    split
    label
    class_name
    source
    ucf_category
    evaluation_type
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UCFBinaryDataset(Dataset):
    """
    Dataset for binary UCF-Crime anomaly detection.

    Each video is converted into a fixed number of temporal
    RGB clips and optical-flow clips.

    Returns:
        video_id
        label
        is_anomaly
        rgb
        flow
        duration
        ucf_category
    """

    def __init__(
        self,
        metadata_file: str,
        split: str,
        num_segments: int = 12,
        clip_length: int = 16,
        stride: int = 16,
        image_size=(224, 224),
        flow_size=(112, 112),
        target_fps: int = 16,
        max_duration_seconds: float = 3600,
        is_training: bool = False,
    ):
        super().__init__()

        self.metadata_file = Path(metadata_file)

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}"
            )

        self.split = split
        self.num_segments = num_segments
        self.clip_length = clip_length
        self.stride = stride
        self.image_size = tuple(image_size)
        self.flow_size = tuple(flow_size)
        self.target_fps = target_fps
        self.max_duration_seconds = max_duration_seconds
        self.is_training = is_training

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        self.metadata = pd.read_csv(self.metadata_file)

        required_columns = {
            "video_id",
            "path",
            "split",
            "label",
            "class_name",
            "ucf_category",
        }

        missing = required_columns - set(self.metadata.columns)

        if missing:
            raise ValueError(
                "Missing required metadata columns: "
                + ", ".join(sorted(missing))
            )

        # ----------------------------------------------------
        # Filter requested split
        # ----------------------------------------------------

        self.metadata = self.metadata[
            self.metadata["split"].astype(str).str.lower() == split.lower()
        ].copy()

        self.metadata = self.metadata.reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"No samples found for split='{split}' "
                f"in {self.metadata_file}"
            )

        # ----------------------------------------------------
        # Validate labels
        # ----------------------------------------------------

        labels = set(
            self.metadata["label"]
            .astype(int)
            .unique()
            .tolist()
        )

        invalid_labels = labels - {0, 1}

        if invalid_labels:
            raise ValueError(
                f"Invalid binary labels found: {invalid_labels}"
            )

        # ----------------------------------------------------
        # Simple cache
        # ----------------------------------------------------

        self._video_cache: Dict[str, Dict] = {}

        print(
            f"[UCFBinaryDataset] split={split} "
            f"samples={len(self.metadata)}"
        )

        print(
            f"[UCFBinaryDataset] "
            f"normal={(self.metadata['label'] == 0).sum()} "
            f"anomaly={(self.metadata['label'] == 1).sum()}"
        )

    # ========================================================
    # BASIC METHODS
    # ========================================================

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        video_id = str(row["video_id"])
        video_path = Path(str(row["path"]))

        label = int(row["label"])

        ucf_category = str(row["ucf_category"])

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found:\n{video_path}"
            )

        video_data = self._load_video(video_path)

        rgb_clips = self._sample_rgb_clips(
            video_data["frames"]
        )

        flow_clips = self._sample_flow_clips(
            video_data["frames"]
        )

        rgb_tensor = self._prepare_rgb_tensor(
            rgb_clips
        )

        flow_tensor = self._prepare_flow_tensor(
            flow_clips
        )

        return {
            "video_id": video_id,

            "label": torch.tensor(
                label,
                dtype=torch.long
            ),

            "is_anomaly": torch.tensor(
                float(label),
                dtype=torch.float32
            ),

            "rgb": rgb_tensor,

            "flow": flow_tensor,

            "duration": torch.tensor(
                video_data["duration"],
                dtype=torch.float32
            ),

            "ucf_category": ucf_category,
        }

    # ========================================================
    # VIDEO LOADING
    # ========================================================

    def _load_video(self, video_path: Path):

        cache_key = str(video_path.resolve())

        if cache_key in self._video_cache:
            return self._video_cache[cache_key]

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video:\n{video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = frame_count / fps

        # Limit extremely long videos.
        max_frames = int(
            min(
                frame_count,
                self.max_duration_seconds * fps
            )
        )

        frames = []

        # ----------------------------------------------------
        # Read video frames.
        #
        # To keep memory reasonable, sample approximately
        # target_fps rather than loading every original frame.
        # ----------------------------------------------------

        sample_step = max(
            1,
            int(round(fps / self.target_fps))
        )

        frame_index = 0

        while frame_index < max_frames:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index
            )

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

            frame_index += sample_step

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"No frames could be read from:\n{video_path}"
            )

        data = {
            "frames": frames,
            "duration": duration,
        }

        self._video_cache[cache_key] = data

        return data

    # ========================================================
    # TEMPORAL SAMPLING
    # ========================================================

    def _sample_start_indices(
        self,
        total_frames: int
    ) -> List[int]:

        required_frames = self.clip_length

        if total_frames <= required_frames:
            return [0] * self.num_segments

        max_start = total_frames - required_frames

        if self.num_segments == 1:
            return [max_start // 2]

        starts = np.linspace(
            0,
            max_start,
            self.num_segments
        )

        starts = np.round(starts).astype(int)

        return starts.tolist()

    def _sample_rgb_clips(
        self,
        frames: List[np.ndarray]
    ):

        starts = self._sample_start_indices(
            len(frames)
        )

        clips = []

        for start in starts:

            clip = []

            for offset in range(self.clip_length):

                frame_index = min(
                    start + offset,
                    len(frames) - 1
                )

                frame = frames[frame_index]

                frame = cv2.resize(
                    frame,
                    self.image_size,
                    interpolation=cv2.INTER_LINEAR
                )

                clip.append(frame)

            clips.append(
                np.stack(clip)
            )

        return clips

    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    def _calculate_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ):

        gray1 = cv2.cvtColor(
            frame1,
            cv2.COLOR_RGB2GRAY
        )

        gray2 = cv2.cvtColor(
            frame2,
            cv2.COLOR_RGB2GRAY
        )

        gray1 = cv2.resize(
            gray1,
            self.flow_size,
            interpolation=cv2.INTER_LINEAR
        )

        gray2 = cv2.resize(
            gray2,
            self.flow_size,
            interpolation=cv2.INTER_LINEAR
        )

        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        return flow.astype(np.float32)

    def _sample_flow_clips(
        self,
        frames: List[np.ndarray]
    ):

        starts = self._sample_start_indices(
            len(frames)
        )

        flow_clips = []

        for start in starts:

            clip_flow = []

            for offset in range(
                self.clip_length - 1
            ):

                idx1 = min(
                    start + offset,
                    len(frames) - 1
                )

                idx2 = min(
                    start + offset + 1,
                    len(frames) - 1
                )

                flow = self._calculate_flow(
                    frames[idx1],
                    frames[idx2]
                )

                clip_flow.append(flow)

            # If the clip is too short, pad with zeros.
            while len(clip_flow) < self.clip_length - 1:

                clip_flow.append(
                    np.zeros(
                        (
                            self.flow_size[1],
                            self.flow_size[0],
                            2,
                        ),
                        dtype=np.float32,
                    )
                )

            flow_clips.append(
                np.stack(clip_flow)
            )

        return flow_clips

    # ========================================================
    # TENSOR PREPARATION
    # ========================================================

    def _prepare_rgb_tensor(
        self,
        clips
    ):

        tensors = []

        for clip in clips:

            clip = clip.astype(
                np.float32
            ) / 255.0

            # ImageNet normalization.
            mean = np.array(
                [0.485, 0.456, 0.406],
                dtype=np.float32
            )

            std = np.array(
                [0.229, 0.224, 0.225],
                dtype=np.float32
            )

            clip = (
                clip - mean
            ) / std

            # T,H,W,C -> T,C,H,W
            clip = np.transpose(
                clip,
                (0, 3, 1, 2)
            )

            tensors.append(
                torch.from_numpy(
                    clip.copy()
                )
            )

        # N,T,C,H,W
        return torch.stack(tensors)

    def _prepare_flow_tensor(
        self,
        clips
    ):

        tensors = []

        for clip in clips:

            # T,H,W,2 -> T,2,H,W
            clip = np.transpose(
                clip,
                (0, 3, 1, 2)
            )

            # Normalize flow values approximately.
            clip = np.clip(
                clip,
                -20.0,
                20.0
            ) / 20.0

            tensors.append(
                torch.from_numpy(
                    clip.astype(
                        np.float32
                    ).copy()
                )
            )

        # N,T,2,H,W
        return torch.stack(tensors)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    dataset = UCFBinaryDataset(
        metadata_file=(
            "datasets/metadata/"
            "ucf_binary_train.csv"
        ),
        split="train",
        num_segments=4,
        clip_length=16,
    )

    print("\nDataset length:", len(dataset))

    sample = dataset[0]

    print("\nSample:")
    print("video_id :", sample["video_id"])
    print("label    :", sample["label"].item())
    print("anomaly  :", sample["is_anomaly"].item())
    print("category :", sample["ucf_category"])
    print("duration :", sample["duration"].item())

    print("\nRGB shape:")
    print(sample["rgb"].shape)

    print("\nFlow shape:")
    print(sample["flow"].shape)