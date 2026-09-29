"""
Video Dataset Loader and Spatio-Temporal Clip Sampler.

Extracts continuous spatio-temporal video clips (C, T, H, W) using sliding window sampling.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_video_frames(video_path, target_size=(128, 128)):
    """
    Loads all frames from a video file into an RGB numpy array (N, H, W, C).
    
    Args:
        video_path (str): Path to video file.
        target_size (tuple): Optional (H, W) resizing target.
        
    Returns:
        frames (np.ndarray): Array of shape (N, H, W, 3) in RGB format.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frames.append(frame)

    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    return np.array(frames)


class VideoClipDataset(Dataset):
    """
    PyTorch Dataset for sliding-window spatio-temporal video clips.
    """
    def __init__(self, video_paths, num_frames=16, stride=4, image_size=(128, 128)):
        super().__init__()
        self.num_frames = num_frames
        self.stride = stride
        self.image_size = image_size
        self.clips = []

        for vpath in video_paths:
            frames = load_video_frames(vpath, target_size=image_size)
            num_total_frames = len(frames)
            
            # Extract sliding windows
            for start_idx in range(0, num_total_frames - num_frames + 1, stride):
                clip_frames = frames[start_idx : start_idx + num_frames]
                self.clips.append(clip_frames)

        if len(self.clips) == 0:
            raise ValueError("Insufficient video length to create at least one clip!")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        # Clip shape: (T, H, W, C) uint8 -> Convert to float (C, T, H, W) in [0, 1]
        clip = self.clips[idx]
        clip = torch.from_numpy(clip).float() / 255.0  # (T, H, W, C)
        clip = clip.permute(3, 0, 1, 2)               # (C, T, H, W)
        return clip
