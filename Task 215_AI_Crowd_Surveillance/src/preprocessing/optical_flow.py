"""
Optical Flow Engine for Motion Extraction in CCTV Surveillance.
Implements dense Farneback optical flow, motion magnitude/direction decomposition,
motion energy statistics, and color-wheel visualization.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any

class OpticalFlowEngine:
    def __init__(self, target_size: Tuple[int, int] = (112, 112), pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
        self.target_size = target_size
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

    def compute_flow_pair(self, prev_frame_rgb: np.ndarray, curr_frame_rgb: np.ndarray) -> np.ndarray:
        """
        Computes dense optical flow between two consecutive RGB frames.
        Returns: np.ndarray of shape [H_flow, W_flow, 2] with (u, v) motion displacements.
        """
        prev_gray = cv2.cvtColor(prev_frame_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_RGB2GRAY)

        if self.target_size is not None:
            prev_gray = cv2.resize(prev_gray, self.target_size, interpolation=cv2.INTER_AREA)
            curr_gray = cv2.resize(curr_gray, self.target_size, interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )
        return flow # [H, W, 2]

    def compute_clip_flow(self, frames_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Processes a clip of frames [T, H, W, 3].
        Returns a dict with:
          - 'flow': [T, 2, H_f, W_f] (transposed for PyTorch channel-first)
          - 'magnitude': [T, H_f, W_f]
          - 'mean_energy': [T] float motion energies
        """
        T = frames_rgb.shape[0]
        flows = []
        magnitudes = []
        energies = []

        # For the first frame, repeat the first computed flow to maintain length T
        if T == 1:
            h, w = self.target_size if self.target_size else (frames_rgb.shape[1], frames_rgb.shape[2])
            zero_flow = np.zeros((h, w, 2), dtype=np.float32)
            flows = [zero_flow]
            magnitudes = [np.zeros((h, w), dtype=np.float32)]
            energies = [0.0]
        else:
            for t in range(1, T):
                fl = self.compute_flow_pair(frames_rgb[t-1], frames_rgb[t])
                mag, _ = cv2.cartToPolar(fl[..., 0], fl[..., 1])
                flows.append(fl)
                magnitudes.append(mag)
                energies.append(float(np.mean(mag)))

            # Pad the initial frame with duplicate of first flow
            flows.insert(0, flows[0].copy())
            magnitudes.insert(0, magnitudes[0].copy())
            energies.insert(0, energies[0])

        flow_arr = np.stack(flows, axis=0) # [T, H_f, W_f, 2]
        flow_ch_first = np.transpose(flow_arr, (0, 3, 1, 2)) # [T, 2, H_f, W_f]
        mag_arr = np.stack(magnitudes, axis=0) # [T, H_f, W_f]
        energy_arr = np.array(energies, dtype=np.float32) # [T]

        return {
            "flow": flow_ch_first.astype(np.float32),
            "magnitude": mag_arr.astype(np.float32),
            "mean_energy": energy_arr
        }

    @staticmethod
    def flow_to_rgb(flow_hw2: np.ndarray) -> np.ndarray:
        """
        Visualizes dense optical flow [H, W, 2] using standard HSV color encoding.
        Hue represents motion direction, Value represents normalized motion magnitude.
        Returns: [H, W, 3] RGB uint8 image.
        """
        mag, ang = cv2.cartToPolar(flow_hw2[..., 0], flow_hw2[..., 1])
        hsv = np.zeros((flow_hw2.shape[0], flow_hw2.shape[1], 3), dtype=np.uint8)
        # Angle 0..2pi mapped to Hue 0..179
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        # Normalize magnitude to 0..255 with clipping
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
