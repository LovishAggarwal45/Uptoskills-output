"""
Spatial Localization and Attention Engine.
Implements motion-energy spatial region-of-interest (ROI) bounding box extraction
and Class Activation Mapping (CAM) heatmaps.
Explicitly distinguishes pseudo-localization / attention from ground truth.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

class SpatialLocalizer:
    def __init__(self, min_box_area_ratio: float = 0.005, threshold_ratio: float = 0.35):
        self.min_box_area_ratio = min_box_area_ratio
        self.threshold_ratio = threshold_ratio

    def extract_motion_rois(
        self,
        optical_flow: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Extracts spatial pseudo-localization bounding boxes from optical flow motion field.
        optical_flow: [H_f, W_f, 2] or [2, H_f, W_f]
        frame_shape: (H_orig, W_orig)
        Returns: list of bounding boxes with normalized and pixel coordinates.
        """
        if optical_flow.shape[0] == 2:
            flow_hw2 = np.transpose(optical_flow, (1, 2, 0))
        else:
            flow_hw2 = optical_flow

        H_orig, W_orig = frame_shape
        H_flow, W_flow = flow_hw2.shape[:2]

        # Compute flow magnitude
        mag, _ = cv2.cartToPolar(flow_hw2[..., 0], flow_hw2[..., 1])
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold high motion regions
        thresh_val = int(255 * self.threshold_ratio)
        _, binary = cv2.threshold(mag_norm, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological close to bridge adjacent motion blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scale_x = W_orig / float(W_flow)
        scale_y = H_orig / float(H_flow)
        min_area = (W_orig * H_orig) * self.min_box_area_ratio

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Map back to original frame coordinates
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w = int(w * scale_x)
            orig_h = int(h * scale_y)

            if (orig_w * orig_h) >= min_area:
                boxes.append({
                    "bbox_pixel": (orig_x, orig_y, orig_w, orig_h),
                    "bbox_norm": (
                        round(orig_x / W_orig, 4),
                        round(orig_y / H_orig, 4),
                        round(orig_w / W_orig, 4),
                        round(orig_h / H_orig, 4)
                    ),
                    "is_ground_truth": False,
                    "method": "motion_energy_pseudo_localization"
                })

        return boxes

    def generate_heatmap_overlay(
        self,
        frame_bgr: np.ndarray,
        optical_flow: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Generates a semi-transparent spatio-temporal attention heatmap overlay on top of frame_bgr.
        """
        if optical_flow.shape[0] == 2:
            flow_hw2 = np.transpose(optical_flow, (1, 2, 0))
        else:
            flow_hw2 = optical_flow

        mag, _ = cv2.cartToPolar(flow_hw2[..., 0], flow_hw2[..., 1])
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Resize heatmap to match frame
        H, W = frame_bgr.shape[:2]
        heatmap_resized = cv2.resize(mag_norm, (W, H), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        return overlay
