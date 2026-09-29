"""
Crowd Outlier Detector & Real-Time Heatmap Renderer.

Uses VideoMAE reconstruction error to localise crowd anomalies in video feeds,
overlays pixel-level spatio-temporal heatmaps, draws bounding boxes around high-risk regions,
triggers automated alerts, and logs event statistics.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import cv2
import yaml
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

from models.videomae import VideoMAE
from dataset.video_dataset import load_video_frames
from analytics.report_generator import EventReportGenerator


class CrowdOutlierDetector:
    """
    Inference & Visualization Engine for VideoMAE Crowd Outlier Localization.
    """
    def __init__(self, config_path="config.yaml", checkpoint_path=None):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = self.cfg["system"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        if checkpoint_path is None:
            checkpoint_path = self.cfg["training"]["checkpoint_path"]

        # Load VideoMAE model
        self.model = VideoMAE(
            img_size=tuple(self.cfg["data"]["image_size"]),
            patch_size=self.cfg["data"]["patch_size"],
            num_frames=self.cfg["data"]["num_frames"],
            tubelet_size=self.cfg["data"]["tubelet_size"],
            in_chans=self.cfg["data"]["channels"],
            embed_dim=self.cfg["model"]["embed_dim"],
            encoder_depth=self.cfg["model"]["encoder_depth"],
            encoder_heads=self.cfg["model"]["encoder_heads"],
            decoder_embed_dim=self.cfg["model"]["decoder_embed_dim"],
            decoder_depth=self.cfg["model"]["decoder_depth"],
            decoder_heads=self.cfg["model"]["decoder_heads"],
            masking_ratio=self.cfg["model"]["masking_ratio"],
            mlp_ratio=self.cfg["model"]["mlp_ratio"]
        ).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}. Train the model first!")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.threshold = self.cfg["detection"]["anomaly_threshold"]
        self.alpha = self.cfg["detection"]["heatmap_alpha"]
        self.report_gen = EventReportGenerator(csv_path=self.cfg["output"]["csv_report"])
        self.last_alert_frame = -100

        # Calibrate baseline normality reconstruction MSE
        self.calibrate_normality_baseline()

    def calibrate_normality_baseline(self, normal_video_path="data/sample_crowd_normal.mp4"):
        """Calibrates baseline reconstruction MSE mean on normal crowd video."""
        if not os.path.exists(normal_video_path):
            self.baseline_mean = 0.0128
            return

        raw_frames = load_video_frames(normal_video_path, target_size=None)
        num_clip_frames = self.cfg["data"]["num_frames"]
        stride = self.cfg["data"]["stride"]
        model_size = tuple(self.cfg["data"]["image_size"])

        scores = []
        with torch.no_grad():
            for start_idx in range(0, len(raw_frames) - num_clip_frames + 1, stride * 2):
                clip_frames = raw_frames[start_idx : start_idx + num_clip_frames]
                resized_clip = [cv2.resize(f, (model_size[1], model_size[0])) for f in clip_frames]
                clip_tensor = torch.from_numpy(np.array(resized_clip)).float() / 255.0
                clip_tensor = clip_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
                
                amap_3d, _, _ = self.model.compute_spatiotemporal_anomaly_map(clip_tensor, num_mask_passes=1)
                for t_idx in range(amap_3d.shape[1]):
                    scores.append(float(np.percentile(amap_3d[0, t_idx].cpu().numpy(), 90)))

        self.baseline_mean = float(np.mean(scores)) + 1e-6

    def process_video(self, video_path, output_path=None):
        """
        Processes full video file, calculates 3D spatio-temporal MSE heatmaps,
        annotates frames, logs alerts, and saves output video & analytics plots.
        """
        if output_path is None:
            output_path = self.cfg["output"]["annotated_video"]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Load raw video frames
        raw_frames = load_video_frames(video_path, target_size=None)
        num_raw = len(raw_frames)

        num_clip_frames = self.cfg["data"]["num_frames"]
        stride = self.cfg["data"]["stride"]
        model_size = tuple(self.cfg["data"]["image_size"])

        frame_scores = np.zeros(num_raw, dtype=np.float32)
        frame_counts = np.zeros(num_raw, dtype=np.float32)
        cumulative_heatmap = np.zeros((height, width), dtype=np.float32)

        frame_heatmaps = [np.zeros((height, width), dtype=np.float32) for _ in range(num_raw)]

        # Sliding window evaluation
        for start_idx in range(0, num_raw - num_clip_frames + 1, stride):
            clip_frames = raw_frames[start_idx : start_idx + num_clip_frames]
            
            resized_clip = [cv2.resize(f, (model_size[1], model_size[0])) for f in clip_frames]
            clip_tensor = torch.from_numpy(np.array(resized_clip)).float() / 255.0
            clip_tensor = clip_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

            amap_3d, _, _ = self.model.compute_spatiotemporal_anomaly_map(clip_tensor, num_mask_passes=2)
            amap_3d = amap_3d[0].cpu().numpy()

            grid_t, grid_h, grid_w = amap_3d.shape
            t_ratio = num_clip_frames // grid_t

            for t_idx in range(grid_t):
                patch_map = amap_3d[t_idx]
                spatial_hm = cv2.resize(patch_map, (width, height), interpolation=cv2.INTER_CUBIC)
                spatial_hm = gaussian_filter(spatial_hm, sigma=self.cfg["detection"]["smoothing_sigma"])
                
                peak_score = float(np.percentile(spatial_hm, 90))

                for f_rel in range(t_idx * t_ratio, (t_idx + 1) * t_ratio):
                    f_abs = start_idx + f_rel
                    if f_abs < num_raw:
                        frame_scores[f_abs] += peak_score
                        frame_counts[f_abs] += 1.0
                        frame_heatmaps[f_abs] += spatial_hm

        # Normalize across overlapping windows
        for i in range(num_raw):
            if frame_counts[i] > 0:
                frame_scores[i] /= frame_counts[i]
                frame_heatmaps[i] /= frame_counts[i]
            cumulative_heatmap += frame_heatmaps[i]

        # Calculate Relative Anomaly Score Ratio (R = Score / Normality_Baseline_Mean)
        rel_anomaly_scores = frame_scores / self.baseline_mean

        # Prepare VideoWriter for output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cooldown = self.cfg["detection"]["alert_cooldown_frames"]

        for f_idx in range(num_raw):
            orig_frame = raw_frames[f_idx].copy()
            score = float(rel_anomaly_scores[f_idx])
            hm = frame_heatmaps[f_idx]

            # Normalize frame heatmap for visualization
            hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(hm_norm, cv2.COLORMAP_TURBO)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            blended = cv2.addWeighted(orig_frame, 1.0 - self.alpha, heatmap_colored, self.alpha, 0)
            annotated = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            is_anomaly = score >= self.threshold
            risk_level = "HIGH RISK" if is_anomaly else "NORMAL"

            if is_anomaly:
                thresh_val = np.percentile(hm_norm, 85)
                _, mask_bin = cv2.threshold(hm_norm, int(thresh_val), 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detected_boxes = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > (width * height * 0.005):
                        x, y, w, h_box = cv2.boundingRect(cnt)
                        cv2.rectangle(annotated, (x, y), (x + w, y + h_box), (0, 0, 255), 2)
                        cv2.putText(annotated, f"OUTLIER ({score:.2f}x)", (x, max(y - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        detected_boxes.append([x, y, w, h_box])

                if (f_idx - self.last_alert_frame) >= cooldown:
                    self.last_alert_frame = f_idx
                    timestamp_str = time.strftime("%H:%M:%S", time.gmtime(f_idx / fps))
                    bbox_summary = detected_boxes if len(detected_boxes) > 0 else [[0, 0, width, height]]
                    self.report_gen.log_event(
                        timestamp=timestamp_str,
                        frame_idx=f_idx,
                        location_bbox=bbox_summary[0],
                        condition="Spatio-Temporal Crowd Motion Outlier",
                        confidence=score,
                        risk_level=risk_level
                    )

            # HUD & Overlay Visuals
            banner_color = (0, 0, 220) if is_anomaly else (40, 40, 40)
            cv2.rectangle(annotated, (0, 0), (width, 35), banner_color, -1)

            time_str = time.strftime("%H:%M:%S", time.gmtime(f_idx / fps))
            status_text = f"TIME: {time_str} | FRAME: {f_idx:03d} | SCORE: {score:.2f}x | STATUS: {risk_level}"
            cv2.putText(annotated, status_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

            # Risk Gauge Meter Bar (Bottom Left)
            gauge_w = 150
            gauge_h = 12
            gx, gy = 15, height - 25
            cv2.rectangle(annotated, (gx, gy), (gx + gauge_w, gy + gauge_h), (50, 50, 50), -1)
            
            fill_ratio = min(max(score - 1.0, 0.0) / (self.threshold - 1.0 + 1e-5), 1.0)
            fill_w = int(gauge_w * fill_ratio)
            gauge_color = (0, 0, 255) if is_anomaly else (0, 220, 0)
            cv2.rectangle(annotated, (gx, gy), (gx + fill_w, gy + gauge_h), gauge_color, -1)
            cv2.rectangle(annotated, (gx, gy), (gx + gauge_w, gy + gauge_h), (200, 200, 200), 1)
            cv2.putText(annotated, "RISK LEVEL", (gx, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

            writer.write(annotated)

        writer.release()

        # Generate summary analytics plots
        self.report_gen.plot_anomaly_timeline(
            frame_scores=rel_anomaly_scores,
            threshold=self.threshold,
            fps=fps,
            output_path=self.cfg["output"]["timeline_plot"]
        )

        self.report_gen.plot_spatial_heatmap_summary(
            cumulative_heatmap=cumulative_heatmap,
            video_shape=(height, width),
            output_path=self.cfg["output"]["spatial_summary"]
        )

        return output_path, rel_anomaly_scores
