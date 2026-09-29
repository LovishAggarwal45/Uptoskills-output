"""
Analytics & Report Generator Module.

Exports anomaly event reports to CSV, generates score timeline plots with threshold markings,
and renders cumulative spatial anomaly heatmaps.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class EventReportGenerator:
    """
    Handles CSV logging of detected crowd anomaly events and visual analytics generation.
    """
    def __init__(self, csv_path="outputs/anomaly_events.csv"):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Initialize CSV header if file does not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "frame_idx",
                    "location_bbox",
                    "detected_condition",
                    "confidence_score",
                    "risk_level"
                ])

    def log_event(self, timestamp, frame_idx, location_bbox, condition, confidence, risk_level):
        """Logs a single anomaly event to the CSV report."""
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_idx,
                str(location_bbox),
                condition,
                f"{confidence:.4f}",
                risk_level
            ])

    @staticmethod
    def plot_anomaly_timeline(frame_scores, threshold, fps=20, output_path="outputs/anomaly_timeline.png"):
        """
        Plots reconstruction MSE anomaly score over video time/frame index with anomaly threshold line.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        frames = np.arange(len(frame_scores))
        timestamps = frames / fps

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.plot(timestamps, frame_scores, color='#1f77b4', linewidth=2, label='VideoMAE Reconstruction MSE')
        ax.axhline(y=threshold, color='#d62728', linestyle='--', linewidth=2, label=f'Anomaly Threshold ({threshold:.4f})')

        # Highlight anomaly regions exceeding threshold
        anom_mask = np.array(frame_scores) > threshold
        ax.fill_between(timestamps, 0, frame_scores, where=anom_mask, color='#ff7f0e', alpha=0.35, label='Anomaly Event Window')

        ax.set_title('Crowd Outlier Localization - Reconstruction Error Timeline', fontsize=12, fontweight='bold')
        ax.set_xlabel('Video Time (seconds)', fontsize=10)
        ax.set_ylabel('Spatio-Temporal MSE Score', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[Analytics] Saved anomaly timeline plot to: {output_path}")

    @staticmethod
    def plot_spatial_heatmap_summary(cumulative_heatmap, video_shape, output_path="outputs/spatial_heatmap_summary.png"):
        """
        Plots cumulative spatial anomaly distribution across the surveillance field of view.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        H, W = video_shape[:2]
        # Smooth heatmap
        smoothed = gaussian_filter(cumulative_heatmap, sigma=2.0)
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = ax.imshow(smoothed, cmap='turbo', extent=[0, W, H, 0], aspect='auto')
        plt.colorbar(im, ax=ax, label='Relative Anomaly Intensity')

        ax.set_title('Cumulative Spatial Anomaly Heatmap (Surveillance FOV)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"[Analytics] Saved spatial summary heatmap to: {output_path}")
