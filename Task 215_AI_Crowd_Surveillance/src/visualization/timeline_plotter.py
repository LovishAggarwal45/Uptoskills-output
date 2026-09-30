"""
Surveillance Timeline & Probability Curve Plotter.
Generates publication-quality charts displaying temporal action probabilities,
anomaly score curves, ground-truth comparisons, and discrete event timelines.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

class TimelinePlotter:
    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_surveillance_timeline(
        self,
        video_id: str,
        timestamps: List[float],
        anomaly_scores: np.ndarray,
        action_probs: np.ndarray,
        detected_events: List[Dict[str, Any]],
        ground_truth_events: Optional[List[Dict[str, Any]]] = None,
        threshold: float = 0.60,
        classes: Optional[List[str]] = None
    ) -> str:
        """
        Renders a 3-panel surveillance diagnostic dashboard:
        Panel 1: Continuous Anomaly Probability vs. Time with Threshold & Detected Intervals
        Panel 2: Multi-class Action Distribution Over Time
        Panel 3: Discrete Event Timeline (Ground Truth vs. System Detections)
        """
        classes = classes or ["Normal", "Fighting", "Panic_Dispersal"]
        times = np.array(timestamps)
        duration = times[-1] if len(times) > 0 else 10.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.5, 1]})

        # --- Panel 1: Anomaly Score Curve ---
        ax1.plot(times, anomaly_scores, color="#D32F2F", linewidth=2.2, label="Anomaly Probability $S(t)$")
        ax1.axhline(threshold, color="#FF9800", linestyle="--", linewidth=1.5, label=f"Detection Threshold ($\theta={threshold}$)")
        
        # Shade detected anomaly regions
        for ev in detected_events:
            ax1.axvspan(ev["start_time"], ev["end_time"], color="#FFCDD2", alpha=0.5,
                        label="Detected Event" if ev == detected_events[0] else "")
            ax1.text((ev["start_time"] + ev["end_time"]) / 2, min(0.92, ev["confidence"] + 0.05),
                     f"{ev['action']}\n({ev['confidence']*100:.0f}%)",
                     ha="center", fontsize=8, weight="bold", color="#B71C1C")

        ax1.set_ylabel("Probability", fontsize=10, weight="bold")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(f"Weakly-Supervised Spatio-Temporal Action Localization: {video_id}", fontsize=13, weight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)

        # --- Panel 2: Multi-Class Probability Curves ---
        colors = ["#4CAF50", "#E53935", "#FB8C00"]
        for c_idx, c_name in enumerate(classes):
            ax2.plot(times, action_probs[:, c_idx], label=c_name, color=colors[c_idx % len(colors)], linewidth=1.8)

        ax2.set_ylabel("Class Prob", fontsize=10, weight="bold")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)

        # --- Panel 3: Discrete Event Gantt Bar ---
        ax3.set_ylim(0, 2.5)
        ax3.set_yticks([0.6, 1.8])
        ax3.set_yticklabels(["Model Detections", "Ground Truth"], weight="bold")
        ax3.set_xlabel("Video Time (seconds)", fontsize=10, weight="bold")
        ax3.grid(True, axis="x", linestyle=":", alpha=0.6)

        # Draw Ground Truth Bars
        if ground_truth_events:
            for gt in ground_truth_events:
                dur = gt["end_time"] - gt["start_time"]
                rect = patches.Rectangle((gt["start_time"], 1.4), dur, 0.7, color="#1976D2", alpha=0.85)
                ax3.add_patch(rect)
                ax3.text(gt["start_time"] + dur/2, 1.75, gt["label"], color="white", weight="bold", ha="center", va="center", fontsize=8)
        else:
            ax3.text(duration / 2, 1.75, "Normal / No Anomalies", ha="center", va="center", color="#555", style="italic")

        # Draw Model Detections
        if detected_events:
            for ev in detected_events:
                dur = ev["end_time"] - ev["start_time"]
                rect = patches.Rectangle((ev["start_time"], 0.2), dur, 0.7, color="#E53935", alpha=0.85)
                ax3.add_patch(rect)
                ax3.text(ev["start_time"] + dur/2, 0.55, f"{ev['action']}", color="white", weight="bold", ha="center", va="center", fontsize=8)
        else:
            ax3.text(duration / 2, 0.55, "Normal (No Anomalous Events Localized)", ha="center", va="center", color="#388E3C", style="italic")

        ax3.set_xlim(0, max(duration, 1.0))
        plt.tight_layout()

        plot_path = self.output_dir / f"{video_id}_timeline_dashboard.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return str(plot_path)
