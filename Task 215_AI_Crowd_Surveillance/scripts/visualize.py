#!/usr/bin/env python3
"""
CLI tool for generating surveillance diagnostic plots and timeline charts.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.timeline_plotter import TimelinePlotter

def main():
    parser = argparse.ArgumentParser(description="Timeline visualization generator.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to JSON predictions file.")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations", help="Output directory.")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    with open(pred_path, "r") as f:
        data = json.load(f)

    plotter = TimelinePlotter(output_dir=args.output_dir)
    res_path = plotter.plot_surveillance_timeline(
        video_id=data["video_id"],
        timestamps=data["timestamps"],
        anomaly_scores=np.array(data["anomaly_scores"]),
        action_probs=np.array(data["action_probs"]),
        detected_events=data["detected_events"],
        ground_truth_events=data.get("ground_truth_events"),
        threshold=data.get("threshold", 0.60)
    )
    print(f"Visualization saved to: {res_path}")

if __name__ == "__main__":
    main()
