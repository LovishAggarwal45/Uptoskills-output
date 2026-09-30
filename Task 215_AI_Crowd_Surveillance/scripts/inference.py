#!/usr/bin/env python3
"""
CLI tool for running inference on untrimmed CCTV surveillance videos.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import SurveillanceInferencePipeline

def main():
    parser = argparse.ArgumentParser(description="Surveillance Video Inference CLI.")
    parser.add_argument("--video", type=str, required=True, help="Path to input CCTV video file.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda).")
    parser.add_argument("--no-video", action="store_true", help="Skip rendering annotated video.")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating timeline plot.")
    args = parser.parse_args()

    pipeline = SurveillanceInferencePipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    pipeline.process_video(
        video_path=args.video,
        output_annotated=not args.no_video,
        output_plot=not args.no_plot
    )

if __name__ == "__main__":
    main()
