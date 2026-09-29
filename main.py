"""
Unified CLI Entry Point for VideoMAE Crowd Outlier Localization System.

Usage:
  python main.py --mode run_all
  python main.py --mode train --video_path data/sample_crowd_normal.mp4
  python main.py --mode detect --video_path data/sample_crowd_test.mp4
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from dataset.generate_sample_video import SyntheticCrowdGenerator
from trainer import train_videomae
from detector.outlier_detector import CrowdOutlierDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Supervised VideoMAE Crowd Outlier Localization System")
    parser.add_argument(
        "--mode",
        type=str,
        default="run_all",
        choices=["generate_data", "train", "detect", "run_all"],
        help="Pipeline execution mode: generate_data, train, detect, or run_all (default)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--video_path", type=str, default=None, help="Input video file path for training/detection")
    parser.add_argument("--output_path", type=str, default=None, help="Output annotated video path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("      AI CROWD SURVEILLANCE SYSTEM: VIDEOMAE OUTLIER LOCALIZATION     ")
    print("=" * 70)

    # Mode 1: Generate Synthetic Benchmark Videos
    if args.mode in ["generate_data", "run_all"]:
        print("\n[STEP 1/3] Generating Synthetic Crowd Benchmark Videos...")
        gen = SyntheticCrowdGenerator()
        gen.generate_normal_video("data/sample_crowd_normal.mp4", total_frames=160)
        gen.generate_anomaly_video("data/sample_crowd_test.mp4", total_frames=240)

    # Mode 2: Train VideoMAE on Normal Crowd Video
    if args.mode in ["train", "run_all"]:
        print("\n[STEP 2/3] Training Self-Supervised VideoMAE on Normal Crowd Dynamics...")
        train_videos = [args.video_path] if args.video_path else ["data/sample_crowd_normal.mp4"]
        train_videomae(config_path=args.config, video_paths=train_videos, epochs=args.epochs)

    # Mode 3: Detect Anomalies & Overlay Heatmaps
    if args.mode in ["detect", "run_all"]:
        print("\n[STEP 3/3] Running Outlier Localization, Heatmap Rendering & Alerting...")
        test_video = args.video_path if args.video_path else "data/sample_crowd_test.mp4"
        detector = CrowdOutlierDetector(config_path=args.config)
        output_vpath, scores = detector.process_video(video_path=test_video, output_path=args.output_path)

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE!")
        print("=" * 70)
        print(f"1. Output Annotated Video: {output_vpath}")
        print(f"2. CSV Event Report:     outputs/anomaly_events.csv")
        print(f"3. Timeline Plot:        outputs/anomaly_timeline.png")
        print(f"4. Spatial Heatmap:      outputs/spatial_heatmap_summary.png")
        print("=" * 70)


if __name__ == "__main__":
    main()
