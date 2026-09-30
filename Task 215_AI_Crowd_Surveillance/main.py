#!/usr/bin/env python3
"""
AI-Based Crowd Surveillance System - Master Unified CLI.
Entrypoint for Environment Inspection, Dataset Preparation, Feature Extraction,
Training, Evaluation, Inference, and Visualizations.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Crowd Surveillance System: Weakly-Supervised Spatio-Temporal Action Localization.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # 1. inspect-env
    subparsers.add_parser("inspect-env", help="Inspect Python, PyTorch, CUDA, OpenCV, hardware.")

    # 2. prepare-data
    p_data = subparsers.add_parser("prepare-data", help="Prepare dataset benchmark and splits.")
    p_data.add_argument("--benchmark-suite", action="store_true", help="Synthesize local CCTV benchmark suite.")

    # 3. compute-flow
    p_flow = subparsers.add_parser("compute-flow", help="Compute and visualize optical flow.")
    p_flow.add_argument("--video", type=str, required=True, help="Input video.")
    p_flow.add_argument("--output", type=str, default="outputs/visualizations", help="Output directory.")
    p_flow.add_argument("--max-frames", type=int, default=64, help="Max frames.")

    # 4. train
    p_train = subparsers.add_parser("train", help="Train weakly-supervised action model.")
    p_train.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML.")
    p_train.add_argument("--epochs", type=int, default=None, help="Epoch count.")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    p_train.add_argument("--device", type=str, default=None, help="Device (cpu/cuda).")

    # 5. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model against ground-truth test annotations.")
    p_eval.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint.")
    p_eval.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML.")

    # 6. inference
    p_inf = subparsers.add_parser("inference", help="Run end-to-end inference on an untrimmed CCTV video.")
    p_inf.add_argument("--video", type=str, required=True, help="Path to video.")
    p_inf.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML.")
    p_inf.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint.")
    p_inf.add_argument("--device", type=str, default=None, help="Device (cpu/cuda).")
    p_inf.add_argument("--no-video", action="store_true", help="Skip rendering annotated video.")
    p_inf.add_argument("--no-plot", action="store_true", help="Skip timeline plot.")

    args = parser.parse_args()

    if args.command == "inspect-env":
        from scripts.inspect_environment import inspect_environment
        inspect_environment()
    elif args.command == "prepare-data":
        from scripts.prepare_dataset import generate_benchmark_suite
        generate_benchmark_suite(Path("."))
    elif args.command == "compute-flow":
        from scripts.compute_optical_flow import compute_and_visualize_flow
        compute_and_visualize_flow(args.video, args.output, args.max_frames)
    elif args.command == "train":
        from scripts.train import main as train_main
        sys.argv = ["scripts/train.py", "--config", args.config]
        if args.epochs:
            sys.argv.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(["--batch-size", str(args.batch_size)])
        if args.device:
            sys.argv.extend(["--device", args.device])
        train_main()
    elif args.command == "evaluate":
        from scripts.evaluate import evaluate_test_set
        evaluate_test_set(args.checkpoint, args.config)
    elif args.command == "inference":
        from scripts.inference import main as inf_main
        sys.argv = ["scripts/inference.py", "--video", args.video, "--config", args.config, "--checkpoint", args.checkpoint]
        if args.device:
            sys.argv.extend(["--device", args.device])
        if args.no_video:
            sys.argv.append("--no-video")
        if args.no_plot:
            sys.argv.append("--no-plot")
        inf_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
