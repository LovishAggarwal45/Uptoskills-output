#!/usr/bin/env python3
"""
CLI wrapper for running surveillance evaluation against ground-truth annotations.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import evaluate_test_set

def main():
    parser = argparse.ArgumentParser(description="Evaluate WSSTAL surveillance model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--output-dir", type=str, default="reports/metrics", help="Output directory.")
    args = parser.parse_args()

    evaluate_test_set(args.checkpoint, args.config, args.output_dir)

if __name__ == "__main__":
    main()
