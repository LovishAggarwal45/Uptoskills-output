#!/usr/bin/env python3
"""
CLI tool for training the Weakly Supervised Action Localization System.

Supports:
1. Existing CCTV 3-class project
2. UCF-Crime binary anomaly experiment
"""

import sys
import yaml
import argparse
import random
import numpy as np
import torch

from pathlib import Path

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.data.dataset import get_dataloader
from src.training.trainer import ModelTrainer


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Train Weakly-Supervised Action Model."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (cpu or cuda)."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: "
            f"{config_path}"
        )

    # --------------------------------------------------------
    # Apply command-line overrides
    # --------------------------------------------------------

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        config.get(
            "runtime",
            {}
        ).get(
            "seed",
            42
        )
    )

    # ========================================================
    # Dataset split paths
    # ========================================================

    dataset_config = config.get(
        "dataset",
        {}
    )

    dataset_name = dataset_config.get(
        "name",
        ""
    )

    # --------------------------------------------------------
    # UCF-Crime
    # --------------------------------------------------------

    if dataset_name == "ucf_crime_binary":

        metadata_file = dataset_config.get(
            "metadata_file"
        )

        if not metadata_file:
            raise KeyError(
                "UCF-Crime configuration requires "
                "'dataset.metadata_file'."
            )

        metadata_path = Path(
            metadata_file
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"UCF metadata file not found: "
                f"{metadata_path}"
            )

        metadata_dir = metadata_path.parent

        train_csv = (
            metadata_dir
            / "ucf_binary_train.csv"
        )

        val_csv = (
            metadata_dir
            / "ucf_binary_val.csv"
        )

        test_csv = (
            metadata_dir
            / "ucf_binary_test.csv"
        )

    # --------------------------------------------------------
    # Existing CCTV project
    # --------------------------------------------------------

    else:

        splits_dir = dataset_config.get(
            "splits_dir"
        )

        if not splits_dir:
            raise KeyError(
                "Dataset configuration requires "
                "'dataset.splits_dir'."
            )

        splits_path = Path(
            splits_dir
        )

        train_csv = (
            splits_path
            / "train.csv"
        )

        val_csv = (
            splits_path
            / "val.csv"
        )

        test_csv = (
            splits_path
            / "test.csv"
        )

    # ========================================================
    # Validate split files
    # ========================================================

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Training split not found:\n"
            f"{train_csv}"
        )

    if not val_csv.exists():
        raise FileNotFoundError(
            f"Validation split not found:\n"
            f"{val_csv}"
        )

    # ========================================================
    # Print experiment information
    # ========================================================

    print()
    print("=" * 60)
    print("WEAKLY SUPERVISED ACTION LOCALIZATION")
    print("=" * 60)

    print(
        f"Configuration : {config_path}"
    )

    print(
        f"Dataset       : {dataset_name}"
    )

    print(
        f"Device        : "
        f"{args.device or 'auto'}"
    )

    print(
        f"Train split   : {train_csv}"
    )

    print(
        f"Validation    : {val_csv}"
    )

    if test_csv.exists():
        print(
            f"Test split    : {test_csv}"
        )

    print("=" * 60)
    print()

    # ========================================================
    # Training DataLoader
    # ========================================================

    print(
        f"Loading training data from: "
        f"{train_csv}"
    )

    train_loader = get_dataloader(
        split_csv=str(train_csv),
        batch_size=config["training"]["batch_size"],
        num_segments=12,
        is_training=True
    )

    # ========================================================
    # Validation DataLoader
    # ========================================================

    print(
        f"Loading validation data from: "
        f"{val_csv}"
    )

    val_loader = get_dataloader(
        split_csv=str(val_csv),
        batch_size=config["training"]["batch_size"],
        num_segments=12,
        is_training=False
    )

    # ========================================================
    # Model Trainer
    # ========================================================

    print()
    print("Initializing ModelTrainer...")

    trainer = ModelTrainer(
        config=config,
        device=args.device
    )

    # ========================================================
    # Training
    # ========================================================

    print()
    print(
        f"Beginning training for "
        f"{config['training']['epochs']} epochs..."
    )

    trainer.fit(
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"]
    )

    print()
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
