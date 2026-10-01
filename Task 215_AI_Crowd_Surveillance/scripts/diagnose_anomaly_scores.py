"""
Diagnose raw anomaly scores produced by the trained WSSTAL model.

This script does NOT modify the model, checkpoint, threshold, or dataset.
It only reports anomaly scores for the test videos.
"""

import argparse
import csv
import sys
from pathlib import Path

# Add project root so "src" imports work when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.data.dataset import SurveillanceVideoDataset
from src.models.mil_classifier import TwoStreamWSSTALNet


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose raw anomaly scores on the test split."
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt"
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    print("=" * 70)
    print("ANOMALY SCORE DIAGNOSTIC")
    print("=" * 70)

    config = load_config(args.config)

    # ---------------------------------------------------------
    # Locate test split
    # ---------------------------------------------------------
    test_csv = (
        Path(config["dataset"]["splits_dir"])
        / "test.csv"
    )

    print(f"Test split : {test_csv}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    # ---------------------------------------------------------
    # Create test dataset using the project's actual Dataset API
    # ---------------------------------------------------------
    dataset = SurveillanceVideoDataset(
        split_csv=str(test_csv),
        root_dir=".",
        target_fps=config["video"]["target_fps"],
        clip_length=config["video"]["clip_length"],
        num_segments=12,
        image_size=tuple(config["video"]["image_size"]),
        flow_size=tuple(config["video"]["flow_size"]),
        is_training=False,
        use_cache=True,
    )

    print(f"Test videos: {len(dataset)}")

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------
    print("\nInitializing model...")

    model = TwoStreamWSSTALNet(
    num_classes=config["dataset"]["num_classes"],
    rgb_feature_dim=config["model"]["rgb_feature_dim"],
    flow_feature_dim=config["model"]["flow_feature_dim"],
    fusion_dim=config["model"]["fusion_dim"],
    encoder_layers=config["model"]["encoder_layers"],
    nheads=config["model"]["nheads"],
    dropout=config["model"]["dropout"],
    mil_top_k=config["model"]["mil_top_k"],
).to(device)
    print("Loading checkpoint...")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("Checkpoint loaded successfully.\n")

    # ---------------------------------------------------------
    # Evaluate every test video
    # ---------------------------------------------------------
    print("-" * 70)
    print(
        f"{'VIDEO':<25}"
        f"{'TRUE CLASS':<20}"
        f"{'ANOMALY SCORE':>15}"
    )
    print("-" * 70)

    results = []

    with torch.no_grad():

        for idx in range(len(dataset)):

            sample = dataset[idx]

            rgb = sample["rgb"].unsqueeze(0).to(device)
            flow = sample["flow"].unsqueeze(0).to(device)

            outputs = model(rgb, flow)

            score = float(
                outputs["video_anomaly_score"].item()
            )

            label_id = int(
                sample["label_id"].item()
            )

            class_names = config["dataset"]["classes"]

            true_class = class_names[label_id]

            video_id = str(
                sample["video_id"]
            )

            results.append(
                {
                    "video_id": video_id,
                    "true_class": true_class,
                    "label_id": label_id,
                    "anomaly_score": score,
                }
            )

            print(
                f"{video_id:<25}"
                f"{true_class:<20}"
                f"{score:>15.4f}"
            )

    print("-" * 70)

    # ---------------------------------------------------------
    # Separate normal and anomalous scores
    # ---------------------------------------------------------
    normal_scores = [
        r["anomaly_score"]
        for r in results
        if r["label_id"] == 0
    ]

    anomaly_scores = [
        r["anomaly_score"]
        for r in results
        if r["label_id"] != 0
    ]

    print("\nSUMMARY")
    print("=" * 70)

    if normal_scores:
        print(
            f"Normal videos     : "
            f"min={min(normal_scores):.4f}, "
            f"mean={sum(normal_scores) / len(normal_scores):.4f}, "
            f"max={max(normal_scores):.4f}"
        )

    if anomaly_scores:
        print(
            f"Anomaly videos    : "
            f"min={min(anomaly_scores):.4f}, "
            f"mean={sum(anomaly_scores) / len(anomaly_scores):.4f}, "
            f"max={max(anomaly_scores):.4f}"
        )

    threshold = config["localization"]["threshold"]

    print(f"\nCurrent threshold : {threshold:.2f}")

    normal_above = sum(
        score >= threshold
        for score in normal_scores
    )

    anomaly_above = sum(
        score >= threshold
        for score in anomaly_scores
    )

    print(
        f"Normal above threshold  : "
        f"{normal_above}/{len(normal_scores)}"
    )

    print(
        f"Anomaly above threshold : "
        f"{anomaly_above}/{len(anomaly_scores)}"
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------
    output_path = (
        Path("outputs")
        / "anomaly_score_diagnostic.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "true_class",
                "label_id",
                "anomaly_score",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nSaved diagnostic CSV: "
        f"{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()