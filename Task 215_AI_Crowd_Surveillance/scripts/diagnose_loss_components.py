"""
Diagnostic evaluation of individual WSSTAL loss components.

Loads an existing checkpoint and evaluates the validation split
without retraining the model.
"""

import argparse
from pathlib import Path

import torch
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import torch

from src.data.dataset import get_dataloader
from src.models.mil_classifier import TwoStreamWSSTALNet
from src.models.loss import WSSTALLoss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--split",
        default="datasets/splits/val.csv"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("WSSTAL LOSS COMPONENT DIAGNOSTIC")
    print("=" * 70)

    print(f"Split      : {args.split}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------
    dataloader = get_dataloader(
        split_csv=args.split,
        root_dir=".",
        batch_size=4,
        num_segments=12,
        is_training=False,
        num_workers=0
    )

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------
    m_cfg = config.get("model", {})
    d_cfg = config.get("dataset", {})
    t_cfg = config.get("training", {})

    print("\nInitializing model...")

    model = TwoStreamWSSTALNet(
        num_classes=d_cfg.get("num_classes", 3),
        rgb_feature_dim=m_cfg.get(
            "rgb_feature_dim",
            512
        ),
        flow_feature_dim=m_cfg.get(
            "flow_feature_dim",
            128
        ),
        fusion_dim=m_cfg.get(
            "fusion_dim",
            256
        ),
        encoder_layers=m_cfg.get(
            "encoder_layers",
            2
        ),
        nheads=m_cfg.get(
            "nheads",
            4
        ),
        dropout=m_cfg.get(
            "dropout",
            0.15
        ),
        mil_top_k=m_cfg.get(
            "mil_top_k",
            3
        ),
        pretrained=m_cfg.get(
            "pretrained",
            True
        ),
        freeze_rgb_backbone=m_cfg.get(
            "freeze_rgb_backbone",
            False
        )
    ).to(device)

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------
    checkpoint_path = Path(args.checkpoint)

    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # ---------------------------------------------------------
    # Loss function
    # ---------------------------------------------------------
    criterion = WSSTALLoss(
        smoothness_weight=t_cfg.get(
            "smoothness_weight",
            0.0001
        ),
        sparsity_weight=t_cfg.get(
            "sparsity_weight",
            0.0001
        ),
        ranking_margin=t_cfg.get(
            "ranking_margin",
            0.20
        ),
        classification_weight=t_cfg.get(
            "classification_weight",
            1.0
        ),
        anomaly_weight=t_cfg.get(
            "anomaly_weight",
            2.0
        ),
        ranking_weight=t_cfg.get(
            "ranking_weight",
            1.0
        )
    )

    # ---------------------------------------------------------
    # Accumulators
    # ---------------------------------------------------------
    totals = {
        "total_loss": 0.0,
        "loss_cls": 0.0,
        "loss_anom": 0.0,
        "loss_rank": 0.0,
        "loss_smooth": 0.0,
        "loss_sparse": 0.0,
    }

    total_samples = 0

    normal_scores = []
    anomaly_scores = []

    correct = 0

    print("\n" + "-" * 70)
    print(
        f"{'VIDEO':<22}"
        f"{'TRUE':<18}"
        f"{'PRED':<18}"
        f"{'ANOMALY':>10}"
    )
    print("-" * 70)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    with torch.no_grad():

        for batch in dataloader:

            rgb = batch["rgb"].to(device)
            flow = batch["flow"].to(device)

            labels = batch["label_id"].to(device)
            is_anom = batch["is_anomaly"].to(device)

            outputs = model(
                rgb,
                flow
            )

            loss_dict = criterion(
                outputs,
                labels,
                is_anom
            )

            # -------------------------------------------------
            # Accumulate losses
            # -------------------------------------------------
            for key in totals:
                totals[key] += loss_dict[key].item()

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------
            predictions = torch.argmax(
                outputs["video_action_logits"],
                dim=-1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total_samples += len(labels)

            # -------------------------------------------------
            # Anomaly scores
            # -------------------------------------------------
            scores = outputs[
                "video_anomaly_score"
            ]

            for i in range(len(labels)):

                score = float(
                    scores[i].item()
                )

                label_id = int(
                    labels[i].item()
                )

                prediction_id = int(
                    predictions[i].item()
                )

                video_id = batch[
                    "video_id"
                ][i]

                class_names = {
                    0: "Normal",
                    1: "Fighting",
                    2: "Panic_Dispersal"
                }

                true_name = class_names.get(
                    label_id,
                    str(label_id)
                )

                pred_name = class_names.get(
                    prediction_id,
                    str(prediction_id)
                )

                print(
                    f"{video_id:<22}"
                    f"{true_name:<18}"
                    f"{pred_name:<18}"
                    f"{score:>10.4f}"
                )

                if int(is_anom[i].item()) == 1:
                    anomaly_scores.append(score)
                else:
                    normal_scores.append(score)

    # ---------------------------------------------------------
    # Average losses
    # ---------------------------------------------------------
    num_batches = max(
        1,
        len(dataloader)
    )

    print("\n" + "=" * 70)
    print("LOSS COMPONENT SUMMARY")
    print("=" * 70)

    print(
        f"Total loss       : "
        f"{totals['total_loss'] / num_batches:.6f}"
    )

    print(
        f"Classification   : "
        f"{totals['loss_cls'] / num_batches:.6f}"
    )

    print(
        f"Anomaly BCE      : "
        f"{totals['loss_anom'] / num_batches:.6f}"
    )

    print(
        f"MIL ranking      : "
        f"{totals['loss_rank'] / num_batches:.6f}"
    )

    print(
        f"Smoothness       : "
        f"{totals['loss_smooth'] / num_batches:.6f}"
    )

    print(
        f"Sparsity         : "
        f"{totals['loss_sparse'] / num_batches:.6f}"
    )

    # ---------------------------------------------------------
    # Classification accuracy
    # ---------------------------------------------------------
    accuracy = (
        correct /
        max(1, total_samples)
    )

    print(
        f"\nClassification accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Anomaly score statistics
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANOMALY SCORE SUMMARY")
    print("=" * 70)

    if normal_scores:
        print(
            f"Normal videos:"
        )
        print(
            f"  min  = {min(normal_scores):.4f}"
        )
        print(
            f"  mean = {sum(normal_scores) / len(normal_scores):.4f}"
        )
        print(
            f"  max  = {max(normal_scores):.4f}"
        )

    if anomaly_scores:
        print(
            f"\nAnomalous videos:"
        )
        print(
            f"  min  = {min(anomaly_scores):.4f}"
        )
        print(
            f"  mean = {sum(anomaly_scores) / len(anomaly_scores):.4f}"
        )
        print(
            f"  max  = {max(anomaly_scores):.4f}"
        )

    if normal_scores and anomaly_scores:

        normal_mean = (
            sum(normal_scores)
            / len(normal_scores)
        )

        anomaly_mean = (
            sum(anomaly_scores)
            / len(anomaly_scores)
        )

        print(
            f"\nMean anomaly-score gap: "
            f"{anomaly_mean - normal_mean:.6f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()