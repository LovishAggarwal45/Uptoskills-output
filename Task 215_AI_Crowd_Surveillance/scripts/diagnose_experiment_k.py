"""
Diagnostic evaluation for Experiment K.

Checks:
1. Video-level predictions
2. Classification accuracy
3. Anomaly scores
4. Loss components
5. Normal-vs-anomalous score separation
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.data.dataset import get_dataloader
from src.models.mil_classifier import TwoStreamWSSTALNet
from src.models.loss import WSSTALLoss


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    config_path = "configs/experiment_k.yaml"
    checkpoint_path = "checkpoints/experiment_k/best_model.pt"
    split_path = "datasets/splits/val.csv"

    print("=" * 70)
    print("EXPERIMENT K CHECKPOINT DIAGNOSTIC")
    print("=" * 70)

    print(f"Split      : {split_path}")
    print(f"Checkpoint : {checkpoint_path}")

    config = load_config(config_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device     : {device}")
    print()

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model_cfg = config["model"]

    print("Initializing model...")

    model = TwoStreamWSSTALNet(
        num_classes=config["dataset"]["num_classes"],
        rgb_feature_dim=model_cfg.get(
            "rgb_feature_dim",
            512
        ),
        flow_feature_dim=model_cfg.get(
            "flow_feature_dim",
            128
        ),
        fusion_dim=model_cfg.get(
            "fusion_dim",
            256
        ),
        encoder_layers=model_cfg.get(
            "encoder_layers",
            2
        ),
        nheads=model_cfg.get(
            "nheads",
            4
        ),
        dropout=model_cfg.get(
            "dropout",
            0.15
        ),
        mil_top_k=model_cfg.get(
            "mil_top_k",
            3
        ),
        pretrained=model_cfg.get(
            "pretrained",
            True
        ),
        freeze_rgb_backbone=model_cfg.get(
            "freeze_rgb_backbone",
            False
        ),
    ).to(device)

    # ------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------

    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    print()

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    print("Loading validation data...")

    val_loader = get_dataloader(
        split_csv=split_path,
        root_dir=".",
        batch_size=1,
        num_segments=12,
        is_training=False,
        num_workers=0,
    )

    # ------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------

    train_cfg = config["training"]

    criterion = WSSTALLoss(
        smoothness_weight=train_cfg.get(
            "smoothness_weight",
            0.0001
        ),
        sparsity_weight=train_cfg.get(
            "sparsity_weight",
            0.0001
        ),
        ranking_margin=train_cfg.get(
            "ranking_margin",
            0.20
        ),
        classification_weight=train_cfg.get(
            "classification_weight",
            1.0
        ),
        anomaly_weight=train_cfg.get(
            "anomaly_weight",
            2.0
        ),
        ranking_weight=train_cfg.get(
            "ranking_weight",
            1.0
        ),
    )

    class_names = config["dataset"]["classes"]

    results = []

    normal_scores = []
    anomaly_scores = []

    correct = 0
    total = 0

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print(
        f"{'VIDEO':<22}"
        f"{'TRUE':<20}"
        f"{'PRED':<20}"
        f"{'ANOMALY':>10}"
    )
    print("-" * 70)

    with torch.no_grad():

        for batch in val_loader:

            rgb = batch["rgb"].to(device)
            flow = batch["flow"].to(device)

            labels = batch["label_id"].to(device)

            anomaly_targets = batch[
                "is_anomaly"
            ].to(device)

            outputs = model(
                rgb,
                flow
            )

            losses = criterion(
                outputs,
                labels,
                anomaly_targets
            )

            logits = outputs[
                "video_action_logits"
            ]

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            anomaly_score = (
                outputs[
                    "video_anomaly_score"
                ]
                .detach()
                .cpu()
                .item()
            )

            true_id = labels.cpu().item()
            pred_id = predictions.cpu().item()

            video_id = batch[
                "video_id"
            ][0]

            true_name = class_names[
                true_id
            ]

            pred_name = class_names[
                pred_id
            ]

            if true_id == pred_id:
                correct += 1

            total += 1

            if anomaly_targets.cpu().item() > 0.5:
                anomaly_scores.append(
                    anomaly_score
                )
            else:
                normal_scores.append(
                    anomaly_score
                )

            results.append({
                "video_id": video_id,
                "true": true_name,
                "pred": pred_name,
                "anomaly_score": anomaly_score,
                "probabilities": probabilities[
                    0
                ].cpu().tolist(),
            })

            print(
                f"{video_id:<22}"
                f"{true_name:<20}"
                f"{pred_name:<20}"
                f"{anomaly_score:>10.4f}"
            )

    # ------------------------------------------------------------
    # LOSS COMPONENTS
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("LOSS COMPONENT SUMMARY")
    print("=" * 70)

    # Recalculate average losses across validation set
    loss_totals = {
        "total_loss": 0.0,
        "classification": 0.0,
        "anomaly_bce": 0.0,
        "mil_ranking": 0.0,
        "smoothness": 0.0,
        "sparsity": 0.0,
    }

    batches = 0

    with torch.no_grad():

        for batch in val_loader:

            rgb = batch["rgb"].to(device)
            flow = batch["flow"].to(device)

            labels = batch["label_id"].to(device)

            anomaly_targets = batch[
                "is_anomaly"
            ].to(device)

            outputs = model(
                rgb,
                flow
            )

            losses = criterion(
                outputs,
                labels,
                anomaly_targets
            )

            loss_totals[
                "total_loss"
            ] += losses[
                "total_loss"
            ].item()

            loss_totals[
                "classification"
            ] += losses[
                "loss_cls"
            ].item()

            loss_totals[
                "anomaly_bce"
            ] += losses[
                "loss_anom"
            ].item()

            loss_totals[
                "mil_ranking"
            ] += losses[
                "loss_rank"
            ].item()

            loss_totals[
                "smoothness"
            ] += losses[
                "loss_smooth"
            ].item()

            loss_totals[
                "sparsity"
            ] += losses[
                "loss_sparse"
            ].item()

            batches += 1

    for key in loss_totals:
        loss_totals[key] /= batches

    print(
        f"Total loss       : "
        f"{loss_totals['total_loss']:.6f}"
    )

    print(
        f"Classification   : "
        f"{loss_totals['classification']:.6f}"
    )

    print(
        f"Anomaly BCE      : "
        f"{loss_totals['anomaly_bce']:.6f}"
    )

    print(
        f"MIL ranking      : "
        f"{loss_totals['mil_ranking']:.6f}"
    )

    print(
        f"Smoothness       : "
        f"{loss_totals['smoothness']:.6f}"
    )

    print(
        f"Sparsity         : "
        f"{loss_totals['sparsity']:.6f}"
    )

    # ------------------------------------------------------------
    # ACCURACY
    # ------------------------------------------------------------

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)

    print(
        f"Correct          : "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # ANOMALY SEPARATION
    # ------------------------------------------------------------

    normal_mean = (
        sum(normal_scores)
        / len(normal_scores)
        if normal_scores
        else 0.0
    )

    anomaly_mean = (
        sum(anomaly_scores)
        / len(anomaly_scores)
        if anomaly_scores
        else 0.0
    )

    gap = anomaly_mean - normal_mean

    print()
    print("=" * 70)
    print("ANOMALY SCORE SUMMARY")
    print("=" * 70)

    if normal_scores:
        print(
            "Normal videos:"
        )
        print(
            f"  min  = "
            f"{min(normal_scores):.4f}"
        )
        print(
            f"  mean = "
            f"{normal_mean:.4f}"
        )
        print(
            f"  max  = "
            f"{max(normal_scores):.4f}"
        )

    if anomaly_scores:
        print()
        print(
            "Anomalous videos:"
        )
        print(
            f"  min  = "
            f"{min(anomaly_scores):.4f}"
        )
        print(
            f"  mean = "
            f"{anomaly_mean:.4f}"
        )
        print(
            f"  max  = "
            f"{max(anomaly_scores):.4f}"
        )

    print()
    print(
        f"Mean anomaly-score gap: "
        f"{gap:.6f}"
    )

    # ------------------------------------------------------------
    # CLASS PROBABILITIES
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASS PROBABILITIES")
    print("=" * 70)

    for result in results:

        print()
        print(
            f"Video: "
            f"{result['video_id']}"
        )

        for class_name, probability in zip(
            class_names,
            result["probabilities"]
        ):
            print(
                f"  {class_name:<20}"
                f"{probability:.4f}"
            )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()