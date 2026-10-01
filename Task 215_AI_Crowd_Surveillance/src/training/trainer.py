"""
Training utilities for the weakly supervised crowd surveillance model.

Features:
- RGB + optical-flow model training
- Weakly supervised MIL loss
- Validation loss tracking
- Classification accuracy tracking
- Anomaly-score separation tracking
- Best-checkpoint selection using validation loss
- Early stopping
- Experiment history logging

Compatible with the existing scripts/train.py API:

    trainer = ModelTrainer(
        config=config,
        device=args.device
    )

    trainer.fit(
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"]
    )
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.optim import AdamW

from src.models.mil_classifier import TwoStreamWSSTALNet
from src.models.loss import WSSTALLoss


class ModelTrainer:

    def __init__(
        self,
        config: Dict[str, Any],
        device=None,
    ):
        """
        Initialize the model, loss function, optimizer,
        checkpoint directory, and training configuration.
        """

        self.config = config

        # =========================================================
        # DEVICE
        # =========================================================

        requested_device = device

        if requested_device is None:
            requested_device = config["runtime"].get(
                "device",
                "auto"
            )

        if requested_device == "auto":
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(
                requested_device
            )

        print(
            f"Initializing ModelTrainer on device: "
            f"{self.device}"
        )

        # =========================================================
        # MODEL
        # =========================================================

        model_cfg = config["model"]

        self.model = TwoStreamWSSTALNet(
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
        ).to(self.device)

        # =========================================================
        # TRAINING CONFIGURATION
        # =========================================================

        train_cfg = config["training"]

        self.epochs = train_cfg.get(
            "epochs",
            10
        )

        self.early_stopping_patience = train_cfg.get(
            "early_stopping_patience",
            5
        )

        # =========================================================
        # LOSS
        # =========================================================

        self.criterion = WSSTALLoss(
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
                1.0
            ),

            classification_weight=train_cfg.get(
                "classification_weight",
                2.0
            ),

            anomaly_weight=train_cfg.get(
                "anomaly_weight",
                1.0
            ),

            ranking_weight=train_cfg.get(
                "ranking_weight",
                0.5
            ),
        )

        # =========================================================
        # OPTIMIZER
        # =========================================================

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.get(
                "learning_rate",
                0.0005
            ),
            weight_decay=train_cfg.get(
                "weight_decay",
                0.0001
            ),
        )

        # =========================================================
        # CHECKPOINT DIRECTORY
        # =========================================================

        self.checkpoint_dir = Path(
            train_cfg.get(
                "checkpoint_dir",
                "checkpoints"
            )
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =========================================================
        # DATA LOADERS
        # =========================================================

        self.train_loader = None
        self.val_loader = None

    # =============================================================
    # TRAIN ONE EPOCH
    # =============================================================

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch and return averaged loss components.
        """

        if self.train_loader is None:
            raise RuntimeError(
                "train_loader has not been assigned. "
                "Pass train_loader to fit()."
            )

        self.model.train()

        totals = {
            "total_loss": 0.0,
            "loss_cls": 0.0,
            "loss_anom": 0.0,
            "loss_rank": 0.0,
            "loss_smooth": 0.0,
            "loss_sparse": 0.0,
        }

        batches = 0

        # ---------------------------------------------------------
        # BATCH LOOP
        # ---------------------------------------------------------

        for batch in self.train_loader:

            rgb = batch["rgb"].to(
                self.device,
                non_blocking=True
            )

            flow = batch["flow"].to(
                self.device,
                non_blocking=True
            )

            labels = batch["label_id"].to(
                self.device
            )

            anomaly_targets = batch["is_anomaly"].to(
                self.device
            )

            # -----------------------------------------------------
            # Clear gradients
            # -----------------------------------------------------

            self.optimizer.zero_grad()

            # -----------------------------------------------------
            # Forward pass
            # -----------------------------------------------------

            outputs = self.model(
                rgb,
                flow
            )

            # -----------------------------------------------------
            # Calculate weakly supervised loss
            # -----------------------------------------------------

            losses = self.criterion(
                outputs,
                labels,
                anomaly_targets
            )

            # -----------------------------------------------------
            # Backpropagation
            # -----------------------------------------------------

            losses["total_loss"].backward()

            self.optimizer.step()

            # -----------------------------------------------------
            # Accumulate loss components
            # -----------------------------------------------------

            for key in totals:
                totals[key] += losses[key].item()

            batches += 1

        # ---------------------------------------------------------
        # Prevent division by zero
        # ---------------------------------------------------------

        if batches == 0:
            return totals

        return {
            key: value / batches
            for key, value in totals.items()
        }

    # =============================================================
    # VALIDATION
    # =============================================================

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.

        Metrics:
        - validation total loss
        - classification loss
        - anomaly BCE
        - ranking loss
        - smoothness loss
        - sparsity loss
        - classification accuracy
        - mean normal anomaly score
        - mean anomalous anomaly score
        - anomaly-score separation gap
        """

        if self.val_loader is None:
            raise RuntimeError(
                "val_loader has not been assigned. "
                "Pass val_loader to fit()."
            )

        self.model.eval()

        totals = {
            "total_loss": 0.0,
            "loss_cls": 0.0,
            "loss_anom": 0.0,
            "loss_rank": 0.0,
            "loss_smooth": 0.0,
            "loss_sparse": 0.0,
        }

        batches = 0

        correct = 0
        total = 0

        normal_scores = []
        anomaly_scores = []

        # ---------------------------------------------------------
        # VALIDATION BATCH LOOP
        # ---------------------------------------------------------

        for batch in self.val_loader:

            rgb = batch["rgb"].to(
                self.device,
                non_blocking=True
            )

            flow = batch["flow"].to(
                self.device,
                non_blocking=True
            )

            labels = batch["label_id"].to(
                self.device
            )

            anomaly_targets = batch["is_anomaly"].to(
                self.device
            )

            # -----------------------------------------------------
            # Forward pass
            # -----------------------------------------------------

            outputs = self.model(
                rgb,
                flow
            )

            # -----------------------------------------------------
            # Calculate losses
            # -----------------------------------------------------

            losses = self.criterion(
                outputs,
                labels,
                anomaly_targets
            )

            for key in totals:
                totals[key] += losses[key].item()

            # -----------------------------------------------------
            # Classification prediction
            # -----------------------------------------------------

            predictions = torch.argmax(
                outputs["video_action_logits"],
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # -----------------------------------------------------
            # Anomaly scores
            # -----------------------------------------------------

            scores = (
                outputs["video_anomaly_score"]
                .detach()
                .cpu()
                .tolist()
            )

            targets = (
                anomaly_targets
                .detach()
                .cpu()
                .tolist()
            )

            for score, target in zip(
                scores,
                targets
            ):

                score = float(score)

                if target > 0.5:
                    anomaly_scores.append(
                        score
                    )
                else:
                    normal_scores.append(
                        score
                    )

            batches += 1

        # ---------------------------------------------------------
        # Empty validation protection
        # ---------------------------------------------------------

        if batches == 0:

            return {
                "val_loss": 0.0,
                "val_cls_loss": 0.0,
                "val_anom_loss": 0.0,
                "val_rank_loss": 0.0,
                "val_smooth_loss": 0.0,
                "val_sparse_loss": 0.0,
                "val_acc": 0.0,
                "normal_anomaly_mean": 0.0,
                "anomaly_anomaly_mean": 0.0,
                "anomaly_gap": 0.0,
            }

        # ---------------------------------------------------------
        # Average losses
        # ---------------------------------------------------------

        averages = {
            key: value / batches
            for key, value in totals.items()
        }

        # ---------------------------------------------------------
        # Classification accuracy
        # ---------------------------------------------------------

        val_acc = (
            correct / total
            if total > 0
            else 0.0
        )

        # ---------------------------------------------------------
        # Normal anomaly score
        # ---------------------------------------------------------

        normal_mean = (
            sum(normal_scores)
            / len(normal_scores)
            if normal_scores
            else 0.0
        )

        # ---------------------------------------------------------
        # Anomalous anomaly score
        # ---------------------------------------------------------

        anomaly_mean = (
            sum(anomaly_scores)
            / len(anomaly_scores)
            if anomaly_scores
            else 0.0
        )

        # ---------------------------------------------------------
        # Anomaly separation
        # ---------------------------------------------------------

        anomaly_gap = (
            anomaly_mean - normal_mean
        )

        return {
            "val_loss": averages["total_loss"],
            "val_cls_loss": averages["loss_cls"],
            "val_anom_loss": averages["loss_anom"],
            "val_rank_loss": averages["loss_rank"],
            "val_smooth_loss": averages["loss_smooth"],
            "val_sparse_loss": averages["loss_sparse"],
            "val_acc": val_acc,
            "normal_anomaly_mean": normal_mean,
            "anomaly_anomaly_mean": anomaly_mean,
            "anomaly_gap": anomaly_gap,
        }

    # =============================================================
    # SAVE CHECKPOINT
    # =============================================================

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """
        Save model, optimizer, epoch and validation metrics.
        """

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    self.model.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "metrics": metrics,
            },
            path,
        )

    # =============================================================
    # FIT
    # =============================================================

    def fit(
        self,
        train_loader=None,
        val_loader=None,
        epochs=None,
    ):
        """
        Train the model.
        """

        # ---------------------------------------------------------
        # Assign loaders
        # ---------------------------------------------------------

        if train_loader is not None:
            self.train_loader = train_loader

        if val_loader is not None:
            self.val_loader = val_loader

        if self.train_loader is None:
            raise RuntimeError(
                "No training DataLoader supplied."
            )

        if self.val_loader is None:
            raise RuntimeError(
                "No validation DataLoader supplied."
            )

        # ---------------------------------------------------------
        # Assign epoch count
        # ---------------------------------------------------------

        if epochs is not None:
            self.epochs = epochs

        print(
            f"Beginning training for "
            f"{self.epochs} epochs..."
        )

        history = []

        best_val_loss = float("inf")

        best_epoch = 0

        epochs_without_improvement = 0

        start_time = time.time()

        # =========================================================
        # EPOCH LOOP
        # =========================================================

        for epoch in range(
            1,
            self.epochs + 1
        ):

            epoch_start = time.time()

            # -----------------------------------------------------
            # Train
            # -----------------------------------------------------

            train_metrics = self.train_epoch()

            # -----------------------------------------------------
            # Validate
            # -----------------------------------------------------

            val_metrics = self.validate()

            epoch_time = (
                time.time()
                - epoch_start
            )

            # -----------------------------------------------------
            # Store history
            # -----------------------------------------------------

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "epoch_seconds": epoch_time,
            }

            history.append(record)

            # -----------------------------------------------------
            # Console output
            # -----------------------------------------------------

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"({epoch_time:.1f}s) - "
                f"Train Loss: "
                f"{train_metrics['total_loss']:.4f} | "
                f"Val Loss: "
                f"{val_metrics['val_loss']:.4f} | "
                f"Val Acc: "
                f"{val_metrics['val_acc'] * 100:.1f}% | "
                f"Anomaly Gap: "
                f"{val_metrics['anomaly_gap']:.4f}"
            )

            # -----------------------------------------------------
            # Latest checkpoint
            # -----------------------------------------------------

            latest_path = (
                self.checkpoint_dir
                / "latest_model.pt"
            )

            self.save_checkpoint(
                latest_path,
                epoch,
                val_metrics,
            )

            # -----------------------------------------------------
            # Best checkpoint
            # -----------------------------------------------------

            if (
                val_metrics["val_loss"]
                < best_val_loss
            ):

                best_val_loss = (
                    val_metrics["val_loss"]
                )

                best_epoch = epoch

                epochs_without_improvement = 0

                best_path = (
                    self.checkpoint_dir
                    / "best_model.pt"
                )

                self.save_checkpoint(
                    best_path,
                    epoch,
                    val_metrics,
                )

                print(
                    f"  * Saved new best checkpoint "
                    f"to {best_path} "
                    f"(Val Loss: "
                    f"{best_val_loss:.4f})"
                )

            else:

                epochs_without_improvement += 1

            # -----------------------------------------------------
            # Early stopping
            # -----------------------------------------------------

            if (
                epochs_without_improvement
                >= self.early_stopping_patience
            ):

                print(
                    f"  Early stopping triggered "
                    f"after "
                    f"{self.early_stopping_patience} "
                    f"epochs without validation-loss "
                    f"improvement."
                )

                break

        # =========================================================
        # FINAL SUMMARY
        # =========================================================

        total_time = (
            time.time()
            - start_time
        )

        print(
            f"Training completed in "
            f"{total_time:.1f}s."
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.4f}"
        )

        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        # =========================================================
        # SAVE EXPERIMENT LOG
        # =========================================================

        log_dir = Path(
            "reports/experiment_logs"
        )

        log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        log_path = (
            log_dir
            / f"experiment_{timestamp}.json"
        )

        experiment_summary = {
            "device": str(self.device),
            "epochs_requested": self.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "total_training_seconds": total_time,
            "history": history,
        }

        with open(
            log_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                experiment_summary,
                f,
                indent=2,
            )

        print(
            f"Log saved to "
            f"{log_path}"
        )

        return experiment_summary